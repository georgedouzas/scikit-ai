"""Run component tasks on local Hugging Face models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    AnimationRequest,
    Audio,
    AudioGenerationRequest,
    AudioOperationRequest,
    ClassificationRequest,
    DiarizationRequest,
    EmbeddingRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOperationRequest,
    Prompt,
    PromptedBackend,
    Region,
    RegionRequest,
    Segment,
    SpeechRequest,
    Transcript,
    TranscriptionRequest,
    Video,
    VideoGenerationRequest,
    build_image_prompt,
    cut_audio,
    import_optional,
    read_env,
)

CHAT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
VISION_MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
IMAGE_MODEL = 'stabilityai/sd-turbo'
CLAP_MODEL = 'laion/clap-htsat-unfused'
CLAP_SAMPLE_RATE = 48000
WAN_FPS = 16
SVD_FPS = 7
SVD_SIZE = (1024, 576)
AUDIO_LANGUAGE_MODEL = 'Qwen/Qwen2-Audio-7B-Instruct'
TURBO_OPTIONS = {'num_inference_steps': 1, 'guidance_scale': 0.0}
RETURN_TIMESTAMPS: dict[str | None, bool | str] = {None: False, 'segment': True, 'word': 'word'}
TRANSCRIPTION_MODEL = 'openai/whisper-small'
DETECTION_MODEL = 'google/owlv2-base-patch16-ensemble'
UPSCALE_MODELS = {2: 'caidas/swin2SR-classical-sr-x2-64', 4: 'caidas/swin2SR-classical-sr-x4-64'}
BACKGROUND_MODEL = 'briaai/RMBG-1.4'
BACKGROUND_MODEL_REVISION = '2ceba5a5efaec153162aedea169f76caf9b46cf8'
SEPARATION_MODEL = 'HDEMUCS_HIGH_MUSDB_PLUS'
ACCOMPANIMENT_STEMS = ('drums', 'bass', 'other')
MUSICGEN_TOKENS_PER_SECOND = 50
AUDIOLDM_SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256
NOISE_PERCENTILE = 10
NOISE_STD_FACTOR = 1.5


def _to_frames(frames: Any) -> NDArray[np.uint8]:
    """Turn the frames of a diffusion pipeline, PIL images or float arrays, into RGB `uint8` frames."""
    stacked = np.stack(
        [np.asarray(frame.convert('RGB')) if isinstance(frame, Image.Image) else np.asarray(frame) for frame in frames],
    )
    if stacked.dtype != np.uint8:
        stacked = (np.clip(stacked, 0, 1) * 255).round().astype(np.uint8)
    return stacked


def _to_segments(chunks: list[dict[str, Any]], duration: float) -> tuple[Segment, ...]:
    """Read the timed chunks of a speech recognition pipeline."""
    segments = []
    for chunk in chunks:
        start, end = chunk['timestamp']
        segments.append(Segment(float(start), float(duration if end is None else end), chunk['text'].strip()))
    return tuple(segments)


def _clamp_box(box: dict[str, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Round the box of a detection pipeline to pixels inside the image."""
    width, height = size
    left, top = max(0, round(box['xmin'])), max(0, round(box['ymin']))
    return left, top, min(width, round(box['xmax'])), min(height, round(box['ymax']))


def _to_mono_samples(samples: Any, length: int) -> NDArray[np.float32]:
    """Mix generated audio of any channel layout to mono, trimmed or padded with silence to a number of samples."""
    array = np.asarray(samples, dtype=np.float32)
    mono = array.reshape(-1, array.shape[-1]).mean(axis=0)[:length]
    return np.pad(mono, (0, length - mono.shape[0])).astype(np.float32)


def _frame(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Split samples, padded by reflection with one frame on each side, into overlapping frames."""
    padded = np.pad(samples, FRAME_LENGTH, mode='reflect')
    count = 1 + math.ceil((padded.shape[0] - FRAME_LENGTH) / HOP_LENGTH)
    padded = np.pad(padded, (0, (count - 1) * HOP_LENGTH + FRAME_LENGTH - padded.shape[0]))
    index = np.arange(FRAME_LENGTH)[None, :] + HOP_LENGTH * np.arange(count)[:, None]
    return padded[index]


def _overlap_add(frames: NDArray[np.floating], window: NDArray[np.floating]) -> NDArray[np.floating]:
    """Join windowed frames back into samples, normalizing by the overlapping window energy."""
    length = (frames.shape[0] - 1) * HOP_LENGTH + FRAME_LENGTH
    output, energy = np.zeros(length), np.zeros(length)
    for number, frame in enumerate(frames):
        start = number * HOP_LENGTH
        output[start : start + FRAME_LENGTH] += frame * window
        energy[start : start + FRAME_LENGTH] += window**2
    return output / np.maximum(energy, np.finfo(float).eps)


def _reduce_noise(samples: NDArray[np.floating]) -> NDArray[np.float32]:
    """Gate the noise of mono samples with a soft spectral mask learned from the quietest frames."""
    if samples.shape[0] <= FRAME_LENGTH:
        return samples.astype(np.float32)
    window = np.hanning(FRAME_LENGTH)
    frames = _frame(samples.astype(np.float64))
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum)
    power = (magnitude**2).mean(axis=1)
    quiet = magnitude[power <= np.percentile(power, NOISE_PERCENTILE)]
    threshold = quiet.mean(axis=0) + NOISE_STD_FACTOR * quiet.std(axis=0)
    gain = magnitude**4 / np.maximum(magnitude**4 + threshold**4, np.finfo(float).eps)
    denoised = _overlap_add(np.fft.irfft(spectrum * gain, n=FRAME_LENGTH, axis=1), window)
    return denoised[FRAME_LENGTH : FRAME_LENGTH + samples.shape[0]].astype(np.float32)


class HFBackend(PromptedBackend):
    """Run component tasks on open-source Hugging Face models on the local machine.

    Text classification uses a zero-shot natural language inference model and image classification uses CLIP, so both
    return real scores but ignore few-shot examples. Text generation and extraction prompt small instruction and
    vision models, and audio text generation uses an audio-language model. Object detection uses OWLv2, and
    segmentation finds boxes with OWLv2 and masks with SAM. Diarization uses pyannote, which needs the `diarize` extra
    and a Hugging Face token that has accepted the model conditions, and transcribes every speaker turn with Whisper.
    Audio separation uses the Hybrid Demucs bundle of torchaudio, and the `model` parameter names another bundle.
    Audio enhancement is spectral-gating noise reduction with no model. Models are downloaded on first use. The
    backend needs the `hf` extra.

    Background removal runs the code of `briaai/RMBG-1.4` that is downloaded from the Hugging Face Hub, pinned to the
    commit `BACKGROUND_MODEL_REVISION`. Another model set with `model` loads without remote code.

    Args:
        model:
            The Hugging Face model id to use. The default model of the task is used when it is `None`.

        device:
            The device that runs the models, for example `'cpu'`, `'cuda'` or `'mps'`. It is chosen automatically when
            it is `None`.

        max_new_tokens:
            The maximum number of tokens that a prompted model generates.

        token_env:
            The name of the environment variable that holds the Hugging Face token of gated models.

        options:
            Extra keyword arguments passed to every pipeline call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {
        'classify_text': 'facebook/bart-large-mnli',
        'generate_text': CHAT_MODEL,
        'generate_text:image': VISION_MODEL,
        'generate_text:video': VISION_MODEL,
        'generate_text:multimodal': VISION_MODEL,
        'generate_text:audio': AUDIO_LANGUAGE_MODEL,
        'extract_fields:audio': AUDIO_LANGUAGE_MODEL,
        'classify_multimodal:audio': AUDIO_LANGUAGE_MODEL,
        'find_moments': VISION_MODEL,
        'find_moments:audio': AUDIO_LANGUAGE_MODEL,
        'classify_multimodal': VISION_MODEL,
        'transcribe_image': VISION_MODEL,
        'pick_thumbnail': VISION_MODEL,
        'extract_fields:image': VISION_MODEL,
        'extract_fields:video': VISION_MODEL,
        'extract_fields:multimodal': VISION_MODEL,
        'classify_image': 'openai/clip-vit-base-patch32',
        'embed_text': 'sentence-transformers/all-MiniLM-L6-v2',
        'synthesize_speech': 'facebook/mms-tts-eng',
        'generate_image': IMAGE_MODEL,
        'transcribe': TRANSCRIPTION_MODEL,
        'classify_audio': CLAP_MODEL,
        'classify_video': VISION_MODEL,
        'extract_fields': CHAT_MODEL,
        'embed_image': 'clip-ViT-B-32',
        'embed_audio': CLAP_MODEL,
        'generate_video': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        'edit_image': 'timbrooks/instruct-pix2pix',
        'animate_image': 'stabilityai/stable-video-diffusion-img2vid-xt',
        'detect_objects': DETECTION_MODEL,
        'segment_image': 'facebook/sam-vit-base',
        'diarize': 'pyannote/speaker-diarization-3.1',
        'separate_audio': SEPARATION_MODEL,
        'upscale_image': UPSCALE_MODELS[2],
        'remove_background': BACKGROUND_MODEL,
        'generate_music': 'facebook/musicgen-small',
        'generate_sound': 'cvssp/audioldm2',
    }
    accepts_audio: ClassVar[bool] = True
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})
    accepts_context: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str | None = None,
        max_new_tokens: int = 512,
        token_env: str = 'HF_TOKEN',  # noqa: S107 - the default names an environment variable
        options: dict | None = None,
        max_concurrency: int = 1,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.token_env = token_env

    def _load(self, key: tuple[str, str], factory: Callable[[], Any]) -> Any:
        """Return a cached model object, creating it on first use."""
        if not hasattr(self, '_models'):
            self._models: dict[tuple[str, str], Any] = {}
        if key not in self._models:
            self._models[key] = factory()
        return self._models[key]

    def _pipeline(self, task: str, model: str, **kwargs: Any) -> Any:
        """Return the cached `transformers` pipeline of a task and a model, created with extra keyword arguments."""
        transformers = import_optional('transformers', 'hf')
        return self._load(
            (task, model),
            lambda: transformers.pipeline(task, model=model, device=self.device, **kwargs),
        )

    def _load_diffusion(self, task: str, class_name: str) -> Any:
        """Return the cached diffusion pipeline of a task, on the backend's device."""
        diffusers = import_optional('diffusers', 'hf')
        model = self.resolve_model(task)
        return self._load(
            (task, model),
            lambda: getattr(diffusers, class_name).from_pretrained(model).to(self.device or 'cpu'),
        )

    def _generator(self, seed: int | None) -> Any:
        """Return a seeded torch generator, or `None` without a seed."""
        torch = import_optional('torch', 'hf')
        return None if seed is None else torch.Generator().manual_seed(seed)

    def _encode_audio(self, audio: Audio) -> NDArray[np.floating]:
        """Embed audio with the CLAP model of the audio embedding task."""
        transformers = import_optional('transformers', 'hf')
        model = self.resolve_model('embed_audio')
        processor = self._load(('clap-processor', model), lambda: transformers.ClapProcessor.from_pretrained(model))
        clap = self._load(('clap-model', model), lambda: transformers.ClapModel.from_pretrained(model))
        samples = audio.resample(CLAP_SAMPLE_RATE).samples
        inputs = processor(audio=samples, sampling_rate=CLAP_SAMPLE_RATE, return_tensors='pt')
        output = clap.get_audio_features(**inputs)
        features = getattr(output, 'pooler_output', output)
        return features[0].detach().cpu().numpy()

    def _transcribe(self, request: TranscriptionRequest, model: str) -> Transcript:
        """Transcribe speech with a speech recognition model."""
        audio = request.audio.to_mono()
        inputs = {'raw': audio.samples.astype(np.float32), 'sampling_rate': audio.sample_rate}
        pipeline = self._pipeline('automatic-speech-recognition', model)
        generate_kwargs: dict[str, Any] = {'language': request.language} if request.language else {}
        if request.vocabulary:
            vocabulary = ', '.join(request.vocabulary)
            generate_kwargs['prompt_ids'] = pipeline.tokenizer.get_prompt_ids(vocabulary, return_tensors='pt')
        output = pipeline(
            inputs,
            return_timestamps=RETURN_TIMESTAMPS[request.timestamps],
            generate_kwargs=generate_kwargs,
            **self.call_options,
        )
        segments = _to_segments(output.get('chunks', []), audio.duration) if request.timestamps else ()
        return Transcript(output['text'].strip(), segments)

    def _find_regions(self, request: RegionRequest, model: str) -> list[Region]:
        """Find the boxes of the labels, or of the prompt, with a zero-shot object detection model."""
        labels = list(request.labels) or [request.prompt or 'object']
        pipeline = self._pipeline('zero-shot-object-detection', model)
        output = pipeline(request.image.convert('RGB'), candidate_labels=labels, **self.call_options)
        return [
            Region(str(item['label']), float(item['score']), _clamp_box(item['box'], request.image.size))
            for item in output
        ]

    def _load_diarization(self, model: str) -> Any:
        """Return the cached pyannote pipeline, loaded with the token read from `token_env`."""
        pyannote_audio = import_optional('pyannote.audio', 'diarize')

        def load() -> Any:
            """Load the pipeline and move it to the backend's device."""
            pipeline = pyannote_audio.Pipeline.from_pretrained(model, use_auth_token=read_env(self.token_env))
            if self.device is not None:
                pipeline.to(import_optional('torch', 'hf').device(self.device))
            return pipeline

        return self._load(('diarize', model), load)

    def _separate(self, audio: Audio, stem: str) -> NDArray[np.float32]:
        """Return a stem of mono audio separated by a torchaudio source separation bundle, at the bundle's rate."""
        torch = import_optional('torch', 'hf')
        torchaudio = import_optional('torchaudio', 'hf')
        name = self.resolve_model('separate_audio')
        bundle = getattr(torchaudio.pipelines, name)
        separator = self._load(('separate_audio', name), lambda: bundle.get_model().to(self.device or 'cpu'))
        samples = audio.resample(bundle.sample_rate).samples.astype(np.float32)
        mean, std = float(samples.mean()), float(samples.std()) or 1.0
        waveform = torch.from_numpy(np.stack([(samples - mean) / std] * 2)[None]).to(self.device or 'cpu')
        with torch.no_grad():
            sources = separator(waveform)[0].cpu().numpy() * std + mean
        stems = dict(zip(separator.sources, sources, strict=True))
        separated = stems['vocals'] if stem == 'vocals' else sum(stems[part] for part in ACCOMPANIMENT_STEMS)
        return np.asarray(separated, dtype=np.float32).mean(axis=0)

    def connect(self) -> 'HFBackend':
        """Check that the `hf` extra is installed.

        Returns:
            The backend.
        """
        import_optional('transformers', 'hf')
        return self

    def _complete_audio(self, prompt: Prompt, model: str) -> str:
        """Return the answer of a local audio-language model to a prompt with audio, whose images are left out."""
        transformers = import_optional('transformers', 'hf')
        processor = self._load(('audio-processor', model), lambda: transformers.AutoProcessor.from_pretrained(model))
        audio_class = transformers.Qwen2AudioForConditionalGeneration
        language_model = self._load(
            ('audio-model', model),
            lambda: audio_class.from_pretrained(model, device_map=self.device or 'auto'),
        )
        content: list[dict[str, Any]] = [
            {'type': 'audio', 'audio_url': f'audio {index}'} for index in range(len(prompt.audio))
        ]
        content.append({'type': 'text', 'text': f'{prompt.system}\n\n{prompt.user}'})
        text = processor.apply_chat_template(
            [{'role': 'user', 'content': content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        rate = processor.feature_extractor.sampling_rate
        samples = [audio.resample(rate).samples for audio in prompt.audio]
        inputs = processor(text=text, audio=samples, sampling_rate=rate, return_tensors='pt')
        output = language_model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = output[:, inputs['input_ids'].shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def complete(self, prompt: Prompt, model: str) -> str:
        """Return the answer of a local chat, vision or audio-language model to a prompt.

        Args:
            prompt:
                The prompt.

            model:
                The model to prompt.

        Returns:
            The answer of the model.
        """
        if prompt.audio:
            return self._complete_audio(prompt, model)
        options = {'max_new_tokens': self.max_new_tokens, 'return_full_text': False, **self.call_options}
        if not prompt.images:
            messages = [{'role': 'system', 'content': prompt.system}, {'role': 'user', 'content': prompt.user}]
            return self._pipeline('text-generation', model)(messages, **options)[0]['generated_text']
        content: list[dict[str, Any]] = [{'type': 'image', 'image': image} for image in prompt.images]
        content.append({'type': 'text', 'text': f'{prompt.system}\n\n{prompt.user}'})
        output = self._pipeline('image-text-to-text', model)(text=[{'role': 'user', 'content': content}], **options)
        return output[0]['generated_text']

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a text with a zero-shot classification model.

        Label descriptions are used as the candidate labels when they are given.

        Args:
            request:
                The classification request.

        Returns:
            A score per label.
        """
        candidates = [request.descriptions.get(label, label) for label in request.labels]
        pipeline = self._pipeline('zero-shot-classification', self.resolve_model('classify_text'))
        output = pipeline(request.sample, candidate_labels=candidates, multi_label=request.multi_label)
        scores = dict(zip(output['labels'], output['scores'], strict=True))
        return {label: float(scores[candidate]) for label, candidate in zip(request.labels, candidates, strict=True)}

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Score an image with a zero-shot image classification model.

        Label descriptions are used as the candidate labels when they are given.

        Args:
            request:
                The classification request.

        Returns:
            A score per label.
        """
        candidates = [request.descriptions.get(label, label) for label in request.labels]
        pipeline = self._pipeline('zero-shot-image-classification', self.resolve_model('classify_image'))
        scores = {item['label']: item['score'] for item in pipeline(request.sample, candidate_labels=candidates)}
        return {label: float(scores[candidate]) for label, candidate in zip(request.labels, candidates, strict=True)}

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed a text with a sentence-transformers model.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        sentence_transformers = import_optional('sentence_transformers', 'hf')
        model = self.resolve_model('embed_text')
        encoder = self._load(
            ('embed_text', model),
            lambda: sentence_transformers.SentenceTransformer(model, device=self.device),
        )
        return np.asarray(encoder.encode([request.sample], **self.call_options)[0])

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Speak a text with a text-to-speech model.

        The default model has a single voice, so the voice of the request is ignored.

        Args:
            request:
                The speech request.

        Returns:
            The speech.
        """
        output = self._pipeline('text-to-speech', self.resolve_model('synthesize_speech'))(
            request.text,
            **self.call_options,
        )
        return Audio(np.asarray(output['audio'], dtype=np.float32).squeeze(), int(output['sampling_rate']))

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Generate an image with a diffusion model.

        Args:
            request:
                The image generation request.

        Returns:
            The image.
        """
        model = self.resolve_model('generate_image')
        pipeline = self._load_diffusion('generate_image', 'AutoPipelineForText2Image')
        generator = self._generator(request.seed)
        width, height = request.size
        output = pipeline(
            prompt=build_image_prompt(request),
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            generator=generator,
            **(TURBO_OPTIONS if model == IMAGE_MODEL else {}),
            **self.call_options,
        )
        return output.images[0]

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Transcribe speech with a speech recognition model.

        Args:
            request:
                The transcription request.

        Returns:
            The transcript.
        """
        return self._transcribe(request, self.resolve_model('transcribe'))

    def classify_audio(self, request: ClassificationRequest) -> dict[str, float]:
        """Score audio with a zero-shot audio classification model.

        Label descriptions are used as the candidate labels when they are given.

        Args:
            request:
                The classification request.

        Returns:
            A score per label.
        """
        assert isinstance(request.sample, Audio)
        candidates = [request.descriptions.get(label, label) for label in request.labels]
        pipeline = self._pipeline('zero-shot-audio-classification', self.resolve_model('classify_audio'))
        samples = request.sample.resample(CLAP_SAMPLE_RATE).samples
        scores = {item['label']: item['score'] for item in pipeline(samples, candidate_labels=candidates)}
        return {label: float(scores[candidate]) for label, candidate in zip(request.labels, candidates, strict=True)}

    def embed_image(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed an image with a CLIP model of sentence-transformers.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        sentence_transformers = import_optional('sentence_transformers', 'hf')
        model = self.resolve_model('embed_image')
        encoder = self._load(
            ('embed_image', model),
            lambda: sentence_transformers.SentenceTransformer(model, device=self.device),
        )
        return np.asarray(encoder.encode([request.sample], **self.call_options)[0])

    def embed_audio(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed audio with a CLAP model.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        assert isinstance(request.sample, Audio)
        return np.asarray(self._encode_audio(request.sample))

    def generate_video(self, request: VideoGenerationRequest) -> Video:
        """Generate a video with a text-to-video diffusion model.

        Args:
            request:
                The video generation request.

        Returns:
            The video, at `fps` or 16 frames per second.
        """
        pipeline = self._load_diffusion('generate_video', 'DiffusionPipeline')
        fps = request.fps or WAN_FPS
        width, height = request.size
        output = pipeline(
            prompt=build_image_prompt(ImageGenerationRequest(request.prompt, request.size, request.style)),
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_frames=4 * max(1, round(request.duration * fps / 4)) + 1,
            generator=self._generator(request.seed),
            **self.call_options,
        )
        return Video(_to_frames(output.frames[0]), fps)

    def edit_image(self, request: ImageEditRequest) -> Image.Image:
        """Edit an image with an instruction-following diffusion model.

        Args:
            request:
                The image edit request.

        Returns:
            The edited image, of the size of the input.
        """
        pipeline = self._load_diffusion('edit_image', 'StableDiffusionInstructPix2PixPipeline')
        output = pipeline(
            prompt=request.instruction,
            image=request.image.convert('RGB'),
            generator=self._generator(request.seed),
            **self.call_options,
        )
        edited = output.images[0]
        return edited if edited.size == request.image.size else edited.resize(request.image.size)

    def animate_image(self, request: AnimationRequest) -> Video:
        """Animate an image with an image-to-video diffusion model.

        The motion prompt is not used.

        Args:
            request:
                The animation request.

        Returns:
            The video, resized to the size of the image.
        """
        pipeline = self._load_diffusion('animate_image', 'StableVideoDiffusionPipeline')
        fps = request.fps or SVD_FPS
        output = pipeline(
            request.image.convert('RGB').resize(SVD_SIZE),
            num_frames=max(1, round(request.duration * fps)),
            generator=self._generator(request.seed),
            **self.call_options,
        )
        return Video(_to_frames(output.frames[0]), fps).resize(request.image.size)

    def detect_objects(self, request: RegionRequest) -> list[Region]:
        """Find the boxes of regions with a zero-shot object detection model.

        The labels are the candidate labels, or the prompt when there are no labels.

        Args:
            request:
                The region request.

        Returns:
            The regions, without masks.
        """
        return self._find_regions(request, self.resolve_model('detect_objects'))

    def segment_image(self, request: RegionRequest) -> list[Region]:
        """Find the masks of regions with SAM, prompted by the boxes of the default object detection model.

        Args:
            request:
                The region request.

        Returns:
            The regions, with masks of the image size.
        """
        regions = self._find_regions(request, DETECTION_MODEL)
        if not regions:
            return []
        torch = import_optional('torch', 'hf')
        transformers = import_optional('transformers', 'hf')
        model = self.resolve_model('segment_image')
        device = self.device or 'cpu'
        processor = self._load(('sam-processor', model), lambda: transformers.SamProcessor.from_pretrained(model))
        sam = self._load(('sam-model', model), lambda: transformers.SamModel.from_pretrained(model).to(device))
        boxes = [[list(region.box) for region in regions]]
        inputs = processor(request.image.convert('RGB'), input_boxes=boxes, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = sam(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs['original_sizes'],
            inputs['reshaped_input_sizes'],
        )[0]
        return [
            Region(region.label, region.score, region.box, Image.fromarray(np.asarray(mask[0]).astype(np.uint8) * 255))
            for region, mask in zip(regions, masks, strict=True)
        ]

    def diarize(self, request: DiarizationRequest) -> Transcript:
        """Find speaker turns with pyannote and transcribe every turn with the default Whisper model.

        Args:
            request:
                The diarization request.

        Returns:
            The transcript, with a speaker per segment.
        """
        pipeline = self._load_diarization(self.resolve_model('diarize'))
        torch = import_optional('torch', 'hf')
        audio = request.audio.to_mono()
        waveform = torch.from_numpy(audio.samples.astype(np.float32)[None, :])
        annotation = pipeline({'waveform': waveform, 'sample_rate': audio.sample_rate})
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            part = cut_audio(audio, turn.start, turn.end)
            turn_request = TranscriptionRequest(part, request.language, None, request.vocabulary)
            text = self._transcribe(turn_request, TRANSCRIPTION_MODEL).text if part.samples.size else ''
            segments.append(Segment(float(turn.start), float(turn.end), text, str(speaker)))
        return Transcript(' '.join(segment.text for segment in segments if segment.text), tuple(segments))

    def separate_audio(self, request: AudioOperationRequest) -> Audio:
        """Separate the vocals, or the sum of drums, bass and other as the accompaniment, with Hybrid Demucs.

        Args:
            request:
                The audio operation request.

        Returns:
            The mono stem, of the length and rate of the input.
        """
        mono = request.audio.to_mono()
        rate = getattr(import_optional('torchaudio', 'hf').pipelines, self.resolve_model('separate_audio')).sample_rate
        stem = Audio(self._separate(mono, request.stem), rate).resample(mono.sample_rate)
        return Audio(_to_mono_samples(stem.samples, mono.samples.shape[0]), mono.sample_rate)

    def enhance_audio(self, request: AudioOperationRequest) -> Audio:
        """Remove stationary noise with spectral gating, learning the noise from the quietest frames.

        Args:
            request:
                The audio operation request.

        Returns:
            The denoised mono audio, of the length and rate of the input.
        """
        mono = request.audio.to_mono()
        return Audio(_reduce_noise(mono.samples), mono.sample_rate)

    def upscale_image(self, request: ImageOperationRequest) -> Image.Image:
        """Upscale an image with Swin2SR, using the x2 model and a resize for scales other than 2 and 4.

        Args:
            request:
                The image operation request.

        Returns:
            The upscaled image, exactly `scale` times the size of the input.
        """
        image = request.image.convert('RGB')
        size = (image.width * request.scale, image.height * request.scale)
        if request.scale == 1:
            return image
        model = self.model or UPSCALE_MODELS.get(request.scale, UPSCALE_MODELS[2])
        upscaled = self._pipeline('image-to-image', model)(image, **self.call_options)
        return upscaled if upscaled.size == size else upscaled.resize(size)

    def remove_background(self, request: ImageOperationRequest) -> Image.Image:
        """Remove the background with the mask of an image segmentation model, by default RMBG at a pinned commit.

        Args:
            request:
                The image operation request.

        Returns:
            An RGBA image of the input size whose alpha is the predicted mask.
        """
        model = self.resolve_model('remove_background')
        image = request.image.convert('RGB')
        if model == BACKGROUND_MODEL:
            pipeline = self._pipeline(
                'image-segmentation',
                model,
                trust_remote_code=True,
                revision=BACKGROUND_MODEL_REVISION,
            )
            mask = pipeline(image, return_mask=True, **self.call_options).convert('L')
        else:
            output = self._pipeline('image-segmentation', model)(image, **self.call_options)
            masks = [np.asarray(item['mask'].convert('L')) for item in output]
            mask = Image.fromarray(np.max(masks, axis=0).astype(np.uint8)) if masks else Image.new('L', image.size)
        cutout = image.convert('RGBA')
        cutout.putalpha(mask if mask.size == image.size else mask.resize(image.size))
        return cutout

    def generate_music(self, request: AudioGenerationRequest) -> Audio:
        """Generate music with MusicGen, about 50 tokens per second, trimmed or padded to the duration.

        Args:
            request:
                The audio generation request.

        Returns:
            The mono music.
        """
        transformers = import_optional('transformers', 'hf')
        if request.seed is not None:
            transformers.set_seed(request.seed)
        pipeline = self._pipeline('text-to-audio', self.resolve_model('generate_music'))
        tokens = max(1, math.ceil(request.duration * MUSICGEN_TOKENS_PER_SECOND))
        output = pipeline(
            request.prompt,
            forward_params={'do_sample': True, 'max_new_tokens': tokens},
            **self.call_options,
        )
        rate = int(output['sampling_rate'])
        return Audio(_to_mono_samples(output['audio'], round(request.duration * rate)), rate)

    def generate_sound(self, request: AudioGenerationRequest) -> Audio:
        """Generate a sound with AudioLDM2, trimmed or padded to the duration.

        Args:
            request:
                The audio generation request.

        Returns:
            The mono sound at 16 kHz.
        """
        pipeline = self._load_diffusion('generate_sound', 'AudioLDM2Pipeline')
        output = pipeline(
            request.prompt,
            audio_length_in_s=request.duration,
            generator=self._generator(request.seed),
            **self.call_options,
        )
        length = round(request.duration * AUDIOLDM_SAMPLE_RATE)
        return Audio(_to_mono_samples(output.audios[0], length), AUDIOLDM_SAMPLE_RATE)
