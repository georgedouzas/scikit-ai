"""Run component tasks on local Hugging Face models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    AnimationRequest,
    Audio,
    ClassificationRequest,
    DescriptionRequest,
    EmbeddingRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    Prompt,
    PromptedBackend,
    Segment,
    SpeechRequest,
    Transcript,
    TranscriptionRequest,
    Video,
    VideoGenerationRequest,
    build_image_prompt,
    import_optional,
)

CHAT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
VISION_MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
IMAGE_MODEL = 'stabilityai/sd-turbo'
CLAP_MODEL = 'laion/clap-htsat-unfused'
CLAP_SAMPLE_RATE = 48000
WAN_FPS = 16
SVD_FPS = 7
SVD_SIZE = (1024, 576)
AUDIO_INSTRUCTIONS = {
    'brief': 'Describe the sounds in this audio in one sentence.',
    'detailed': 'Describe the sounds, music and speech in this audio in one detailed paragraph.',
}
TURBO_OPTIONS = {'num_inference_steps': 1, 'guidance_scale': 0.0}
RETURN_TIMESTAMPS: dict[str | None, bool | str] = {None: False, 'segment': True, 'word': 'word'}


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


class HFBackend(PromptedBackend):
    """Run component tasks on open-source Hugging Face models on the local machine.

    Text classification uses a zero-shot natural language inference model and image classification uses CLIP, so both
    return real scores but ignore few-shot examples. Summarization, translation and image description prompt small
    instruction models. Models are downloaded on first use. The backend needs the `hf` extra.

    Args:
        model:
            The Hugging Face model id to use. The default model of the task is used when it is `None`.

        device:
            The device that runs the models, for example `'cpu'`, `'cuda'` or `'mps'`. It is chosen automatically when
            it is `None`.

        max_new_tokens:
            The maximum number of tokens that a prompted model generates.

        options:
            Extra keyword arguments passed to every pipeline call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {
        'classify_text': 'facebook/bart-large-mnli',
        'summarize': CHAT_MODEL,
        'translate': CHAT_MODEL,
        'describe_image': VISION_MODEL,
        'classify_image': 'openai/clip-vit-base-patch32',
        'embed_text': 'sentence-transformers/all-MiniLM-L6-v2',
        'synthesize_speech': 'facebook/mms-tts-eng',
        'generate_image': IMAGE_MODEL,
        'transcribe': 'openai/whisper-small',
        'classify_audio': CLAP_MODEL,
        'describe_audio': 'Qwen/Qwen2-Audio-7B-Instruct',
        'classify_video': VISION_MODEL,
        'describe_video': VISION_MODEL,
        'extract_fields': CHAT_MODEL,
        'embed_image': 'clip-ViT-B-32',
        'embed_audio': CLAP_MODEL,
        'generate_video': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        'edit_image': 'timbrooks/instruct-pix2pix',
        'animate_image': 'stabilityai/stable-video-diffusion-img2vid-xt',
    }
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})
    accepts_context: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str | None = None,
        max_new_tokens: int = 512,
        options: dict | None = None,
        max_concurrency: int = 1,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _load(self, key: tuple[str, str], factory: Callable[[], Any]) -> Any:
        """Return a cached model object, creating it on first use."""
        if not hasattr(self, '_models'):
            self._models: dict[tuple[str, str], Any] = {}
        if key not in self._models:
            self._models[key] = factory()
        return self._models[key]

    def _pipeline(self, task: str, model: str) -> Any:
        """Return the cached `transformers` pipeline of a task and a model."""
        transformers = import_optional('transformers', 'hf')
        return self._load((task, model), lambda: transformers.pipeline(task, model=model, device=self.device))

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

    def connect(self) -> 'HFBackend':
        """Check that the `hf` extra is installed.

        Returns:
            The backend.
        """
        import_optional('transformers', 'hf')
        return self

    def complete(self, prompt: Prompt, model: str) -> str:
        """Return the answer of a local chat or vision model to a prompt.

        Args:
            prompt:
                The prompt.

            model:
                The model to prompt.

        Returns:
            The answer of the model.
        """
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
        audio = request.audio.to_mono()
        inputs = {'raw': audio.samples.astype(np.float32), 'sampling_rate': audio.sample_rate}
        pipeline = self._pipeline('automatic-speech-recognition', self.resolve_model('transcribe'))
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

    def describe_audio(self, request: DescriptionRequest) -> str:
        """Describe audio with an audio-language model.

        Examples of the request are not used.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        assert isinstance(request.sample, Audio)
        transformers = import_optional('transformers', 'hf')
        model = self.resolve_model('describe_audio')
        processor = self._load(('audio-processor', model), lambda: transformers.AutoProcessor.from_pretrained(model))
        audio_class = transformers.Qwen2AudioForConditionalGeneration
        language_model = self._load(
            ('audio-model', model),
            lambda: audio_class.from_pretrained(model, device_map=self.device or 'auto'),
        )
        extra = f' {request.instructions}' if request.instructions else ''
        instructions = AUDIO_INSTRUCTIONS[request.detail] + extra
        content = [{'type': 'audio', 'audio_url': 'sample'}, {'type': 'text', 'text': instructions}]
        conversation = [{'role': 'user', 'content': content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        rate = processor.feature_extractor.sampling_rate
        samples = [request.sample.resample(rate).samples]
        inputs = processor(text=text, audio=samples, sampling_rate=rate, return_tensors='pt')
        output = language_model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = output[:, inputs['input_ids'].shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

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
