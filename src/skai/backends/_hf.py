"""Run component tasks on local Hugging Face models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    Audio,
    ClassificationRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    Prompt,
    PromptedBackend,
    Segment,
    SpeechRequest,
    Transcript,
    TranscriptionRequest,
    build_image_prompt,
    import_optional,
)

CHAT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
VISION_MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
IMAGE_MODEL = 'stabilityai/sd-turbo'
TURBO_OPTIONS = {'num_inference_steps': 1, 'guidance_scale': 0.0}
RETURN_TIMESTAMPS: dict[str | None, bool | str] = {None: False, 'segment': True, 'word': 'word'}


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
    }
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})

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
        return np.asarray(encoder.encode([request.text], **self.call_options)[0])

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
        diffusers = import_optional('diffusers', 'hf')
        torch = import_optional('torch', 'hf')
        model = self.resolve_model('generate_image')
        pipeline = self._load(
            ('generate_image', model),
            lambda: diffusers.AutoPipelineForText2Image.from_pretrained(model).to(self.device or 'cpu'),
        )
        generator = None if request.seed is None else torch.Generator().manual_seed(request.seed)
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
        output = self._pipeline('automatic-speech-recognition', self.resolve_model('transcribe'))(
            inputs,
            return_timestamps=RETURN_TIMESTAMPS[request.timestamps],
            generate_kwargs={'language': request.language} if request.language else {},
            **self.call_options,
        )
        segments = _to_segments(output.get('chunks', []), audio.duration) if request.timestamps else ()
        return Transcript(output['text'].strip(), segments)
