"""Return deterministic placeholder outputs for every task."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import zlib
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from skai.core import (
    AnimationRequest,
    Audio,
    AudioGenerationRequest,
    AudioOperationRequest,
    AvatarRequest,
    Backend,
    ClassificationRequest,
    DiarizationRequest,
    EmbeddingRequest,
    ExtractionRequest,
    FineTuneRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOperationRequest,
    ImageTranscriptionRequest,
    LipSyncRequest,
    Moment,
    MomentRequest,
    Region,
    RegionRequest,
    Segment,
    SpeechRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    Transcript,
    TranscriptionRequest,
    Video,
    VideoFrames,
    VideoGenerationRequest,
)

SAMPLE_RATE = 16000
FRAMES_PER_SECOND = 25
EMBEDDING_DIMENSIONS = 8
AVATAR_SIZE = 64
VIDEO_FPS = 8
EDIT_STRENGTH = 0.5
SPEECH_SECONDS_PER_WORD = 0.1
SPEECH_PADDING_SECONDS = 0.2
LIST_ITEMS = 2
MOMENT_SCORE = 0.9
STEM_GAIN = 0.5


def _items_of(part: dict | list | tuple | VideoFrames) -> list[Any]:
    """Return the items of a container part of a request, with the keys of a dict."""
    if isinstance(part, VideoFrames):
        return list(part.frames)
    if isinstance(part, dict):
        return [item for key, value in part.items() for item in (key, value)]
    return list(part)


def _to_bytes(part: Any) -> bytes:
    """Return a stable byte representation of a part of a request."""
    if isinstance(part, bytes):
        return part
    if isinstance(part, Image.Image | np.ndarray):
        return part.tobytes()
    if isinstance(part, Audio | Video):
        return (part.samples if isinstance(part, Audio) else part.frames).tobytes()
    if isinstance(part, dict | list | tuple | VideoFrames):
        return b''.join(_to_bytes(item) for item in _items_of(part))
    return repr(part).encode()


def _seed_of(*parts: Any) -> int:
    """Derive a seed from the parts of a request."""
    seed = 0
    for part in parts:
        seed = zlib.crc32(_to_bytes(part), seed)
    return seed


def _color_of(*parts: Any) -> tuple[int, int, int]:
    """Derive an RGB color from the parts of a request."""
    red, green, blue = (int(value) for value in np.random.default_rng(_seed_of(*parts)).integers(0, 256, size=3))
    return red, green, blue


def _describe_media(sample: Any) -> str:
    """Return a short sentence about an image, audio, a video or its sampled frames."""
    if isinstance(sample, Image.Image):
        return f'A {sample.width}x{sample.height} image.'
    if isinstance(sample, Audio):
        return f'{sample.duration:.2f} seconds of audio.'
    if isinstance(sample, VideoFrames):
        return f'{len(sample.frames)} frames of a {sample.duration:.2f} second video.'
    if isinstance(sample, Video):
        return f'A {sample.duration:.2f} second video.'
    return str(sample)


def _describe(sample: Any) -> str:
    """Return a short sentence about a sample of any input kind."""
    if isinstance(sample, dict):
        return ' '.join(f'{key}: {_describe(value)}' for key, value in sample.items())
    return sample if isinstance(sample, str) else _describe_media(sample)


def _fill(declaration: Any, words: list[str], seed: int) -> Any:
    """Fill a field declaration with words of the sample, following its nesting."""
    if isinstance(declaration, dict):
        return {name: _fill(value, words, _seed_of(seed, name)) for name, value in declaration.items()}
    if isinstance(declaration, list):
        return [_fill(declaration[0] if declaration else '', words, seed + item) for item in range(LIST_ITEMS)]
    return words[seed % len(words)] if words else None


def _tone(duration: float, *parts: Any) -> Audio:
    """Return a quiet sine tone of a duration, whose pitch depends on the parts."""
    times = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    frequency = 200 + _seed_of(*parts) % 400
    return Audio((0.1 * np.sin(2 * np.pi * frequency * times)).astype(np.float32), SAMPLE_RATE)


def _box(size: tuple[int, int], *parts: Any) -> tuple[int, int, int, int]:
    """Return a deterministic box that covers a quarter of an image."""
    width, height = size
    shift = _seed_of(*parts) % max(1, width // 2)
    return (shift, height // 4, min(width, shift + width // 2), min(height, height // 4 + height // 2))


def _still_video(image: Image.Image, duration: float, fps: int) -> Video:
    """Repeat an image for the length of a video."""
    frames = np.repeat(np.asarray(image.convert('RGB'))[np.newaxis], max(1, round(duration * fps)), axis=0)
    return Video(frames, fps)


def _score_labels(request: ClassificationRequest) -> dict[str, float]:
    """Return random but deterministic scores for the labels of a request."""
    rng = np.random.default_rng(_seed_of(request.sample))
    size = len(request.labels)
    scores = rng.random(size) if request.multi_label else rng.dirichlet(np.ones(size))
    return dict(zip(request.labels, scores.tolist(), strict=True))


class DummyBackend(Backend):
    """Return deterministic placeholder outputs for every task, without network access or model weights.

    Outputs depend only on the request, so repeated calls return the same results. The backend is meant for tests and
    documentation.

    Args:
        model:
            Ignored.

        options:
            Ignored.

        max_concurrency:
            The maximum number of requests that run at the same time.

    Examples:
        >>> from skai.generation import TextToTextGenerator
        >>> generator = TextToTextGenerator(instructions='Summarize.', max_length=3, backend='dummy').fit()
        >>> generator.predict(['One two three four five.']).tolist()
        ['One two three']
    """

    supports_voice_cloning: ClassVar[bool] = True
    supports_masks: ClassVar[bool] = True
    supports_reference_images: ClassVar[bool] = True
    accepts_audio: ClassVar[bool] = True
    sends_video_audio: ClassVar[bool] = True
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})
    accepts_context: ClassVar[bool] = True
    shortens_embeddings: ClassVar[bool] = True
    avatar_size: ClassVar[int | None] = AVATAR_SIZE

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the text.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the image.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def classify_audio(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the audio.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def classify_video(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the sampled frames.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def classify_multimodal(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from every part of the sample.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def generate_text(self, request: TextGenerationRequest) -> str:
        """Return a sentence about the sample, with the glossary applied and cut to the maximum length.

        Args:
            request:
                The text generation request.

        Returns:
            The text itself for a text sample, or a sentence that states the size of other samples.
        """
        text = _describe(request.sample)
        for term, rendering in request.glossary:
            text = text.replace(term, rendering)
        return ' '.join(text.split()[: request.max_length])

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Fill every field with words of the sample, following the nesting of the declarations.

        Args:
            request:
                The extraction request.

        Returns:
            The value of every field. A list field holds two items.
        """
        words = _describe(request.sample).split()
        return {name: _fill(declaration, words, _seed_of(name, request.sample)) for name, declaration in request.fields}

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Return a sentence that states the length of the speech.

        Args:
            request:
                The transcription request.

        Returns:
            The transcript, with one segment or with evenly spaced words when timestamps are requested.
        """
        duration = request.audio.duration
        text = f'Speech of {duration:.2f} seconds.'
        text = f'{text} Terms: {", ".join(request.vocabulary)}.' if request.vocabulary else text
        if request.timestamps == 'segment':
            return Transcript(text, (Segment(0.0, duration, text),))
        if request.timestamps == 'word':
            words = text.split()
            step = duration / len(words)
            return Transcript(text, tuple(Segment(i * step, (i + 1) * step, word) for i, word in enumerate(words)))
        return Transcript(text)

    def transcribe_image(self, request: ImageTranscriptionRequest) -> str:
        """Return a sentence that states the size of the image.

        Args:
            request:
                The image transcription request.

        Returns:
            The text, as a Markdown heading when the request asks for Markdown.
        """
        text = f'Text of a {request.image.width}x{request.image.height} image.'
        return f'# {text}' if request.markdown else text

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Return a deterministic random vector derived from the text.

        Args:
            request:
                The embedding request.

        Returns:
            A vector with `dimensions` entries, or 8 when `dimensions` is `None`.
        """
        rng = np.random.default_rng(_seed_of(request.sample))
        return rng.standard_normal(request.dimensions or EMBEDDING_DIMENSIONS)

    def embed_image(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Return a deterministic random vector derived from the image.

        Args:
            request:
                The embedding request.

        Returns:
            A vector with `dimensions` entries, or 8 when `dimensions` is `None`.
        """
        return self.embed_text(request)

    def embed_audio(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Return a deterministic random vector derived from the audio.

        Args:
            request:
                The embedding request.

        Returns:
            A vector with `dimensions` entries, or 8 when `dimensions` is `None`.
        """
        return self.embed_text(request)

    def detect_objects(self, request: RegionRequest) -> list[Region]:
        """Return one box per label, or one box for the prompt.

        Args:
            request:
                The region request.

        Returns:
            The regions, each covering a quarter of the image.
        """
        labels = request.labels or (request.prompt or 'object',)
        return [Region(label, MOMENT_SCORE, _box(request.image.size, label)) for label in labels]

    def segment_image(self, request: RegionRequest) -> list[Region]:
        """Return one box and the mask of the box per label, or for the prompt.

        Args:
            request:
                The region request.

        Returns:
            The regions, with masks of the image size.
        """
        regions = []
        for region in self.detect_objects(request):
            mask = Image.new('L', request.image.size, 0)
            mask.paste(255, region.box)
            regions.append(Region(region.label, region.score, region.box, mask))
        return regions

    def diarize(self, request: DiarizationRequest) -> Transcript:
        """Split the audio into two halves spoken by two speakers.

        Args:
            request:
                The diarization request.

        Returns:
            The transcript, with two segments.
        """
        half = request.audio.duration / 2
        segments = (
            Segment(0.0, half, f'First half of {half:.2f} seconds.', 'SPEAKER_0'),
            Segment(half, 2 * half, f'Second half of {half:.2f} seconds.', 'SPEAKER_1'),
        )
        return Transcript(' '.join(segment.text for segment in segments), segments)

    def find_moments(self, request: MomentRequest) -> list[Moment]:
        """Return the middle third of the audio or video as the one matching moment.

        Args:
            request:
                The moment request.

        Returns:
            One moment.
        """
        third = request.sample.duration / 3
        return [Moment(third, 2 * third, MOMENT_SCORE)]

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Return a sine tone whose length grows with the number of words.

        Args:
            request:
                The speech request.

        Returns:
            A mono tone at 16 kHz.
        """
        duration = SPEECH_PADDING_SECONDS + SPEECH_SECONDS_PER_WORD * len(request.text.split())
        return _tone(duration, request.text, request.voice, request.seed)

    def generate_music(self, request: AudioGenerationRequest) -> Audio:
        """Return a sine tone of the requested length.

        Args:
            request:
                The audio generation request.

        Returns:
            A mono tone at 16 kHz.
        """
        return _tone(request.duration, 'music', request.prompt, request.seed)

    def generate_sound(self, request: AudioGenerationRequest) -> Audio:
        """Return a sine tone of the requested length.

        Args:
            request:
                The audio generation request.

        Returns:
            A mono tone at 16 kHz.
        """
        return _tone(request.duration, 'sound', request.prompt, request.seed)

    def separate_audio(self, request: AudioOperationRequest) -> Audio:
        """Return the audio at half its volume as the stem.

        Args:
            request:
                The audio operation request.

        Returns:
            Audio as long as the input.
        """
        mono = request.audio.to_mono()
        return Audio((mono.samples * STEM_GAIN).astype(np.float32), mono.sample_rate)

    def enhance_audio(self, request: AudioOperationRequest) -> Audio:
        """Return the audio clipped to `[-1, 1]` as the denoised copy.

        Args:
            request:
                The audio operation request.

        Returns:
            Audio as long as the input.
        """
        mono = request.audio.to_mono()
        return Audio(np.clip(mono.samples, -1.0, 1.0).astype(np.float32), mono.sample_rate)

    def convert_voice(self, request: AudioOperationRequest) -> Audio:
        """Return a tone as long as the input whose pitch depends on the target voice.

        Args:
            request:
                The audio operation request.

        Returns:
            A mono tone at 16 kHz.
        """
        return _tone(request.audio.duration, request.voice, request.seed)

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Return an image of one color derived from the request.

        Args:
            request:
                The image generation request.

        Returns:
            An RGB image of the requested size.
        """
        color = _color_of(
            request.prompt,
            request.style,
            request.negative_prompt,
            request.seed,
            request.reference_images,
        )
        return Image.new('RGB', request.size, color)

    def edit_image(self, request: ImageEditRequest) -> Image.Image:
        """Blend the image with a color derived from the instruction, inside the mask when one is given.

        Args:
            request:
                The image edit request.

        Returns:
            The edited image, of the size of the input.
        """
        original = request.image.convert('RGB')
        tint = Image.new('RGB', original.size, _color_of(request.instruction, request.seed))
        edited = Image.blend(original, tint, EDIT_STRENGTH)
        if request.mask is None:
            return edited
        return Image.composite(edited, original, request.mask.convert('L').resize(original.size))

    def upscale_image(self, request: ImageOperationRequest) -> Image.Image:
        """Resize the image by the scale.

        Args:
            request:
                The image operation request.

        Returns:
            The upscaled image.
        """
        width, height = request.image.size
        return request.image.convert('RGB').resize((width * request.scale, height * request.scale))

    def remove_background(self, request: ImageOperationRequest) -> Image.Image:
        """Keep a centered ellipse of the image and make the rest transparent.

        Args:
            request:
                The image operation request.

        Returns:
            An RGBA image of the input size.
        """
        image = request.image.convert('RGBA')
        alpha = Image.new('L', image.size, 0)
        ImageDraw.Draw(alpha).ellipse((0, 0, image.width - 1, image.height - 1), fill=255)
        image.putalpha(alpha)
        return image

    def pick_thumbnail(self, request: ThumbnailRequest) -> int:
        """Pick a frame derived from the frames.

        Args:
            request:
                The thumbnail request.

        Returns:
            The index of the frame.
        """
        return _seed_of(request.frames) % len(request.frames.frames)

    def generate_video(self, request: VideoGenerationRequest) -> Video:
        """Return a video of one color derived from the request.

        Args:
            request:
                The video generation request.

        Returns:
            A video of the requested duration and size, at `fps` or 8 frames per second.
        """
        color = _color_of(request.prompt, request.style, request.negative_prompt, request.seed)
        return _still_video(Image.new('RGB', request.size, color), request.duration, request.fps or VIDEO_FPS)

    def animate_image(self, request: AnimationRequest) -> Video:
        """Return the still image repeated for the length of the video.

        Args:
            request:
                The animation request.

        Returns:
            A video of the requested duration, at `fps` or 8 frames per second.
        """
        return _still_video(request.image, request.duration, request.fps or VIDEO_FPS)

    def animate_avatar(self, request: AvatarRequest) -> Video:
        """Return the still avatar repeated for the length of the audio.

        Args:
            request:
                The avatar request.

        Returns:
            A video at 25 frames per second with the audio as its track.
        """
        avatar = request.avatar.convert('RGB')
        avatar.thumbnail((AVATAR_SIZE, AVATAR_SIZE))
        n_frames = max(1, round(request.audio.duration * FRAMES_PER_SECOND))
        frames = np.repeat(np.asarray(avatar)[np.newaxis], n_frames, axis=0)
        return Video(frames, FRAMES_PER_SECOND, request.audio)

    def lip_sync(self, request: LipSyncRequest) -> Video:
        """Loop or cut the frames of the video to the length of the audio, with the audio as its track.

        Args:
            request:
                The lip sync request.

        Returns:
            A video as long as the audio.
        """
        frames = request.video.frames
        count = max(1, round(request.audio.duration * request.video.fps))
        return Video(frames[np.arange(count) % frames.shape[0]], request.video.fps, request.audio)

    def fine_tune(self, request: FineTuneRequest) -> str:
        """Return a model id derived from the training prompts and answers, without training anything.

        Args:
            request:
                The fine-tuning request.

        Returns:
            A deterministic model id.
        """
        parts = [part for prompt, answer in request.examples for part in (prompt.user, answer)]
        return f'dummy-finetuned-{_seed_of(request.task, *parts)}'
