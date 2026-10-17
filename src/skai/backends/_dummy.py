"""Return deterministic placeholder outputs for every task."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import zlib
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    AnimationRequest,
    Audio,
    AvatarRequest,
    Backend,
    ClassificationRequest,
    DescriptionRequest,
    EmbeddingRequest,
    ExtractionRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    Segment,
    SpeechRequest,
    SummaryRequest,
    Transcript,
    TranscriptionRequest,
    TranslationRequest,
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


def _to_bytes(part: Any) -> bytes:
    """Return a stable byte representation of a part of a request."""
    if isinstance(part, bytes):
        return part
    if isinstance(part, Image.Image):
        return part.tobytes()
    if isinstance(part, np.ndarray):
        return part.tobytes()
    if isinstance(part, Audio):
        return part.samples.tobytes()
    if isinstance(part, VideoFrames):
        return b''.join(frame.tobytes() for frame in part.frames)
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
        >>> from skai.text import TextSummarizer
        >>> summarizer = TextSummarizer(max_length=3, backend='dummy').fit()
        >>> summarizer.predict(['One two three four five.']).tolist()
        ['One two three']
    """

    supports_voice_cloning: ClassVar[bool] = True
    supports_masks: ClassVar[bool] = True
    accepts_audio: ClassVar[bool] = True
    sends_video_audio: ClassVar[bool] = True
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the text.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

    def summarize(self, request: SummaryRequest) -> str:
        """Return the first words of the text.

        Args:
            request:
                The summary request.

        Returns:
            The first `max_length` words.
        """
        return ' '.join(request.text.split()[: request.max_length])

    def translate(self, request: TranslationRequest) -> str:
        """Return the text prefixed with the target language.

        Args:
            request:
                The translation request.

        Returns:
            The text in the form `[<target language>] <text>`.
        """
        return f'[{request.target_language}] {request.text}'

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

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Return a sine tone whose length grows with the number of words.

        Args:
            request:
                The speech request.

        Returns:
            A mono tone at 16 kHz.
        """
        duration = SPEECH_PADDING_SECONDS + SPEECH_SECONDS_PER_WORD * len(request.text.split())
        times = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
        frequency = 200 + _seed_of(request.text, request.voice, request.seed) % 400
        return Audio((0.1 * np.sin(2 * np.pi * frequency * times)).astype(np.float32), SAMPLE_RATE)

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Return an image of one color derived from the request.

        Args:
            request:
                The image generation request.

        Returns:
            An RGB image of the requested size.
        """
        color = _color_of(request.prompt, request.style, request.negative_prompt, request.seed)
        return Image.new('RGB', request.size, color)

    def describe_image(self, request: DescriptionRequest) -> str:
        """Return a sentence that states the size of the image.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        assert isinstance(request.sample, Image.Image)
        width, height = request.sample.size
        description = f'A {request.detail} description of a {width}x{height} image.'
        return f'{description} {request.instructions}' if request.instructions else description

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Return deterministic scores derived from the image.

        Args:
            request:
                The classification request.

        Returns:
            A score per label. The scores sum to one unless the request is multilabel.
        """
        return _score_labels(request)

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
        if request.timestamps == 'segment':
            return Transcript(text, (Segment(0.0, duration, text),))
        if request.timestamps == 'word':
            words = text.split()
            step = duration / len(words)
            return Transcript(text, tuple(Segment(i * step, (i + 1) * step, word) for i, word in enumerate(words)))
        return Transcript(text)

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

    def describe_audio(self, request: DescriptionRequest) -> str:
        """Return a sentence that states the length of the audio.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        assert isinstance(request.sample, Audio)
        return f'A {request.detail} description of {request.sample.duration:.2f} seconds of audio.'

    def describe_video(self, request: DescriptionRequest) -> str:
        """Return a sentence that states the number of sampled frames and the length of the video.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        assert isinstance(request.sample, VideoFrames)
        frames, seconds = len(request.sample.frames), request.sample.audio.duration
        return f'A {request.detail} description of {frames} frames of a {seconds:.2f} second video.'

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Return a word of the text for every field whose name appears in it, and `None` otherwise.

        Args:
            request:
                The extraction request.

        Returns:
            The value of every field.
        """
        words = request.text.split()
        return {
            name: words[_seed_of(name, request.text) % len(words)] if name.lower() in request.text.lower() else None
            for name, _ in request.fields
        }

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

    def animate_image(self, request: AnimationRequest) -> Video:
        """Return the still image repeated for the length of the video.

        Args:
            request:
                The animation request.

        Returns:
            A video of the requested duration, at `fps` or 8 frames per second.
        """
        return _still_video(request.image, request.duration, request.fps or VIDEO_FPS)
