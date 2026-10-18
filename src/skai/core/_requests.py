"""Define the requests that components send to backends and the transcripts backends return."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from PIL import Image

from ._media import Audio

Timestamps: TypeAlias = Literal['segment', 'word'] | None


@dataclass(frozen=True, eq=False)
class VideoFrames:
    """Hold frames sampled from a video, their times and the audio track.

    Args:
        frames:
            The sampled frames as PIL images.

        timestamps:
            The time of every frame in seconds.

        duration:
            The length of the video in seconds.

        audio:
            The mono audio track of the video, or `None` when it has none.
    """

    frames: tuple[Image.Image, ...]
    timestamps: tuple[float, ...]
    duration: float
    audio: Audio | None = None


Sample: TypeAlias = str | Image.Image | Audio | VideoFrames


@dataclass(frozen=True)
class ClassificationRequest:
    """Ask a backend to score a text or an image against a set of labels.

    Args:
        sample:
            The text, image, audio or sampled video frames to classify.

        labels:
            The candidate labels.

        descriptions:
            Descriptions of the labels, keyed by label.

        instructions:
            Extra guidance on how to assign the labels.

        examples:
            Labeled examples, as pairs of a sample and its labels.

        multi_label:
            Whether a sample can have several labels.
    """

    sample: Sample
    labels: tuple[str, ...]
    descriptions: dict[str, str] = field(default_factory=dict)
    instructions: str | None = None
    examples: tuple[tuple[Any, tuple[str, ...]], ...] = ()
    multi_label: bool = False


@dataclass(frozen=True)
class SummaryRequest:
    """Ask a backend to summarize a text.

    Args:
        text:
            The text to summarize.

        max_length:
            The approximate maximum number of words of the summary.

        style:
            A description of the style of the summary.

        examples:
            Example pairs of a text and its reference summary.
    """

    text: str
    max_length: int
    style: str | None = None
    examples: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TranslationRequest:
    """Ask a backend to translate a text.

    Args:
        text:
            The text to translate.

        target_language:
            The language to translate into.

        source_language:
            The language of the text. It is detected when it is `None`.

        examples:
            Example pairs of a text and its reference translation.

        glossary:
            Pairs of a term and its required translation.
    """

    text: str
    target_language: str
    source_language: str | None = None
    examples: tuple[tuple[str, str], ...] = ()
    glossary: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EmbeddingRequest:
    """Ask a backend to embed a text, an image or audio.

    Args:
        sample:
            The text, image or audio to embed.

        dimensions:
            The number of dimensions of the embedding, for backends that can shorten embeddings.
    """

    sample: str | Image.Image | Audio
    dimensions: int | None = None


@dataclass(frozen=True)
class SpeechRequest:
    """Ask a backend to speak a text.

    Args:
        text:
            The text to speak.

        voice:
            A voice name of the backend, or reference audio of the voice to clone.

        language:
            The language of the text.

        seed:
            The seed for backends that support seeding.
    """

    text: str
    voice: str | Audio | None = None
    language: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Ask a backend to generate an image from a prompt.

    Args:
        prompt:
            The description of the image.

        size:
            The width and height of the image in pixels.

        style:
            A description of the visual style.

        negative_prompt:
            A description of what the image must not show.

        seed:
            The seed for backends that support seeding.
    """

    prompt: str
    size: tuple[int, int]
    style: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DescriptionRequest:
    """Ask a backend to describe an image, audio or sampled video frames.

    Args:
        sample:
            The image, audio or sampled video frames to describe.

        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe.

        examples:
            Example pairs of a sample and its reference description.
    """

    sample: Image.Image | Audio | VideoFrames
    detail: Literal['brief', 'detailed'] = 'brief'
    instructions: str | None = None
    examples: tuple[tuple[Any, str], ...] = ()


@dataclass(frozen=True)
class TranscriptionRequest:
    """Ask a backend to transcribe speech.

    Args:
        audio:
            The speech to transcribe.

        language:
            The language of the speech. It is detected when it is `None`.

        timestamps:
            The granularity of the timestamps, or `None` for plain text.

        vocabulary:
            Domain terms that the speech may contain, given to the model as context for spelling.
    """

    audio: Audio
    language: str | None = None
    timestamps: Timestamps = None
    vocabulary: tuple[str, ...] = ()


@dataclass(frozen=True)
class AvatarRequest:
    """Ask a backend to animate a portrait so that it speaks an audio track.

    Args:
        avatar:
            The portrait to animate.

        audio:
            The speech the portrait says.

        seed:
            The seed for backends that support seeding.
    """

    avatar: Image.Image
    audio: Audio
    seed: int | None = None


@dataclass(frozen=True)
class ExtractionRequest:
    """Ask a backend to extract named fields from a text.

    Args:
        text:
            The text to extract from.

        fields:
            The field names and their descriptions, in output order.

        examples:
            Example pairs of a text and its reference record.
    """

    text: str
    fields: tuple[tuple[str, str], ...]
    examples: tuple[tuple[str, dict[str, Any]], ...] = ()


@dataclass(frozen=True)
class VideoGenerationRequest:
    """Ask a backend to generate a video from a prompt.

    Args:
        prompt:
            The description of the video.

        duration:
            The length of the video in seconds.

        size:
            The width and height of the video in pixels.

        fps:
            The number of frames per second, or `None` for the model's default.

        style:
            A description of the visual style.

        negative_prompt:
            A description of what the video must not show.

        seed:
            The seed for backends that support seeding.
    """

    prompt: str
    duration: float
    size: tuple[int, int]
    fps: int | None = None
    style: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class ImageEditRequest:
    """Ask a backend to edit an image with an instruction.

    Args:
        image:
            The image to edit.

        instruction:
            The edit to apply.

        mask:
            The region to edit, as a grayscale image where white marks the region, or `None` for the whole image.

        seed:
            The seed for backends that support seeding.
    """

    image: Image.Image
    instruction: str
    mask: Image.Image | None = None
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class AnimationRequest:
    """Ask a backend to turn an image into a short video that starts from it.

    Args:
        image:
            The first frame of the video.

        motion_prompt:
            A description of the motion.

        duration:
            The length of the video in seconds.

        fps:
            The number of frames per second, or `None` for the model's default.

        seed:
            The seed for backends that support seeding.
    """

    image: Image.Image
    motion_prompt: str | None = None
    duration: float = 2.0
    fps: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Segment:
    """Hold one timed piece of a transcript.

    Args:
        start:
            The start time in seconds.

        end:
            The end time in seconds.

        text:
            The transcribed text.
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """Hold a transcript and its timed segments.

    Args:
        text:
            The full transcribed text.

        segments:
            The timed segments or words.
    """

    text: str
    segments: tuple[Segment, ...] = ()
