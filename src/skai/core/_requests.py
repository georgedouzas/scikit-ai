"""Define the requests that components send to backends and the transcripts backends return."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from PIL import Image

from ._media import Audio

Timestamps: TypeAlias = Literal['segment', 'word'] | None


@dataclass(frozen=True)
class ClassificationRequest:
    """Ask a backend to score a text or an image against a set of labels.

    Args:
        sample:
            The text or the image to classify.

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

    sample: str | Image.Image
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
    """

    text: str
    max_length: int
    style: str | None = None


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
    """

    text: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    """Ask a backend to embed a text.

    Args:
        text:
            The text to embed.

        dimensions:
            The number of dimensions of the embedding, for backends that can shorten embeddings.
    """

    text: str
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
    """Ask a backend to describe an image.

    Args:
        image:
            The image to describe.

        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe.
    """

    image: Image.Image
    detail: Literal['brief', 'detailed'] = 'brief'
    instructions: str | None = None


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
    """

    audio: Audio
    language: str | None = None
    timestamps: Timestamps = None


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
