"""Define the requests that components send to backends and the transcripts backends return."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from PIL import Image

from ._media import Audio, Video

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


Sample: TypeAlias = str | Image.Image | Audio | VideoFrames | dict[str, Any]


@dataclass(frozen=True)
class ClassificationRequest:
    """Ask a backend to score a sample against a set of labels.

    Args:
        sample:
            The text, image, audio, sampled video frames or multimodal sample to classify.

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

        reference_images:
            Images of a subject that the image must show consistently.
    """

    prompt: str
    size: tuple[int, int]
    style: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    reference_images: tuple[Image.Image, ...] = ()


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
    """Ask a backend to extract named fields from a sample.

    Args:
        sample:
            The text, image, audio, sampled video frames or multimodal sample to extract from.

        fields:
            The field names and their declarations, in output order. A declaration is a description, a dict of
            subfields, or a one-element list of either for a list of values.

        instructions:
            Extra guidance on how to extract the fields.

        examples:
            Example pairs of a sample and its reference record.
    """

    sample: Sample
    fields: tuple[tuple[str, Any], ...]
    instructions: str | None = None
    examples: tuple[tuple[Any, dict[str, Any]], ...] = ()


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

        speaker:
            The label of the speaker, or `None` when speakers are not labeled.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


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


@dataclass(frozen=True, eq=False)
class TextGenerationRequest:
    """Ask a backend to produce text from a sample by following instructions.

    Args:
        sample:
            The text, image, audio, sampled video frames or multimodal sample.

        instructions:
            What to produce.

        examples:
            Example pairs of a sample and its reference output.

        max_length:
            The maximum number of words of the output, or `None` for no limit.

        glossary:
            Pairs of a term and its required rendering.
    """

    sample: Sample
    instructions: str
    examples: tuple[tuple[Any, str], ...] = ()
    max_length: int | None = None
    glossary: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, eq=False)
class ImageTranscriptionRequest:
    """Ask a backend to transcribe the text written in an image.

    Args:
        image:
            The image to read.

        markdown:
            Whether to keep the page structure as Markdown.
    """

    image: Image.Image
    markdown: bool = False


@dataclass(frozen=True, eq=False)
class RegionRequest:
    """Ask a backend to find regions of an image, as boxes or masks.

    Args:
        image:
            The image to search.

        labels:
            The kinds of regions to find.

        prompt:
            A description of the regions to find, used when `labels` is empty.
    """

    image: Image.Image
    labels: tuple[str, ...] = ()
    prompt: str | None = None


@dataclass(frozen=True)
class Region:
    """Hold one region that a backend found in an image.

    Args:
        label:
            The kind of region.

        score:
            The confidence of the backend.

        box:
            The `(left, top, right, bottom)` pixels of the region.

        mask:
            A grayscale image of the input size whose white pixels belong to the region, or `None` for boxes.
    """

    label: str
    score: float
    box: tuple[int, int, int, int]
    mask: Image.Image | None = None


@dataclass(frozen=True, eq=False)
class MomentRequest:
    """Ask a backend to find the parts of audio or a video that match a description.

    Args:
        sample:
            The audio or the sampled video frames.

        prompt:
            A description of the moments to find.
    """

    sample: Audio | VideoFrames
    prompt: str


@dataclass(frozen=True)
class Moment:
    """Hold one part of audio or a video that matches a description.

    Args:
        start:
            The start time in seconds.

        end:
            The end time in seconds.

        score:
            The confidence of the backend.
    """

    start: float
    end: float
    score: float


@dataclass(frozen=True, eq=False)
class DiarizationRequest:
    """Ask a backend to find who speaks when, with what they say.

    Args:
        audio:
            The speech to diarize.

        language:
            The language of the speech. It is detected when it is `None`.

        vocabulary:
            Domain terms that the speech may contain.
    """

    audio: Audio
    language: str | None = None
    vocabulary: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class ImageOperationRequest:
    """Ask a backend to upscale an image or remove its background.

    Args:
        image:
            The image.

        scale:
            The factor of upscaling.

        seed:
            The seed for backends that support seeding.
    """

    image: Image.Image
    scale: int = 2
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class ThumbnailRequest:
    """Ask a backend to pick the frame that best represents a video.

    Args:
        frames:
            The sampled frames of the video.
    """

    frames: VideoFrames


@dataclass(frozen=True, eq=False)
class AudioGenerationRequest:
    """Ask a backend to generate music or a sound from a description.

    Args:
        prompt:
            The description of the audio.

        duration:
            The length of the audio in seconds.

        seed:
            The seed for backends that support seeding.
    """

    prompt: str
    duration: float
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class AudioOperationRequest:
    """Ask a backend to separate a stem of audio or to convert its voice.

    Args:
        audio:
            The audio.

        stem:
            The stem to keep when separating, `'vocals'` or `'accompaniment'`.

        voice:
            Reference audio of the target voice when converting.

        seed:
            The seed for backends that support seeding.
    """

    audio: Audio
    stem: str = 'vocals'
    voice: Audio | None = None
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class LipSyncRequest:
    """Ask a backend to make a video speak new audio.

    Args:
        video:
            The video of a speaking person.

        audio:
            The new speech.

        seed:
            The seed for backends that support seeding.
    """

    video: Video
    audio: Audio
    seed: int | None = None
