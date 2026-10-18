"""Coerce samples into media objects and write generated media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import base64
import io
import os
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeAlias

import httpx
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from ._errors import InvalidSampleError, MediaReadError
from ._media import Audio, Video
from ._requests import VideoFrames

URL_PREFIXES = ('http://', 'https://')
URL_TIMEOUT = 30.0
FILE_EXTENSIONS = {'image': 'png', 'audio': 'wav', 'video': 'mp4'}
TABLE_NDIM = 2
AUDIO_TUPLE_LENGTH = 2

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tif', '.tiff'})
AUDIO_EXTENSIONS = frozenset({'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'})
VIDEO_EXTENSIONS = frozenset({'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v'})

MediaObject: TypeAlias = Image.Image | Audio | Video
MediaKind: TypeAlias = Literal['image', 'audio', 'video']
InputKind: TypeAlias = Literal['text', 'image', 'audio', 'video', 'multimodal']
MEDIA_KINDS: tuple[MediaKind, ...] = ('image', 'audio', 'video')
OutputFormat: TypeAlias = Literal['object', 'bytes', 'path']


def _fetch_url(url: str, index: int) -> bytes:
    """Download the content of a URL."""
    try:
        response = httpx.get(url, timeout=URL_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        msg = f'Sample {index}: cannot download `{url}` ({error}).'
        raise MediaReadError(msg) from error
    return response.content


def _read_file(path: str | os.PathLike, index: int) -> bytes:
    """Read the content of a local file."""
    try:
        return Path(path).read_bytes()
    except OSError as error:
        msg = f'Sample {index}: cannot read file `{path}` ({error.strerror}).'
        raise MediaReadError(msg) from error


def _read_source(sample: str | os.PathLike | bytes, index: int) -> bytes:
    """Read the bytes behind a path, a URL or raw bytes."""
    if isinstance(sample, bytes):
        return sample
    if isinstance(sample, str) and sample.startswith(URL_PREFIXES):
        return _fetch_url(sample, index)
    return _read_file(sample, index)


def _write_bytes(item: MediaObject, kind: MediaKind) -> bytes:
    """Encode one generated media object as bytes."""
    if isinstance(item, Image.Image):
        return encode_png(item)
    return item.to_bytes()


def _write_file(item: MediaObject, kind: MediaKind, directory: Path, prefix: str) -> str:
    """Write one generated media object to a new file and return its path."""
    path = directory / f'{prefix}-{uuid.uuid4().hex}.{FILE_EXTENSIONS[kind]}'
    with path.open('xb') as file:
        file.write(_write_bytes(item, kind))
    if isinstance(item, Video) and item.subtitles is not None:
        path.with_suffix('.srt').write_text(item.subtitles, encoding='utf-8')
    return str(path)


def encode_png(image: Image.Image) -> bytes:
    """Encode an image as PNG bytes.

    Args:
        image:
            The image to encode.

    Returns:
        The PNG-encoded image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def to_data_url(image: Image.Image) -> str:
    """Encode an image as a PNG data URL.

    Args:
        image:
            The image to encode.

    Returns:
        The data URL.
    """
    return f'data:image/png;base64,{base64.b64encode(encode_png(image.convert("RGB"))).decode()}'


def to_object_array(items: Iterable[Any]) -> NDArray[np.object_]:
    """Pack items into a one-dimensional object array without broadcasting them.

    Args:
        items:
            The items to pack.

    Returns:
        The object array that holds the items.
    """
    items = list(items)
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def to_samples(X: Any, rows: bool = False) -> list[Any]:
    """Convert a container of samples into a list of samples.

    Args:
        X:
            A list, tuple, NumPy array, pandas Series or single-column DataFrame of samples, or `None`.

        rows:
            Whether every row of a DataFrame is one multimodal sample, a dict keyed by column.

    Returns:
        The samples.

    Raises:
        ValueError:
            If `X` is two-dimensional with more than one column and `rows` is `False`.
    """
    if X is None:
        return []
    if rows and hasattr(X, 'to_dict') and hasattr(X, 'columns'):
        return [{str(key): value for key, value in row.items()} for row in X.to_dict('records')]
    if hasattr(X, 'to_numpy') and not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(X, np.ndarray):
        return list(X)
    if X.ndim == TABLE_NDIM and X.dtype.kind in 'USO':
        if X.shape[1] != 1:
            msg = f'Expected one column of samples, but got an array with {X.shape[1]} columns.'
            raise ValueError(msg)
        return list(X[:, 0])
    return list(X)


def check_text(sample: object, index: int) -> str:
    """Check that a sample is a string.

    Args:
        sample:
            The sample to check.

        index:
            The position of the sample in the input.

    Returns:
        The sample.

    Raises:
        InvalidSampleError:
            If the sample is not a string.
    """
    if not isinstance(sample, str):
        msg = f'Sample {index} must be a string, but got {type(sample).__name__}.'
        raise InvalidSampleError(msg)
    return sample


def read_image(sample: object, index: int = 0) -> Image.Image:
    """Read an image from a path, a URL, bytes, a PIL image or a NumPy array.

    Args:
        sample:
            The image to read.

        index:
            The position of the sample in the input.

    Returns:
        The image.

    Raises:
        MediaReadError:
            If the sample cannot be read as an image.
    """
    if isinstance(sample, Image.Image):
        return sample
    if isinstance(sample, np.ndarray):
        try:
            return Image.fromarray(sample)
        except (TypeError, ValueError) as error:
            msg = f'Sample {index}: cannot convert an array of shape {sample.shape} to an image ({error}).'
            raise MediaReadError(msg) from error
    if not isinstance(sample, str | bytes | os.PathLike):
        msg = f'Sample {index}: cannot read an image from {type(sample).__name__}.'
        raise MediaReadError(msg)
    try:
        image = Image.open(io.BytesIO(_read_source(sample, index)))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        msg = f'Sample {index}: cannot decode the image ({error}).'
        raise MediaReadError(msg) from error
    return image


def read_audio(sample: object, index: int = 0) -> Audio:
    """Read audio from a path, a URL, bytes, an `Audio` object or a `(samples, sample_rate)` tuple.

    Args:
        sample:
            The audio to read.

        index:
            The position of the sample in the input.

    Returns:
        The audio.

    Raises:
        MediaReadError:
            If the sample cannot be read as audio.
    """
    if isinstance(sample, Audio):
        return sample
    if isinstance(sample, tuple) and len(sample) == AUDIO_TUPLE_LENGTH:
        samples, sample_rate = sample
        return Audio(np.asarray(samples, dtype=np.float32), int(sample_rate))
    if not isinstance(sample, str | bytes | os.PathLike):
        msg = f'Sample {index}: cannot read audio from {type(sample).__name__}.'
        raise MediaReadError(msg)
    try:
        return Audio.from_bytes(_read_source(sample, index))
    except sf.LibsndfileError as error:
        msg = f'Sample {index}: cannot decode the audio ({error}).'
        raise MediaReadError(msg) from error


def read_video(sample: object, index: int = 0) -> Video:
    """Read a video from a path, a URL, bytes or a `Video` object.

    Args:
        sample:
            The video to read.

        index:
            The position of the sample in the input.

    Returns:
        The video.

    Raises:
        MediaReadError:
            If the sample cannot be read as a video.
    """
    if isinstance(sample, Video):
        return sample
    if not isinstance(sample, str | bytes | os.PathLike):
        msg = f'Sample {index}: cannot read a video from {type(sample).__name__}.'
        raise MediaReadError(msg)
    return Video(data=_read_source(sample, index))


def read_video_frames(sample: object, index: int, n_frames: int) -> VideoFrames:
    """Read a video and sample evenly spaced frames of it.

    Args:
        sample:
            The video to read, or frames that were already sampled.

        index:
            The position of the sample in the input.

        n_frames:
            The number of frames to sample.

    Returns:
        The sampled frames.

    Raises:
        MediaReadError:
            If the video cannot be decoded.
    """
    if isinstance(sample, VideoFrames):
        return sample
    video = read_video(sample, index)
    try:
        return sample_frames(video, n_frames)
    except (OSError, ValueError) as error:
        msg = f'Sample {index}: cannot decode the video ({error}).'
        raise MediaReadError(msg) from error


def _kind_of_path(path: str | os.PathLike) -> MediaKind | None:
    """Return the media kind of a path or URL by its file extension."""
    suffix = Path(str(path).split('?')[0]).suffix.lower()
    for kind, extensions in zip(MEDIA_KINDS, (IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS), strict=True):
        if suffix in extensions:
            return kind
    return None


def _kind_of_part(value: object) -> MediaKind | None:
    """Return the media kind of a path or URL part, or `None` for anything else."""
    if isinstance(value, os.PathLike):
        return _kind_of_path(value)
    if not isinstance(value, str):
        return None
    if value.startswith(URL_PREFIXES) or (Path(value).suffix and Path(value).is_file()):
        return _kind_of_path(value)
    return None


def _read_bytes(value: bytes, index: int, n_frames: int | None) -> object:
    """Read raw bytes as an image, then as audio, and otherwise as a video."""
    for kind in MEDIA_KINDS[:2]:
        try:
            return read_input(kind, value, index)
        except MediaReadError:
            continue
    return read_input('video', value, index, n_frames)


def _read_part(value: object, index: int, n_frames: int | None) -> object:
    """Read one named part of a multimodal sample by its own kind."""
    if isinstance(value, Image.Image | Audio | VideoFrames):
        return value
    if isinstance(value, Video):
        return value if n_frames is None else sample_frames(value, n_frames)
    if isinstance(value, np.ndarray):
        return read_image(value, index)
    if isinstance(value, tuple) and len(value) == AUDIO_TUPLE_LENGTH and isinstance(value[0], np.ndarray):
        return read_audio(value, index)
    if isinstance(value, bytes):
        return _read_bytes(value, index, n_frames)
    kind = _kind_of_part(value)
    return value if kind is None else read_input(kind, value, index, n_frames)


def read_multimodal(
    sample: object,
    index: int = 0,
    n_frames: int | None = None,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Read a multimodal sample, a dict of named texts, images, audio, videos or transcripts.

    Strings are read as texts, unless they are URLs or paths of existing files with a media extension.

    Args:
        sample:
            The dict, or a pandas row, to read.

        index:
            The position of the sample in the input.

        n_frames:
            The number of frames to sample from every video, or `None` to keep videos whole.

        required:
            The keys the sample must have.

    Returns:
        The sample, with every part read by its own kind.

    Raises:
        InvalidSampleError:
            If the sample is not a dict or misses a required key.
    """
    if hasattr(sample, 'to_dict') and not isinstance(sample, dict):
        sample = sample.to_dict()
    if not isinstance(sample, dict):
        msg = f'Sample {index} must be a dict of named media, but got {type(sample).__name__}.'
        raise InvalidSampleError(msg)
    missing = [key for key in required if key not in sample]
    if missing:
        msg = f'Sample {index} has no `{missing[0]}` key. The keys {list(required)} are required.'
        raise InvalidSampleError(msg)
    return {str(key): _read_part(value, index, n_frames) for key, value in sample.items()}


def read_input(kind: InputKind, sample: object, index: int, n_frames: int | None = None) -> object:
    """Read one sample of an input kind.

    Args:
        kind:
            The input kind.

        sample:
            The sample.

        index:
            The position of the sample in the input.

        n_frames:
            The number of frames to sample from videos, or `None` to keep videos whole.

    Returns:
        The read sample.
    """
    if kind == 'text':
        return check_text(sample, index)
    if kind == 'image':
        return read_image(sample, index)
    if kind == 'audio':
        return read_audio(sample, index)
    if kind == 'video':
        return read_video(sample, index) if n_frames is None else read_video_frames(sample, index, n_frames)
    return read_multimodal(sample, index, n_frames)


def sample_frames(video: Video, n_frames: int) -> VideoFrames:
    """Sample evenly spaced frames of a video, starting with the first frame.

    A video with fewer frames than `n_frames` gives all of its frames.

    Args:
        video:
            The video to sample.

        n_frames:
            The number of frames to sample.

    Returns:
        The sampled frames, their times, the length of the video, and its mono audio track when it has one.
    """
    frames = video.frames
    count = min(n_frames, frames.shape[0])
    indices = np.arange(count) * frames.shape[0] // count
    images = tuple(Image.fromarray(frames[index]) for index in indices)
    timestamps = tuple(float(index / video.fps) for index in indices)
    audio = None if video.audio is None else video.audio_track()
    return VideoFrames(images, timestamps, video.duration, audio)


def write_outputs(
    items: list[Any],
    kind: MediaKind,
    output_format: OutputFormat,
    output_dir: str | os.PathLike | None,
    prefix: str,
) -> NDArray[np.object_]:
    """Return generated media in the requested output format.

    Args:
        items:
            The generated media objects. `None` marks a failed sample and is kept as `None`.

        kind:
            The kind of media.

        output_format:
            `'object'` returns the objects, `'bytes'` returns encoded bytes, and `'path'` writes new files and returns
            their paths.

        output_dir:
            The directory for `'path'` outputs. A new temporary directory is used when it is `None`.

        prefix:
            The prefix of the written file names.

    Returns:
        The outputs in input order.
    """
    if output_format == 'object':
        return to_object_array(items)
    if output_format == 'bytes':
        return to_object_array(None if item is None else _write_bytes(item, kind) for item in items)
    directory = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix='skai-'))
    directory.mkdir(parents=True, exist_ok=True)
    return to_object_array(None if item is None else _write_file(item, kind, directory, prefix) for item in items)
