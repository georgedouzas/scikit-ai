"""Hold decoded audio and video."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import io
from fractions import Fraction
from pathlib import Path
from typing import Any, Self

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ._utils import import_optional

SILENCE_SAMPLE_RATE = 16000
VIDEO_FORMAT = 'mp4'
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'


def _encode_frames(container: Any, stream: Any, frames: NDArray[np.uint8]) -> None:
    """Encode RGB frames into a video stream of an open container."""
    av = import_optional('av', 'video')
    for array in frames[:, : stream.height, : stream.width]:
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format='rgb24')):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def _encode_samples(container: Any, stream: Any, audio: 'Audio') -> None:
    """Encode mono audio samples into an audio stream of an open container."""
    av = import_optional('av', 'video')
    samples = np.ascontiguousarray(audio.to_mono().samples.astype(np.float32).reshape(1, -1))
    frame = av.AudioFrame.from_ndarray(samples, format='flt', layout='mono')
    frame.sample_rate = audio.sample_rate
    frame.pts = 0
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def _encode_video(frames: NDArray[np.uint8], fps: float, audio: 'Audio | None') -> bytes:
    """Encode frames and an optional audio track as MP4 bytes."""
    av = import_optional('av', 'video')
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format=VIDEO_FORMAT) as container:
        video_stream = container.add_stream(VIDEO_CODEC, rate=Fraction(fps).limit_denominator(1000))
        video_stream.height, video_stream.width = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
        video_stream.pix_fmt = 'yuv420p'
        audio_stream = None if audio is None else container.add_stream(AUDIO_CODEC, rate=audio.sample_rate)
        _encode_frames(container, video_stream, frames)
        if audio is not None:
            _encode_samples(container, audio_stream, audio)
    return buffer.getvalue()


def _decode_video(data: bytes) -> tuple[NDArray[np.uint8], float, 'Audio | None']:
    """Decode MP4 bytes into frames, frame rate and an optional audio track."""
    av = import_optional('av', 'video')
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(stream)])
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            return frames, fps, None
        stream = container.streams.audio[0]
        chunks = [frame.to_ndarray() for frame in container.decode(stream)]
        samples = np.concatenate([chunk.reshape(-1, chunk.shape[-1]) for chunk in chunks], axis=1)
        return frames, fps, Audio(samples.mean(axis=0), int(stream.rate))


class Audio:
    """Hold audio samples and their sample rate.

    Args:
        samples:
            The audio samples, with shape `(n_samples,)` for mono or `(n_samples, n_channels)`.

        sample_rate:
            The number of samples per second.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Audio
        >>> audio = Audio(np.zeros(8000), 16000)
        >>> audio.duration
        0.5
    """

    def __init__(self, samples: NDArray[np.floating], sample_rate: int) -> None:
        self.samples = np.asarray(samples)
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        """Return a short description of the audio."""
        return f'Audio(duration={self.duration:.2f}s, sample_rate={self.sample_rate})'

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Decode audio from encoded bytes.

        Args:
            data:
                The encoded audio, in any format that libsndfile reads.

        Returns:
            The decoded audio.
        """
        samples, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
        return cls(samples, int(sample_rate))

    @property
    def duration(self) -> float:
        """The length of the audio in seconds."""
        return float(self.samples.shape[0] / self.sample_rate)

    def to_mono(self) -> 'Audio':
        """Return the audio mixed down to one channel.

        Returns:
            The mono audio.
        """
        if self.samples.ndim == 1:
            return self
        return Audio(self.samples.mean(axis=1), self.sample_rate)

    def to_bytes(self) -> bytes:
        """Encode the audio as WAV bytes.

        Returns:
            The WAV-encoded audio.
        """
        buffer = io.BytesIO()
        sf.write(buffer, self.samples, self.sample_rate, format='WAV')
        return buffer.getvalue()

    def save(self, path: str | Path) -> Path:
        """Write the audio to a WAV file.

        Args:
            path:
                The path of the file to write.

        Returns:
            The path of the written file.
        """
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path


class Video:
    """Hold a video as decoded frames or as encoded MP4 bytes.

    Either `frames` and `fps` or `data` must be given. The other representation is computed when it is first needed,
    which requires the `video` extra.

    Args:
        frames:
            The RGB frames, with shape `(n_frames, height, width, 3)` and type `uint8`.

        fps:
            The number of frames per second.

        audio:
            The audio track.

        data:
            The MP4-encoded video.

    Raises:
        ValueError:
            If neither `frames` with `fps` nor `data` are given.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> video = Video(np.zeros((50, 64, 64, 3), dtype=np.uint8), fps=25)
        >>> video.duration
        2.0
    """

    def __init__(
        self,
        frames: NDArray[np.uint8] | None = None,
        fps: float | None = None,
        audio: Audio | None = None,
        *,
        data: bytes | None = None,
    ) -> None:
        if data is None and (frames is None or fps is None):
            msg = 'Provide either `frames` and `fps`, or the encoded `data`.'
            raise ValueError(msg)
        self._frames = frames
        self._fps = fps
        self._audio = audio
        self._data = data

    def __repr__(self) -> str:
        """Return a short description of the video."""
        if self._frames is None:
            return f'Video(encoded_bytes={len(self._data or b"")})'
        return f'Video(duration={self.duration:.2f}s, fps={self._fps})'

    def _decode(self) -> None:
        """Decode the encoded bytes into frames, frame rate and audio."""
        if self._frames is None and self._data is not None:
            self._frames, self._fps, self._audio = _decode_video(self._data)

    @property
    def frames(self) -> NDArray[np.uint8]:
        """The RGB frames of the video."""
        self._decode()
        assert self._frames is not None
        return self._frames

    @property
    def fps(self) -> float:
        """The number of frames per second."""
        self._decode()
        assert self._fps is not None
        return self._fps

    @property
    def audio(self) -> Audio | None:
        """The audio track of the video."""
        self._decode()
        return self._audio

    @property
    def duration(self) -> float:
        """The length of the video in seconds."""
        return self.frames.shape[0] / self.fps

    def audio_track(self) -> Audio:
        """Return the audio track mixed to mono, or silence as long as the video when it has no track.

        Returns:
            The mono audio track.
        """
        if self.audio is None:
            return Audio(np.zeros(round(self.duration * SILENCE_SAMPLE_RATE), dtype=np.float32), SILENCE_SAMPLE_RATE)
        return self.audio.to_mono()

    def to_bytes(self) -> bytes:
        """Encode the video as MP4 bytes.

        Returns:
            The MP4-encoded video.
        """
        if self._data is None:
            self._data = _encode_video(self.frames, self.fps, self._audio)
        return self._data

    def save(self, path: str | Path) -> Path:
        """Write the video to an MP4 file.

        Args:
            path:
                The path of the file to write.

        Returns:
            The path of the written file.
        """
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path
