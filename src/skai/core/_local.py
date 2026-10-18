"""Run deterministic media operations locally, without a model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
import textwrap
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from ._media import Audio, Video
from ._requests import Transcript

SAMPLE_RATE = 16000
HISTOGRAM_BINS = 16
ENVELOPE_SECONDS = 0.05
SMOOTHING_SECONDS = 0.3
SUBTITLE_MARGIN = 0.05
SUBTITLE_WIDTH = 40
SENTENCE = re.compile(r'[^.!?]+(?:[.!?]+|$)\s*')
PARAGRAPH = re.compile(r'[^\n](?:[^\n]|\n(?!\s*\n))*')


def _rate_of(tracks: list[Audio]) -> int:
    """Return the sample rate of the first track, or the default rate."""
    return tracks[0].sample_rate if tracks else SAMPLE_RATE


def _fit_length(samples: NDArray[np.floating], length: int) -> NDArray[np.float32]:
    """Cut or pad samples with silence to a length."""
    samples = np.asarray(samples, dtype=np.float32)[:length]
    return np.pad(samples, (0, length - samples.shape[0]))


def concatenate_audio(tracks: list[Audio], gap: float = 0.0) -> Audio:
    """Join audio tracks in order, mixed to mono at the rate of the first one.

    Args:
        tracks:
            The tracks to join.

        gap:
            The seconds of silence between consecutive tracks.

    Returns:
        The joined audio.
    """
    rate = _rate_of(tracks)
    silence = np.zeros(round(gap * rate), dtype=np.float32)
    parts: list[NDArray[np.float32]] = []
    for position, track in enumerate(tracks):
        if position and silence.size:
            parts.append(silence)
        parts.append(track.resample(rate).samples.astype(np.float32))
    return Audio(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), rate)


def concatenate_video(clips: list[Video]) -> Video:
    """Join video clips in order, at the size and frame rate of the first one.

    Args:
        clips:
            The clips to join.

    Returns:
        The joined video, with the joined audio tracks when any clip has one.

    Raises:
        ValueError:
            If no clips are given.
    """
    if not clips:
        msg = 'Concatenation needs at least one clip.'
        raise ValueError(msg)
    first = clips[0]
    height, width = first.frames.shape[1:3]
    fps = first.fps
    frames = []
    for clip in clips:
        resized = clip.resize((width, height))
        count = max(1, round(clip.duration * fps))
        indices = np.minimum((np.arange(count) * clip.fps / fps).astype(int), resized.frames.shape[0] - 1)
        frames.append(resized.frames[indices])
    audio = None
    if any(clip.audio is not None for clip in clips):
        audio = concatenate_audio([_audio_of(clip) for clip in clips])
    return Video(np.concatenate(frames), fps, audio)


def _audio_of(clip: Video) -> Audio:
    """Return the mono track of a clip, or silence as long as the clip, at its exact length."""
    track = clip.audio_track()
    return Audio(_fit_length(track.samples, round(clip.duration * track.sample_rate)), track.sample_rate)


def extract_audio(video: Video) -> Audio:
    """Return the mono audio track of a video, or silence as long as the video.

    Args:
        video:
            The video.

    Returns:
        The audio track.
    """
    return video.audio_track()


def _envelope(samples: NDArray[np.floating], rate: int) -> NDArray[np.float64]:
    """Return a smoothed RMS envelope of samples, one value per sample."""
    window = max(1, round(ENVELOPE_SECONDS * rate))
    power = np.convolve(np.asarray(samples, dtype=np.float64) ** 2, np.ones(window) / window, mode='same')
    smoothing = max(1, round(SMOOTHING_SECONDS * rate))
    return np.convolve(np.sqrt(power), np.ones(smoothing) / smoothing, mode='same')


def mix_audio(
    tracks: dict[str, Audio],
    *,
    volumes: dict[str, float] | None = None,
    duck_under: list[str] | None = None,
    duck_for: str | None = None,
    duck_level: float = 0.25,
    threshold: float = 0.01,
) -> Audio:
    """Mix named tracks, lowering some of them while another one is not silent.

    Args:
        tracks:
            The tracks to mix, by name.

        volumes:
            The gain of every track, one when it is missing.

        duck_under:
            The names of the tracks to lower.

        duck_for:
            The name of the track whose sound lowers the others.

        duck_level:
            The gain of the lowered tracks while `duck_for` sounds.

        threshold:
            The RMS level above which `duck_for` counts as sounding.

    Returns:
        A mono track as long as the longest one, clipped to `[-1, 1]`.
    """
    rate = _rate_of(list(tracks.values()))
    mono = {name: track.resample(rate).samples.astype(np.float64) for name, track in tracks.items()}
    length = max((samples.shape[0] for samples in mono.values()), default=0)
    volumes = volumes or {}
    gains = np.ones(length)
    if duck_for is not None and duck_for in mono and duck_under:
        sounding = _envelope(_fit_length(mono[duck_for], length), rate) > threshold
        weight = np.convolve(sounding.astype(float), np.ones(max(1, round(SMOOTHING_SECONDS * rate))), mode='same')
        gains = 1.0 - (1.0 - duck_level) * np.clip(weight / max(1, round(SMOOTHING_SECONDS * rate)), 0, 1)
    mixed = np.zeros(length)
    for name, samples in mono.items():
        track = _fit_length(samples, length).astype(np.float64) * volumes.get(name, 1.0)
        mixed += track * gains if duck_under and name in duck_under else track
    return Audio(np.clip(mixed, -1.0, 1.0).astype(np.float32), rate)


def mux(video: Video, audio: Audio) -> Video:
    """Replace the soundtrack of a video, cut or padded to the length of the video.

    Args:
        video:
            The video.

        audio:
            The new soundtrack.

    Returns:
        The video with the new soundtrack.
    """
    mono = audio.to_mono()
    samples = _fit_length(mono.samples, round(video.duration * mono.sample_rate))
    return Video(video.frames, video.fps, Audio(samples, mono.sample_rate), subtitles=video.subtitles)


def to_cues(transcript: Any, duration: float) -> list[tuple[float, float, str]]:
    """Turn a transcript into timed subtitle cues.

    Args:
        transcript:
            A string shown for the whole duration, a `Transcript`, or a list of dicts or segments with `start`, `end`
            and `text`.

        duration:
            The length of the video in seconds.

    Returns:
        The cues as `(start, end, text)`.
    """
    if isinstance(transcript, Transcript):
        transcript = list(transcript.segments) or transcript.text
    if isinstance(transcript, str):
        return [(0.0, duration, transcript)] if transcript.strip() else []
    cues = []
    for item in transcript:
        values = item if isinstance(item, dict) else vars(item)
        cues.append((float(values['start']), float(values['end']), str(values.get('text', ''))))
    return cues


def _timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def to_srt(cues: list[tuple[float, float, str]]) -> str:
    """Write subtitle cues in SRT format.

    Args:
        cues:
            The cues as `(start, end, text)`.

    Returns:
        The SRT text.
    """
    blocks = [
        f'{number}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n'
        for number, (start, end, text) in enumerate(cues, start=1)
    ]
    return '\n'.join(blocks)


def subtitle(video: Video, transcript: Any, burn: bool = True) -> Video:
    """Show a transcript as subtitles, burned into the frames or attached in SRT format.

    Args:
        video:
            The video.

        transcript:
            The transcript, in any form that `to_cues` accepts.

        burn:
            Whether to draw the subtitles on the frames. Otherwise they are attached as `subtitles`.

    Returns:
        The subtitled video.
    """
    cues = to_cues(transcript, video.duration)
    if not burn:
        return Video(video.frames, video.fps, video.audio, subtitles=to_srt(cues))
    frames = video.frames.copy()
    font = ImageFont.load_default()
    for index in range(frames.shape[0]):
        time = index / video.fps
        texts = [text for start, end, text in cues if start <= time < end]
        if not texts:
            continue
        image = Image.fromarray(frames[index])
        draw = ImageDraw.Draw(image)
        lines = textwrap.wrap(' '.join(texts), SUBTITLE_WIDTH)
        line_height = draw.textbbox((0, 0), 'Ag', font=font)[3] + 2
        top = image.height - round(SUBTITLE_MARGIN * image.height) - line_height * len(lines)
        for number, line in enumerate(lines):
            width = draw.textlength(line, font=font)
            position = ((image.width - width) / 2, top + number * line_height)
            draw.text(position, line, fill='white', font=font, stroke_width=1, stroke_fill='black')
        frames[index] = np.asarray(image)
    return Video(frames, video.fps, video.audio, subtitles=to_srt(cues))


def collage(images: list[Image.Image], columns: int | None = None, tile: int = 256) -> Image.Image:
    """Lay images out in a grid of square tiles.

    Args:
        images:
            The images.

        columns:
            The number of columns, the square root of the number of images rounded up when it is `None`.

        tile:
            The side of every tile in pixels.

    Returns:
        The collage, with black where there are no images.
    """
    columns = columns or max(1, math.ceil(math.sqrt(len(images))))
    rows = max(1, math.ceil(len(images) / columns))
    canvas = Image.new('RGB', (columns * tile, rows * tile))
    for index, image in enumerate(images):
        thumbnail = image.convert('RGB')
        thumbnail.thumbnail((tile, tile))
        row, column = divmod(index, columns)
        left = column * tile + (tile - thumbnail.width) // 2
        top = row * tile + (tile - thumbnail.height) // 2
        canvas.paste(thumbnail, (left, top))
    return canvas


def frame_at(video: Video, at: float) -> Image.Image:
    """Return the frame of a video at a time.

    Args:
        video:
            The video.

        at:
            The time in seconds, clamped to the video.

    Returns:
        The frame.
    """
    index = min(video.frames.shape[0] - 1, max(0, int(at * video.fps)))
    return Image.fromarray(video.frames[index])


def shot_spans(video: Video, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Find the shots of a video from the histogram distance between consecutive frames.

    Args:
        video:
            The video.

        threshold:
            The total variation distance between the color histograms of consecutive frames above which a cut is
            found, from 0 to 1.

    Returns:
        The `(start, end)` seconds of every shot, covering the video in order.
    """
    frames = video.frames
    histograms = []
    for frame in frames:
        counts = [np.histogram(frame[..., channel], bins=HISTOGRAM_BINS, range=(0, 256))[0] for channel in range(3)]
        histogram = np.concatenate(counts).astype(float)
        histograms.append(histogram / histogram.sum())
    cuts = [
        index
        for index in range(1, len(histograms))
        if 0.5 * np.abs(histograms[index] - histograms[index - 1]).sum() > threshold
    ]
    bounds = [0, *cuts, frames.shape[0]]
    return [(start / video.fps, end / video.fps) for start, end in pairwise(bounds)]


def sound_spans(audio: Audio, threshold: float = 0.01, min_silence: float = 0.3) -> list[tuple[float, float]]:
    """Find the parts of audio between silences.

    Args:
        audio:
            The audio.

        threshold:
            The RMS level below which audio counts as silent.

        min_silence:
            The shortest silence in seconds that separates two parts.

    Returns:
        The `(start, end)` seconds of every part with sound, in order.
    """
    mono = audio.to_mono()
    window = max(1, round(ENVELOPE_SECONDS * mono.sample_rate))
    count = mono.samples.shape[0] // window
    if count == 0:
        return []
    blocks = np.asarray(mono.samples[: count * window], dtype=np.float64).reshape(count, window)
    sounding = np.sqrt((blocks**2).mean(axis=1)) > threshold
    spans: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate([*sounding.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    merged: list[tuple[int, int]] = []
    gap = min_silence * mono.sample_rate / window
    for span in spans:
        if merged and span[0] - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    seconds = window / mono.sample_rate
    return [(start * seconds, min(mono.duration, end * seconds)) for start, end in merged]


def text_spans(text: str, by: str, chunk_size: int = 1000) -> list[tuple[int, int]]:
    """Split a text into paragraphs, sentences or fixed-size chunks.

    Args:
        text:
            The text.

        by:
            `'paragraph'`, `'sentence'` or `'chunk'`.

        chunk_size:
            The number of characters of a chunk.

    Returns:
        The `(start, end)` character offsets of every piece. Chunks cover the text exactly.
    """
    if by == 'chunk':
        return [(start, min(start + chunk_size, len(text))) for start in range(0, len(text), chunk_size)]
    pattern = PARAGRAPH if by == 'paragraph' else SENTENCE
    return [(match.start(), match.end()) for match in pattern.finditer(text) if match.group().strip()]


def crop_to_aspect(video: Video, aspect_ratio: float, centers: list[float]) -> Video:
    """Crop a video to an aspect ratio, following horizontal subject centers over time.

    Args:
        video:
            The video.

        aspect_ratio:
            The width divided by the height of the output.

        centers:
            The horizontal center of the subject, from 0 to 1, at evenly spaced times. An empty list centers the crop.

    Returns:
        The cropped video.
    """
    frames = video.frames
    count, height, width = frames.shape[:3]
    crop_width = min(width, round(height * aspect_ratio))
    crop_height = min(height, round(crop_width / aspect_ratio))
    times = np.linspace(0, 1, num=max(1, len(centers)))
    positions = np.interp(np.linspace(0, 1, num=count), times, centers or [0.5])
    top = (height - crop_height) // 2
    cropped = []
    for frame, center in zip(frames, positions, strict=True):
        left = int(np.clip(round(center * width - crop_width / 2), 0, width - crop_width))
        cropped.append(frame[top : top + crop_height, left : left + crop_width])
    return Video(np.stack(cropped), video.fps, video.audio)


def cut_audio(audio: Audio, start: float, end: float) -> Audio:
    """Return the part of audio between two times.

    Args:
        audio:
            The audio.

        start:
            The start time in seconds.

        end:
            The end time in seconds.

    Returns:
        The part of the audio.
    """
    first, last = round(start * audio.sample_rate), round(end * audio.sample_rate)
    return Audio(audio.samples[first:last], audio.sample_rate)


def cut_video(video: Video, start: float, end: float) -> Video:
    """Return the part of a video between two times, with its audio.

    Args:
        video:
            The video.

        start:
            The start time in seconds.

        end:
            The end time in seconds.

    Returns:
        The part of the video, with at least one frame.
    """
    count = video.frames.shape[0]
    first = min(count - 1, max(0, round(start * video.fps)))
    last = max(first + 1, min(count, round(end * video.fps)))
    audio = None if video.audio is None else cut_audio(video.audio, first / video.fps, last / video.fps)
    return Video(video.frames[first:last], video.fps, audio)


def place_audio(clips: list[tuple[float, Audio]], length: float = 0.0) -> Audio:
    """Place audio clips at their start times on one silent track.

    Args:
        clips:
            Pairs of a start time in seconds and a clip. Overlapping clips are added.

        length:
            The minimum length of the track in seconds.

    Returns:
        A mono track at the rate of the first clip, long enough for every clip.
    """
    rate = _rate_of([clip for _, clip in clips])
    mono = [(round(start * rate), clip.resample(rate).samples.astype(np.float64)) for start, clip in clips]
    size = max([round(length * rate), *(start + samples.shape[0] for start, samples in mono)])
    track = np.zeros(size)
    for start, samples in mono:
        track[start : start + samples.shape[0]] += samples
    return Audio(np.clip(track, -1.0, 1.0).astype(np.float32), rate)
