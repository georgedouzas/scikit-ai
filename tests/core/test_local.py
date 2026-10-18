"""Test the local media operations that run without a model."""

import re

import numpy as np
import pytest
from PIL import Image

from skai.core import (
    Audio,
    Segment,
    Transcript,
    Video,
    collage,
    concatenate_audio,
    concatenate_video,
    crop_to_aspect,
    cut_audio,
    cut_video,
    extract_audio,
    frame_at,
    mix_audio,
    mux,
    place_audio,
    shot_spans,
    sound_spans,
    subtitle,
    text_spans,
    to_cues,
    to_srt,
)

RATE = 16000
SRT_TIMESTAMP = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$')


def tone(seconds, amplitude=0.1, rate=RATE):
    """Return a sine tone."""
    times = np.arange(round(seconds * rate)) / rate
    return Audio((amplitude * np.sin(2 * np.pi * 440 * times)).astype(np.float32), rate)


def silence(seconds, rate=RATE):
    """Return silence."""
    return Audio(np.zeros(round(seconds * rate), dtype=np.float32), rate)


def rms(samples):
    """Return the RMS level of samples."""
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def clip(seconds, fps=10, size=(8, 6), value=0, audio=None):
    """Return a video of one gray level."""
    width, height = size
    frames = np.full((round(seconds * fps), height, width, 3), value, dtype=np.uint8)
    return Video(frames, fps, audio)


def test_concatenate_audio_adds_durations_and_gaps():
    """Join tracks at the rate of the first one, with silence between them."""
    joined = concatenate_audio([tone(1.0), tone(0.5, rate=8000)], gap=0.25)
    assert joined.sample_rate == RATE
    assert joined.duration == pytest.approx(1.75)
    assert rms(joined.samples[RATE : RATE + RATE // 4]) == 0
    assert concatenate_audio([]).duration == 0


def test_concatenate_video_adds_durations_at_the_first_size():
    """Join clips at the size and frame rate of the first clip, joining their audio."""
    joined = concatenate_video([clip(1.0, audio=tone(1.0)), clip(0.5, fps=20, size=(16, 12), value=255)])
    assert joined.duration == pytest.approx(1.5)
    assert joined.frames.shape[1:3] == (6, 8)
    assert joined.fps == 10
    assert joined.frames[-1].mean() == 255
    assert joined.audio is not None
    assert joined.audio.duration == pytest.approx(1.5)
    assert concatenate_video([clip(1.0), clip(1.0)]).audio is None
    with pytest.raises(ValueError, match='at least one clip'):
        concatenate_video([])


def test_extract_audio_returns_the_track_or_silence(audio):
    """Return the mono track of a video, or silence as long as a silent video."""
    assert extract_audio(clip(1.0, audio=audio)).duration == pytest.approx(1.0)
    silent = extract_audio(clip(2.0))
    assert silent.duration == pytest.approx(2.0)
    assert rms(silent.samples) == 0


def test_mix_audio_is_as_long_as_the_longest_track():
    """Mix tracks with their volumes into one track as long as the longest one."""
    mixed = mix_audio({'a': tone(1.0), 'b': tone(2.0)}, volumes={'b': 0.0})
    assert mixed.duration == pytest.approx(2.0)
    assert rms(mixed.samples[RATE:]) == 0
    assert rms(mixed.samples[:RATE]) == pytest.approx(rms(tone(1.0).samples), rel=1e-3)


def test_ducking_makes_music_quieter_where_the_voice_sounds():
    """Lower the music while the voice sounds and keep it while the voice is silent."""
    voice = concatenate_audio([silence(1.0), tone(1.0, amplitude=0.3), silence(1.0)])
    music = tone(3.0, amplitude=0.2)
    ducked = mix_audio({'voice': voice, 'music': music}, volumes={'voice': 0.0}, duck_under=['music'], duck_for='voice')
    plain = mix_audio({'voice': voice, 'music': music}, volumes={'voice': 0.0})
    during = slice(round(1.3 * RATE), round(1.7 * RATE))
    before = slice(0, round(0.5 * RATE))
    assert rms(ducked.samples[during]) == pytest.approx(0.25 * rms(plain.samples[during]), rel=0.05)
    assert rms(ducked.samples[before]) == pytest.approx(rms(plain.samples[before]), rel=0.01)


def test_mux_fits_the_soundtrack_to_the_video():
    """Cut or pad the new soundtrack to the length of the video, keeping the frames."""
    video = clip(2.0)
    assert mux(video, tone(3.0)).audio.duration == pytest.approx(2.0)
    padded = mux(video, tone(0.5))
    assert padded.audio.duration == pytest.approx(2.0)
    assert rms(padded.audio.samples[RATE:]) == 0
    np.testing.assert_array_equal(padded.frames, video.frames)


def test_to_cues_accepts_every_transcript_form():
    """Turn strings, transcripts and lists of segments or dicts into cues."""
    segments = (Segment(0.0, 1.0, 'Hello.'), Segment(1.0, 2.5, 'World.'))
    assert to_cues('All of it.', 3.0) == [(0.0, 3.0, 'All of it.')]
    assert to_cues('  ', 3.0) == []
    assert to_cues(Transcript('Hello. World.', segments), 3.0) == [(0.0, 1.0, 'Hello.'), (1.0, 2.5, 'World.')]
    assert to_cues(Transcript('Only text.'), 2.0) == [(0.0, 2.0, 'Only text.')]
    assert to_cues([{'start': 0, 'end': 1, 'text': 'Hi.'}], 2.0) == [(0.0, 1.0, 'Hi.')]


def test_to_srt_writes_numbered_timed_blocks():
    """Write numbered blocks with SRT timestamps separated by blank lines."""
    srt = to_srt([(0.0, 1.5, 'Hello.'), (3661.25, 3662.0, 'Later.')])
    assert srt == '1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n01:01:01,250 --> 01:01:02,000\nLater.\n'
    blocks = srt.strip().split('\n\n')
    assert all(SRT_TIMESTAMP.match(block.split('\n')[1]) for block in blocks)
    assert to_srt([]) == ''


def test_subtitle_burns_text_only_during_cues():
    """Draw the text on the frames of its cue and attach the cues in SRT format."""
    video = clip(2.0, size=(160, 120))
    burned = subtitle(video, [{'start': 0.0, 'end': 1.0, 'text': 'Hello there'}])
    assert burned.frames[:10].max() > 0
    assert burned.frames[10:].max() == 0
    assert burned.subtitles.startswith('1\n00:00:00,000 --> 00:00:01,000\nHello there')
    attached = subtitle(video, 'Hello there', burn=False)
    np.testing.assert_array_equal(attached.frames, video.frames)
    assert 'Hello there' in attached.subtitles


def test_collage_lays_images_out_in_square_tiles():
    """Lay images out in rows of tiles, with black where there are no images."""
    images = [Image.new('RGB', (20, 10), 'red') for _ in range(3)]
    grid = collage(images, tile=16)
    assert grid.size == (32, 32)
    assert grid.getpixel((8, 8)) == (255, 0, 0)
    assert grid.getpixel((24, 24)) == (0, 0, 0)
    assert collage(images, columns=3, tile=16).size == (48, 16)


def test_frame_at_clamps_the_time():
    """Return the frame at a time, clamped to the video."""
    frames = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in (0, 100, 200)])
    video = Video(frames, fps=1)
    assert np.asarray(frame_at(video, 1.5)).mean() == 100
    assert np.asarray(frame_at(video, -1)).mean() == 0
    assert np.asarray(frame_at(video, 10)).mean() == 200


def test_shot_spans_find_cuts():
    """Find a cut where the colors change and cover the video in order."""
    video = concatenate_video([clip(1.0, value=0), clip(1.5, value=255), clip(0.5, value=0)])
    assert shot_spans(video) == [(0.0, 1.0), (1.0, 2.5), (2.5, 3.0)]
    assert shot_spans(clip(2.0)) == [(0.0, 2.0)]


def test_sound_spans_split_on_silences():
    """Find the parts with sound, merging those separated by short silences."""
    audio = concatenate_audio([tone(1.0), silence(1.0), tone(0.5), silence(0.1), tone(0.5), silence(0.5)])
    spans = sound_spans(audio)
    assert len(spans) == 2
    assert spans[0] == pytest.approx((0.0, 1.0), abs=0.06)
    assert spans[1] == pytest.approx((2.0, 3.1), abs=0.06)
    assert len(sound_spans(audio, min_silence=0.05)) == 3
    assert sound_spans(silence(1.0)) == []
    assert sound_spans(silence(0.01)) == []


@pytest.mark.parametrize('by', ['paragraph', 'sentence', 'chunk'])
def test_text_spans_join_back_to_the_text(by):
    """Split into pieces whose text, joined, gives back the text."""
    text = 'First sentence. Second one!\n\nA new paragraph? Yes.'
    spans = text_spans(text, by, chunk_size=7)
    assert ''.join(text[start:end] for start, end in spans).replace('\n', '') == text.replace('\n', '')
    assert all(start < end for start, end in spans)


def test_text_spans_find_paragraphs_sentences_and_chunks():
    """Find two paragraphs, four sentences and chunks of the given size."""
    text = 'First sentence. Second one!\n\nA new paragraph? Yes.'
    paragraphs = [text[start:end] for start, end in text_spans(text, 'paragraph')]
    sentences = [text[start:end].strip() for start, end in text_spans(text, 'sentence')]
    chunks = text_spans(text, 'chunk', chunk_size=20)
    assert paragraphs == ['First sentence. Second one!', 'A new paragraph? Yes.']
    assert sentences == ['First sentence.', 'Second one!', 'A new paragraph?', 'Yes.']
    assert chunks == [(0, 20), (20, 40), (40, len(text))]
    assert ''.join(text[start:end] for start, end in chunks) == text


def test_crop_to_aspect_follows_the_subject():
    """Crop to the aspect ratio, moving the crop with the subject center."""
    frames = np.zeros((3, 90, 160, 3), dtype=np.uint8)
    frames[:, :, :80] = 255
    cropped = crop_to_aspect(Video(frames, fps=3), 9 / 16, [0.0, 1.0])
    assert cropped.frames.shape == (3, 90, 51, 3)
    assert cropped.frames[0].mean() == 255
    assert cropped.frames[-1].mean() == 0
    centered = crop_to_aspect(Video(frames, fps=3), 1.0, [])
    assert centered.frames.shape == (3, 90, 90, 3)


def test_cut_audio_and_video_keep_the_span():
    """Return the part between two times, with the audio of a video cut alike."""
    assert cut_audio(tone(2.0), 0.5, 1.25).duration == pytest.approx(0.75)
    part = cut_video(clip(2.0, audio=tone(2.0)), 0.5, 1.5)
    assert part.duration == pytest.approx(1.0)
    assert part.audio.duration == pytest.approx(1.0)
    assert cut_video(clip(1.0), 0.95, 0.95).frames.shape[0] == 1


def test_place_audio_puts_clips_at_their_times():
    """Place clips on one silent track at their start times."""
    track = place_audio([(1.0, tone(0.5)), (0.0, tone(0.25))], length=2.0)
    assert track.duration == pytest.approx(2.0)
    assert rms(track.samples[round(0.5 * RATE) : RATE]) == 0
    assert rms(track.samples[RATE : round(1.5 * RATE)]) > 0
    assert place_audio([(1.0, tone(1.0))]).duration == pytest.approx(2.0)
