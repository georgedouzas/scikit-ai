"""Test the generators that return audio."""

from typing import ClassVar

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend, OpenAIBackend
from skai.core import Audio, InvalidSampleError, UnsupportedBackendError, Video
from skai.generation import (
    AudioToAudioGenerator,
    MultimodalToAudioGenerator,
    TextToAudioGenerator,
    VideoToAudioGenerator,
)
from tests.fakes import namespace

RATE = 16000


def tone(seconds, frequency=440.0, level=0.2):
    """Return a tone of a length."""
    times = np.arange(round(seconds * RATE)) / RATE
    return Audio((level * np.sin(2 * np.pi * frequency * times)).astype(np.float32), RATE)


def rms(samples):
    """Return the root mean square of samples."""
    return float(np.sqrt(np.mean(np.square(samples))))


class RecordingBackend(DummyBackend):
    """Keep every request the backend receives."""

    requests: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_requests():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []


def test_speech_returns_one_audio_per_text():
    """Return one non-empty `Audio` per text, with the language in every request."""
    speech = TextToAudioGenerator(language='el', backend=RecordingBackend()).fit().predict(['Hello', 'Bye now'])
    assert len(speech) == 2
    assert all(isinstance(audio, Audio) and audio.duration > 0 for audio in speech)
    assert [request.language for request in RecordingBackend.requests] == ['el', 'el']


def test_music_length():
    """Return music and sound of the requested duration."""
    music = TextToAudioGenerator(operation='music', duration=5, backend='dummy').fit().predict(['calm piano'])
    assert len(music) == 1
    assert isinstance(music[0], Audio)
    assert music[0].duration == pytest.approx(5.0)
    sound = TextToAudioGenerator(operation='sound', duration=1.5, backend='dummy').fit().predict(['rain'])[0]
    assert sound.duration == pytest.approx(1.5)


def test_voice_name_is_kept():
    """Keep a voice name as is."""
    TextToAudioGenerator(voice='nova', backend=RecordingBackend()).fit().predict(['Hi'])
    assert RecordingBackend.requests[-1].voice == 'nova'
    assert TextToAudioGenerator(backend='dummy').fit().voice_ is None


def test_voice_is_read_from_every_form(tmp_path, audio):
    """Read reference audio for cloning from an object, a tuple, bytes and a path."""
    for voice in [audio, (audio.samples, audio.sample_rate), audio.to_bytes(), audio.save(tmp_path / 'voice.wav')]:
        assert isinstance(TextToAudioGenerator(voice=voice, backend='dummy').fit().voice_, Audio)


def test_voice_learned_from_samples(audio):
    """Join the recordings into `voice_` at the first recording's rate and send it with every speech request."""
    other = Audio(audio.samples[::2], 8000)
    synthesizer = TextToAudioGenerator(voice=[audio, other], backend=RecordingBackend()).fit()
    assert synthesizer.voice_.sample_rate == 16000
    assert synthesizer.voice_.duration == pytest.approx(2.3, abs=0.01)
    synthesizer.predict(['Hello', 'Bye'])
    assert all(request.voice is synthesizer.voice_ for request in RecordingBackend.requests)


def test_voice_list_must_not_be_empty():
    """Reject an empty list of recordings."""
    with pytest.raises(ValueError, match='at least one recording'):
        TextToAudioGenerator(voice=[], backend='dummy').fit()


def test_voice_cloning_needs_a_capable_backend(audio):
    """Reject recordings on a backend that cannot clone voices, listing the backends that can."""
    with pytest.raises(UnsupportedBackendError, match=r"cannot clone voices.*\['dummy'"):
        TextToAudioGenerator(voice=[audio], backend=OpenAIBackend(client=namespace())).fit()


def test_separated_vocals():
    """Return one stem as long as the song, with the stem in the request."""
    song = tone(2.5)
    vocals = (
        AudioToAudioGenerator(operation='separate', stem='vocals', backend=RecordingBackend()).fit().predict([song])
    )
    assert len(vocals) == 1
    assert isinstance(vocals[0], Audio)
    assert vocals[0].duration == pytest.approx(song.duration)
    AudioToAudioGenerator(stem='accompaniment', backend=RecordingBackend()).fit().predict([song])
    assert [request.stem for request in RecordingBackend.requests] == ['vocals', 'accompaniment']


def test_enhance_keeps_length():
    """Return a denoised copy as long as the input."""
    noisy = tone(1.2)
    assert AudioToAudioGenerator(operation='enhance', backend='dummy').fit().predict([noisy])[0].duration == 1.2


def test_convert_voice(audio):
    """Learn the target voice from recordings, send it with every request, and keep the length of the input."""
    converter = AudioToAudioGenerator(operation='convert_voice', voice=[audio, audio], backend=RecordingBackend()).fit()
    assert isinstance(converter.voice_, Audio)
    assert converter.voice_.duration == pytest.approx(2.3, abs=0.01)
    output = converter.predict([tone(1.5)])[0]
    assert output.duration == pytest.approx(1.5)
    assert RecordingBackend.requests[-1].voice is converter.voice_


def test_convert_voice_requires_voice():
    """Reject voice conversion without `voice` or with an empty list."""
    with pytest.raises(InvalidParameterError, match='voice'):
        AudioToAudioGenerator(operation='convert_voice', backend='dummy').fit()
    with pytest.raises(ValueError, match='at least one recording'):
        AudioToAudioGenerator(operation='convert_voice', voice=[], backend='dummy').fit()


def test_concatenate_with_gaps():
    """Join a list of audio with `gap` seconds between, without a backend."""
    joiner = AudioToAudioGenerator(operation='concatenate', gap=0.5).fit()
    assert joiner.backend_ is None
    joined = joiner.predict([[tone(1.0), tone(0.5), tone(0.25)], [tone(1.0)]])
    assert [audio.duration for audio in joined] == pytest.approx([2.75, 1.0])


def test_extract_audio_track():
    """Return the mono track of a video, and silence as long as a video without one."""
    stereo = Audio(np.zeros((32000, 2), dtype=np.float32), RATE)
    speaking = Video(np.zeros((16, 4, 4, 3), dtype=np.uint8), fps=8, audio=stereo)
    silent = Video(np.zeros((12, 4, 4, 3), dtype=np.uint8), fps=8)
    tracks = VideoToAudioGenerator().fit().predict([speaking, silent])
    assert [track.duration for track in tracks] == pytest.approx([2.0, 1.5])
    assert tracks[0].samples.ndim == 1
    assert not np.any(tracks[1].samples)


def test_ducked_music():
    """Return a mix as long as the longer track, with the music quieter where the voice speaks."""
    voice = Audio(np.concatenate([tone(1.0, 300).samples, np.zeros(RATE, dtype=np.float32)]), RATE)
    song = tone(3.0, 500)
    mixer = MultimodalToAudioGenerator(duck_under=['music'], duck_for='voice').fit()
    assert mixer.backend_ is None
    mixed = mixer.predict([{'voice': voice, 'music': song}])[0]
    assert mixed.duration == pytest.approx(3.0)
    music = mixed.samples[: voice.samples.shape[0]] - voice.samples
    music = np.concatenate([music, mixed.samples[voice.samples.shape[0] :]])
    assert rms(music[: RATE // 2]) < 0.5 * rms(music[2 * RATE :])


def test_mix_volumes_and_video_tracks():
    """Apply the volume of every track and take the audio track of a video."""
    clip = Video(np.zeros((8, 4, 4, 3), dtype=np.uint8), fps=8, audio=tone(1.0))
    mixed = MultimodalToAudioGenerator(volumes={'clip': 0.5}).fit().predict([{'clip': clip}])[0]
    assert rms(mixed.samples) == pytest.approx(0.5 * rms(tone(1.0).samples), rel=1e-3)


def test_mix_checks_its_samples():
    """Name the missing key of `duck_for` and parts that are not audio."""
    with pytest.raises(InvalidSampleError, match=r'Sample 0.*`voice`'):
        MultimodalToAudioGenerator(duck_under=['music'], duck_for='voice').fit().predict([{'music': tone(1.0)}])
    with pytest.raises(InvalidSampleError, match=r'Sample 0.*`notes`'):
        MultimodalToAudioGenerator().fit().predict([{'music': tone(1.0), 'notes': 'loud'}])


def test_timed_track():
    """Place every timed record of a track at its start."""
    dialogue = Audio(np.zeros(3 * RATE, dtype=np.float32), RATE)
    clip = Audio(np.full(RATE // 2, 0.5, dtype=np.float32), RATE)
    mixed = (
        MultimodalToAudioGenerator()
        .fit()
        .predict([{'dialogue': dialogue, 'description': [{'start': 2.0, 'media': clip}]}])[0]
    )
    assert mixed.duration == pytest.approx(3.0)
    sounding = np.flatnonzero(mixed.samples)
    assert sounding[0] / RATE == pytest.approx(2.0)
    assert sounding[-1] / RATE == pytest.approx(2.5, abs=1 / RATE)


def test_timed_track_extends_the_mix():
    """Make the mix as long as the end of the last timed record."""
    clip = Audio(np.full(RATE, 0.5, dtype=np.float32), RATE)
    sample = {'voice': tone(1.0), 'effects': [{'start': 0.5, 'media': clip}, {'start': 1.5, 'media': clip}]}
    mixed = MultimodalToAudioGenerator().fit().predict([sample])[0]
    assert mixed.duration == pytest.approx(2.5)


def test_voice_per_sample(audio):
    """Speak the text of every sample in its own voice, a recording or a name."""
    generator = MultimodalToAudioGenerator(operation='speech', language='en', backend=RecordingBackend()).fit()
    outputs = generator.predict(
        [{'text': 'Hello', 'voice': audio}, {'text': 'Hi there', 'voice': 'nova'}, {'text': 'Bye'}],
    )
    assert all(isinstance(output, Audio) for output in outputs)
    first, second, third = RecordingBackend.requests
    assert isinstance(first.voice, Audio)
    assert np.array_equal(first.voice.samples, audio.samples)
    assert (first.text, second.text, third.text) == ('Hello', 'Hi there', 'Bye')
    assert (second.voice, third.voice) == ('nova', None)
    assert {request.language for request in RecordingBackend.requests} == {'en'}


def test_speech_per_sample_needs_text():
    """Name the missing `text` key and the index of the sample."""
    with pytest.raises(InvalidSampleError, match=r'Sample 1.*`text`'):
        MultimodalToAudioGenerator(operation='speech', backend='dummy').fit().predict(
            [{'text': 'a'}, {'voice': 'nova'}],
        )
