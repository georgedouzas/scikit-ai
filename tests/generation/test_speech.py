"""Test the speech synthesizer."""

import pytest

from skai.backends import OpenAIBackend
from skai.core import Audio, UnsupportedBackendError
from skai.generation import SpeechSynthesizer
from tests.fakes import namespace


def test_predict_returns_one_audio_per_text():
    """Return one audio output per text."""
    speech = SpeechSynthesizer(backend='dummy').fit().predict(['Hello', 'Bye'])
    assert len(speech) == 2
    assert all(isinstance(audio, Audio) for audio in speech)
    assert speech[0].duration > 0


def test_fit_reads_reference_voice(tmp_path, audio):
    """Read reference audio for voice cloning on a backend that clones voices."""
    for voice in [audio, (audio.samples, audio.sample_rate), audio.to_bytes(), audio.save(tmp_path / 'voice.wav')]:
        synthesizer = SpeechSynthesizer(voice=voice, backend='dummy').fit()
        assert isinstance(synthesizer.voice_, Audio)
    assert SpeechSynthesizer(voice='nova', backend='dummy').fit().voice_ == 'nova'


def test_fit_joins_voice_recordings(audio):
    """Join several recordings into one reference voice at the first recording's rate."""
    other = Audio(audio.samples[::2], 8000)
    synthesizer = SpeechSynthesizer(voice=[audio, other], backend='dummy').fit()
    assert synthesizer.voice_.sample_rate == 16000
    assert synthesizer.voice_.duration == pytest.approx(2.3, abs=0.01)


def test_fit_rejects_empty_voice_list():
    """Reject an empty list of recordings."""
    with pytest.raises(ValueError, match='at least one recording'):
        SpeechSynthesizer(voice=[], backend='dummy').fit()


def test_fit_rejects_recordings_without_cloning(audio):
    """Reject recordings on a backend that cannot clone voices."""
    with pytest.raises(UnsupportedBackendError, match='cannot clone voices'):
        SpeechSynthesizer(voice=[audio], backend=OpenAIBackend(client=namespace())).fit()
