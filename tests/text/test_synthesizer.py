"""Test the speech synthesizer."""

from skai.core import Audio
from skai.text import SpeechSynthesizer


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
