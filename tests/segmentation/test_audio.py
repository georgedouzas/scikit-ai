"""Test the audio segmenter."""

import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend
from skai.core import Audio, InvalidSampleError
from skai.segmentation import AudioSegmenter

SAMPLE_RATE = 16000
TONE = (0.2 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE)).astype(np.float32)
GAPPED = (np.concatenate([TONE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32), TONE]), SAMPLE_RATE)
TRANSCRIPTS = ['We deployed Kubernetes with Terraform on AWS.', 'Kubernetes runs on AWS.']
PROVIDERS = ('openai', 'transformers', 'litellm', 'torch', 'replicate')


def recording_backend():
    """Return a dummy backend that records its transcription requests in `requests`."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def transcribe(self, request):
            """Record the request and transcribe like the dummy backend."""
            requests.append(request)
            return super().transcribe(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


def test_timed_speech_by_default(audio):
    """Return timed speech with ordered times, text and the audio of the region."""
    regions = AudioSegmenter(backend='dummy').fit().predict([audio, audio])
    assert len(regions) == 2
    for region in regions[0]:
        assert region['label'] == 'speech'
        assert 0 <= region['start'] <= region['end'] <= audio.duration
        assert region['text']
        assert isinstance(region['media'], Audio)
    assert regions[0][0]['text'] == 'Speech of 1.00 seconds.'


def test_speaker_turns(audio):
    """Return speaker turns with `start <= end`, a speaker label and text."""
    regions = AudioSegmenter(by='speaker', backend='dummy').fit().predict([audio])[0]
    assert len(regions) == 2
    for region in regions:
        assert region['start'] <= region['end']
        assert region['speaker']
        assert region['text']
        assert region['media'].duration == pytest.approx(region['end'] - region['start'], abs=1e-3)
    assert [region['speaker'] for region in regions] == ['SPEAKER_0', 'SPEAKER_1']


def test_silence_splits_the_parts_with_sound():
    """Return the two parts with sound around a half-second silence, found locally."""
    segmenter = AudioSegmenter(by='silence').fit()
    assert segmenter.backend_ is None
    regions = segmenter.predict([GAPPED])[0]
    assert [(region['start'], region['end']) for region in regions] == [(0.0, 0.5), (1.0, 1.5)]
    assert all(region['media'].duration == pytest.approx(0.5) for region in regions)


def test_silence_threshold_and_min_silence():
    """Keep one part when the silence is shorter than `min_silence`, and none when all is below `threshold`."""
    assert len(AudioSegmenter(by='silence', min_silence=1.0).fit().predict([GAPPED])[0]) == 1
    assert AudioSegmenter(by='silence', threshold=1.0).fit().predict([GAPPED])[0] == []


def test_silence_imports_no_provider():
    """Split at silences without importing any provider."""
    before = set(sys.modules)
    AudioSegmenter(by='silence').fit().predict([GAPPED])
    new = set(sys.modules) - before
    assert not [name for name in new if name.split('.')[0] in PROVIDERS]


def test_moments_match_the_prompt(audio):
    """Return the moments the backend finds for `prompt`, and require a prompt."""
    regions = AudioSegmenter(by='moment', prompt='a laugh', backend='dummy').fit().predict([audio])[0]
    assert len(regions) == 1
    assert regions[0]['label'] == 'moment'
    assert regions[0]['start'] < regions[0]['end']
    with pytest.raises(ValueError, match='needs `prompt`'):
        AudioSegmenter(by='moment', backend='dummy').fit()


def test_vocabulary_learned_from_y_and_sent_with_every_request(audio):
    """Learn the terms of reference transcripts and send them as context."""
    backend = recording_backend()
    segmenter = AudioSegmenter(language='en', backend=backend).fit([audio, audio], TRANSCRIPTS)
    assert {'Kubernetes', 'Terraform', 'AWS'} <= set(segmenter.vocabulary_)
    segmenter.predict([audio, audio])
    assert len(backend.requests) == 2
    assert all(request.vocabulary == segmenter.vocabulary_ for request in backend.requests)
    assert all(request.language == 'en' and request.timestamps == 'segment' for request in backend.requests)


def test_vocabulary_respects_max_vocabulary_and_use_y(audio):
    """Cap the vocabulary, skip it without `y`, and ignore `y` in `fit_transform` unless `use_y=True`."""
    assert len(AudioSegmenter(max_vocabulary=1, backend='dummy').fit([audio] * 2, TRANSCRIPTS).vocabulary_) == 1
    assert AudioSegmenter(backend='dummy').fit().vocabulary_ == ()
    assert AudioSegmenter(use_y=False, backend='dummy').fit([audio] * 2, TRANSCRIPTS).vocabulary_ == ()
    auto = AudioSegmenter(backend='dummy')
    auto.fit_transform([audio] * 2, TRANSCRIPTS)
    assert auto.vocabulary_ == ()
    forced = AudioSegmenter(use_y=True, backend='dummy')
    forced.fit_transform([audio] * 2, TRANSCRIPTS)
    assert 'Kubernetes' in forced.vocabulary_


def test_invalid_by_names_the_parameter_and_choices():
    """Reject a `by` that audio does not offer and name the valid values."""
    with pytest.raises(InvalidParameterError, match="'by' parameter") as error:
        AudioSegmenter(by='shot', backend='dummy').fit()
    assert all(repr(value) in str(error.value) for value in ['moment', 'silence', 'speaker', 'speech'])


def test_rejects_list_valued_samples(audio):
    """Reject a list sample and point to `ForEach`."""
    with pytest.raises(InvalidSampleError, match='ForEach'):
        AudioSegmenter(backend='dummy').fit().predict([[audio, audio]])


def test_clone_and_not_fitted(audio):
    """Clone with the same parameters and refuse to predict before `fit`."""
    segmenter = AudioSegmenter(by='speaker', language='el', backend='dummy')
    cloned = clone(segmenter)
    assert cloned.get_params() == segmenter.get_params()
    with pytest.raises(NotFittedError):
        cloned.predict([audio])


def test_timed_words() -> None:
    """With `by='word'`, every region holds one word, in time order."""
    regions = AudioSegmenter(by='word', backend='dummy').fit().predict([(np.zeros(16000), 16000)])[0]
    assert all(len(region['text'].split()) == 1 for region in regions)
    starts = [region['start'] for region in regions]
    assert starts == sorted(starts)
