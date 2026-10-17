"""Test the speech transcriber."""

import pytest

from skai.audio import SpeechTranscriber
from skai.backends import DummyBackend
from skai.core import UnsupportedBackendError


class SegmentOnlyBackend(DummyBackend):
    """Return segment timestamps but not word timestamps."""

    timestamp_granularities = frozenset({'segment'})


def test_predict_returns_plain_transcripts(audio):
    """Return one transcript string per audio sample."""
    transcripts = SpeechTranscriber(backend='dummy').fit().predict([audio, audio])
    assert transcripts.tolist() == ['Speech of 1.00 seconds.'] * 2


@pytest.mark.parametrize('timestamps', ['segment', 'word'])
def test_predict_returns_timed_records(audio, timestamps):
    """Return records with ordered times and non-empty text."""
    records = SpeechTranscriber(timestamps=timestamps, backend='dummy').fit().predict([audio])[0]
    assert records
    assert all(record['start'] <= record['end'] and record['text'] for record in records)


def test_fit_rejects_unsupported_granularity():
    """Reject word timestamps on a backend without word timing."""
    transcriber = SpeechTranscriber(timestamps='word', backend=SegmentOnlyBackend())
    with pytest.raises(UnsupportedBackendError, match='cannot return `word` timestamps'):
        transcriber.fit()
    SpeechTranscriber(timestamps='segment', backend=SegmentOnlyBackend()).fit()
