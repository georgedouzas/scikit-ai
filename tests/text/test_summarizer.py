"""Test the text summarizer."""

from skai.core import Backend
from skai.text import TextSummarizer


class VerboseBackend(Backend):
    """Return summaries that are longer than requested."""

    def summarize(self, request):
        """Return the text twice."""
        return f'{request.text} {request.text}'


def test_predict_respects_max_length():
    """Return summaries of at most `max_length` words."""
    text = ' '.join(f'word{index}' for index in range(200))
    summary = TextSummarizer(max_length=50, backend='dummy').fit().predict([text])[0]
    assert len(summary.split()) <= 50


def test_predict_cuts_long_backend_summaries():
    """Cut summaries that the backend returns longer than `max_length`."""
    summary = TextSummarizer(max_length=3, backend=VerboseBackend()).fit().predict(['one two'])[0]
    assert summary == 'one two one'
