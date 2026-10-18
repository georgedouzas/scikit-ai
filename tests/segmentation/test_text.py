"""Test the text segmenter."""

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from skai.core import InvalidSampleError
from skai.segmentation import TextSegmenter

TEXT = 'First paragraph. It has two sentences.\n\nSecond paragraph! Is it short?'


def test_chunks_join_back_to_the_text():
    """Split a 250-character text into three chunks that join back to it."""
    text = ''.join(chr(ord('a') + index % 26) for index in range(250))
    regions = TextSegmenter(by='chunk', chunk_size=100).fit().predict([text])[0]
    assert len(regions) == 3
    assert ''.join(region['media'] for region in regions) == text
    assert [(region['start'], region['end']) for region in regions] == [(0, 100), (100, 200), (200, 250)]


def test_paragraphs_and_sentences_carry_offsets():
    """Return regions whose offsets locate their passage in the text."""
    paragraphs = TextSegmenter().fit().predict([TEXT])[0]
    sentences = TextSegmenter(by='sentence').fit().predict([TEXT])[0]
    assert len(paragraphs) == 2
    assert len(sentences) == 4
    for region in [*paragraphs, *sentences]:
        assert TEXT[region['start'] : region['end']] == region['media']
        assert region['score'] == 1.0
    assert {region['label'] for region in paragraphs} == {'paragraph'}
    assert {region['label'] for region in sentences} == {'sentence'}


def test_runs_locally_without_a_backend():
    """Split texts without resolving a backend."""
    segmenter = TextSegmenter().fit()
    assert segmenter.backend_ is None
    assert segmenter.task_ is None


def test_invalid_by_names_the_parameter_and_choices():
    """Reject an unknown `by` and name the valid values."""
    with pytest.raises(InvalidParameterError, match="'by' parameter") as error:
        TextSegmenter(by='word').fit()
    assert all(repr(value) in str(error.value) for value in ['chunk', 'paragraph', 'sentence'])


def test_rejects_list_valued_samples():
    """Reject a list sample and point to `ForEach`."""
    with pytest.raises(InvalidSampleError, match='ForEach'):
        TextSegmenter().fit().predict([['one', 'two']])


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to predict before `fit`."""
    segmenter = clone(TextSegmenter(by='chunk', chunk_size=10))
    assert segmenter.get_params() == {'by': 'chunk', 'chunk_size': 10}
    with pytest.raises(NotFittedError):
        segmenter.predict([TEXT])
