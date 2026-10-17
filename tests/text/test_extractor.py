"""Test the text extractor."""

import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.text import TextExtractor

FIELDS = {'name': 'person name', 'city': 'city of residence'}


def test_predict_returns_records_with_every_field():
    """Return one dict per text with exactly the field names."""
    records = TextExtractor(fields=FIELDS, backend='dummy').fit().predict(['Ada lives in London.'])
    assert set(records[0]) == {'name', 'city'}


def test_fit_requires_fields():
    """Reject a missing `fields` at `fit`."""
    with pytest.raises(InvalidParameterError, match='fields'):
        TextExtractor(backend='dummy').fit()


def test_transform_returns_columns_with_names():
    """Return a DataFrame with one column per field when pandas output is set."""
    extractor = TextExtractor(fields=FIELDS, backend='dummy').set_output(transform='pandas')
    frame = extractor.fit_transform(['The name is Ada and the city is London.', 'Nothing here.'])
    assert list(frame.columns) == ['name', 'city']
    assert frame.shape == (2, 2)
    assert frame.iloc[1].isna().all()


def test_fit_keeps_reference_records():
    """Keep text and record pairs as examples."""
    extractor = TextExtractor(fields=FIELDS, backend='dummy').fit(['Ada, London.'], [{'name': 'Ada', 'city': 'London'}])
    assert extractor.examples_ == (('Ada, London.', {'name': 'Ada', 'city': 'London'}),)
