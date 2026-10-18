"""Test the text extractor."""

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend
from skai.core import InvalidSampleError
from skai.extraction import TextExtractor

FIELDS = {'name': 'person name', 'city': 'city of residence'}
STORYBOARD = {'scenes': [{'narration': 'spoken text', 'visual': 'what is shown', 'seconds': 'length'}]}


def recording_backend(answer=None):
    """Return a dummy backend that records its extraction requests and can answer with a fixed record."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def extract_fields(self, request):
            """Record the request and answer it."""
            requests.append(request)
            return dict(answer) if answer is not None else super().extract_fields(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


def test_predict_returns_records_with_every_field():
    """Return one dict per text with exactly the field names."""
    records = TextExtractor(fields=FIELDS, backend='dummy').fit().predict(['Ada lives in London.', 'Bob, Paris.'])
    assert records.shape == (2,)
    assert all(set(record) == {'name', 'city'} for record in records)


def test_predict_keeps_exactly_the_fields_with_none_for_absent_ones():
    """Drop undeclared keys of the answer and fill absent fields with `None`."""
    backend = recording_backend({'name': 'Ada', 'extra': 'dropped'})
    record = TextExtractor(fields=FIELDS, backend=backend).fit().predict(['Ada.'])[0]
    assert record == {'name': 'Ada', 'city': None}


@pytest.mark.parametrize('fields', [None, 'name', ['name']])
def test_fit_requires_fields(fields):
    """Reject a missing or non-dict `fields` at `fit`."""
    with pytest.raises(InvalidParameterError, match='fields'):
        TextExtractor(fields=fields, backend='dummy').fit()


def test_storyboard_list_of_records():
    """Return a Python list of dicts with exactly the subfield names for a list field of records."""
    record = TextExtractor(fields=STORYBOARD, backend='dummy').fit().predict(['A brief about a sunny beach day.'])[0]
    assert set(record) == {'scenes'}
    assert isinstance(record['scenes'], list)
    assert record['scenes']
    assert all(set(scene) == {'narration', 'visual', 'seconds'} for scene in record['scenes'])


def test_nested_and_list_value_fields():
    """Return a dict for a nested field and a list for a list of values."""
    fields = {'address': {'street': 'street name', 'number': 'house number'}, 'tags': ['a keyword']}
    record = TextExtractor(fields=fields, backend='dummy').fit().predict(['Main street 5, quiet and green.'])[0]
    assert set(record['address']) == {'street', 'number'}
    assert isinstance(record['tags'], list)
    assert all(isinstance(tag, str) for tag in record['tags'])


def test_request_carries_fields_and_instructions():
    """Send the declarations in order and the instructions to the backend."""
    backend = recording_backend()
    TextExtractor(fields=FIELDS, instructions='Use full names.', backend=backend).fit().predict(['Ada.'])
    request = backend.requests[0]
    assert request.fields == (('name', 'person name'), ('city', 'city of residence'))
    assert request.instructions == 'Use full names.'


def test_transform_returns_columns_in_field_order():
    """Return a 2-D object array with one column per top-level field."""
    extractor = TextExtractor(fields={'b': 'second', 'a': 'first', **STORYBOARD}, backend='dummy').fit()
    table = extractor.transform(['One two three.', 'Four five six.'])
    assert table.shape == (2, 3)
    assert table.dtype == object
    assert isinstance(table[0, 2], list)
    assert extractor.get_feature_names_out().tolist() == ['b', 'a', 'scenes']


def test_transform_returns_dataframe_with_named_columns():
    """Return a DataFrame with one column per field when pandas output is set."""
    extractor = TextExtractor(fields=FIELDS, backend='dummy').set_output(transform='pandas')
    frame = extractor.fit_transform(['The name is Ada and the city is London.', ''])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ['name', 'city']
    assert frame.shape == (2, 2)
    assert frame.iloc[1].isna().all()


def test_fit_keeps_reference_records():
    """Keep text and record pairs as examples."""
    extractor = TextExtractor(fields=FIELDS, backend='dummy').fit(['Ada, London.'], [{'name': 'Ada', 'city': 'London'}])
    assert extractor.examples_ == (('Ada, London.', {'name': 'Ada', 'city': 'London'}),)


def test_fit_sends_k_shot_examples():
    """Select `k_shot` pairs and send them with every request."""
    backend = recording_backend()
    texts = [f'Person {index} lives in city {index}.' for index in range(5)]
    records = [{'name': f'P{index}', 'city': f'C{index}'} for index in range(5)]
    extractor = TextExtractor(fields=FIELDS, k_shot=2, random_state=0, backend=backend).fit(texts, records)
    assert len(extractor.examples_) == 2
    assert all(records[texts.index(text)] == record for text, record in extractor.examples_)
    extractor.predict(['x'])
    assert backend.requests[0].examples == extractor.examples_


def test_fit_caps_examples_at_max_examples():
    """Pass at most `max_examples` pairs when `k_shot` is `None`."""
    texts = [f'text {index}' for index in range(6)]
    records = [{'name': str(index), 'city': str(index)} for index in range(6)]
    extractor = TextExtractor(fields=FIELDS, max_examples=3, backend='dummy').fit(texts, records)
    assert len(extractor.examples_) == 3
    assert TextExtractor(fields=FIELDS, k_shot=0, backend='dummy').fit(texts, records).examples_ == ()


def test_fit_rejects_mismatched_records():
    """Reject records whose number differs from the number of texts."""
    with pytest.raises(ValueError, match='2 samples but `y` has 1 records'):
        TextExtractor(fields=FIELDS, backend='dummy').fit(['a', 'b'], [{'name': 'a'}])


def test_predict_rejects_list_valued_samples():
    """Reject a list sample and point to `ForEach`."""
    extractor = TextExtractor(fields=FIELDS, backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match='ForEach'):
        extractor.predict([['a', 'b']])


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to predict before `fit`."""
    extractor = TextExtractor(fields=STORYBOARD, instructions='Be short.', k_shot=1, backend='dummy')
    cloned = clone(extractor)
    assert cloned.get_params() == extractor.get_params()
    with pytest.raises(NotFittedError):
        cloned.predict(['x'])
    with pytest.raises(NotFittedError):
        cloned.get_feature_names_out()
