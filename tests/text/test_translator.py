"""Test the text translator."""

import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.text import TextTranslator


def test_fit_requires_target_language():
    """Reject a missing `target_language` at `fit`."""
    with pytest.raises(InvalidParameterError, match='target_language'):
        TextTranslator(backend='dummy').fit()


def test_predict_translates_every_text():
    """Return one translation per text."""
    translator = TextTranslator(target_language='Spanish', backend='dummy').fit()
    assert translator.predict(['Hello', 'Bye']).tolist() == ['[Spanish] Hello', '[Spanish] Bye']
