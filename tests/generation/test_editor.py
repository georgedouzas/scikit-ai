"""Test the image editor."""

import pytest
from PIL import Image
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import HFBackend
from skai.core import UnsupportedBackendError
from skai.generation import ImageEditor
from tests.fakes import install_module


def test_predict_keeps_input_sizes():
    """Return one edited image per input, of the input's size."""
    editor = ImageEditor(instruction='add snow', backend='dummy').fit()
    assert [image.size for image in editor.predict([Image.new('RGB', (32, 24)), Image.new('RGB', (8, 8))])] == [
        (32, 24),
        (8, 8),
    ]


def test_mask_limits_the_edit():
    """Leave the image unchanged outside the mask."""
    image = Image.new('RGB', (8, 8), 'black')
    mask = Image.new('L', (4, 4), 0)
    mask.paste(255, (0, 0, 2, 4))
    edited = ImageEditor(instruction='add snow', mask=mask, backend='dummy').fit().predict([image])[0]
    assert edited.getpixel((7, 0)) == (0, 0, 0)
    assert edited.getpixel((0, 0)) != (0, 0, 0)


def test_fit_requires_instruction():
    """Reject a missing `instruction` at `fit`."""
    with pytest.raises(InvalidParameterError, match='instruction'):
        ImageEditor(backend='dummy').fit()


def test_fit_rejects_mask_without_support(monkeypatch):
    """List the backends that support masks when the backend does not."""
    install_module(monkeypatch, 'transformers', pipeline=None)
    with pytest.raises(UnsupportedBackendError, match=r"cannot edit inside a mask.*\['dummy', 'openai'\]"):
        ImageEditor(instruction='x', mask=Image.new('L', (4, 4)), backend=HFBackend()).fit()
