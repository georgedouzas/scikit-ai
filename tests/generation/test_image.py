"""Test the image generator."""

from typing import ClassVar

import pytest
from PIL import Image
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend, ReplicateBackend
from skai.core import UnsupportedBackendError
from skai.generation import ImageGenerator
from tests.fakes import namespace


def test_predict_returns_requested_size():
    """Return one image of the requested size per prompt."""
    images = ImageGenerator(size=(512, 512), backend='dummy').fit().predict(['a cat', 'a dog'])
    assert [image.size for image in images] == [(512, 512)] * 2


@pytest.mark.parametrize('size', [(0, 5), (1, 2, 3), (1.5, 2)])
def test_fit_rejects_invalid_size(size):
    """Reject sizes that are not a pair of positive integers."""
    with pytest.raises(InvalidParameterError, match='size'):
        ImageGenerator(size=size, backend='dummy').fit()


def test_predict_differs_by_seed():
    """Return different images for different seeds and equal images for equal seeds."""
    first = ImageGenerator(size=(4, 4), random_state=0, backend='dummy').fit().predict(['a cat'])[0]
    second = ImageGenerator(size=(4, 4), random_state=0, backend='dummy').fit().predict(['a cat'])[0]
    other = ImageGenerator(size=(4, 4), random_state=1, backend='dummy').fit().predict(['a cat'])[0]
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != other.tobytes()


class StyleRecordingBackend(DummyBackend):
    """Keep the style of every image generation request."""

    styles: ClassVar[list] = []

    def generate_image(self, request):
        """Keep the style and generate like the dummy backend."""
        StyleRecordingBackend.styles.append(request.style)
        return super().generate_image(request)


def test_fit_learns_style_from_reference_images():
    """Learn a style from the reference images, send it with every request, and let `style` override it."""
    references = [Image.new('RGB', (8, 8), color) for color in ('red', 'blue', 'green')]
    generator = ImageGenerator(size=(4, 4), style_images=references, backend=StyleRecordingBackend()).fit()
    assert generator.style_.count('A brief description') == 3
    generator.predict(['a cat'])
    assert StyleRecordingBackend.styles[-1] == generator.style_
    explicit = ImageGenerator(size=(4, 4), style='flat', style_images=references, backend=StyleRecordingBackend())
    explicit.fit().predict(['a cat'])
    assert StyleRecordingBackend.styles[-1] == 'flat'
    assert ImageGenerator(backend='dummy').fit().style_ is None


def test_fit_rejects_style_images_without_description():
    """Reject style images on a backend that cannot describe images."""
    generator = ImageGenerator(style_images=[Image.new('RGB', (4, 4))], backend=ReplicateBackend(client=namespace()))
    with pytest.raises(UnsupportedBackendError, match='cannot describe images'):
        generator.fit()
