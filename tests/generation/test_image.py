"""Test the generators that return images."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend, HFBackend, ReplicateBackend
from skai.core import InvalidSampleError, UnsupportedBackendError, Video
from skai.generation import (
    ImageToImageGenerator,
    MultimodalToImageGenerator,
    TextToImageGenerator,
    VideoToImageGenerator,
)
from tests.fakes import install_module, namespace

PHOTO = Image.new('RGB', (64, 48), (90, 120, 150))


class RecordingBackend(DummyBackend):
    """Keep every request the backend receives."""

    requests: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_requests():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []


def test_text_to_image_returns_requested_size():
    """Return one PIL image of the requested size per prompt."""
    images = TextToImageGenerator(size=(512, 256), backend='dummy').fit().predict(['a cat', 'a dog'])
    assert all(isinstance(image, Image.Image) for image in images)
    assert [image.size for image in images] == [(512, 256)] * 2


@pytest.mark.parametrize('size', [(0, 5), (1, 2, 3), (1.5, 2)])
def test_text_to_image_rejects_invalid_size(size):
    """Reject sizes that are not a pair of positive integers."""
    with pytest.raises(InvalidParameterError, match='size'):
        TextToImageGenerator(size=size, backend='dummy').fit()


def test_text_to_image_differs_by_seed():
    """Return equal images for equal seeds and different images for different seeds."""

    def generate(random_state):
        """Return the image of one prompt with a seed."""
        generator = TextToImageGenerator(size=(4, 4), random_state=random_state, backend='dummy')
        return generator.fit().predict(['a cat'])[0].tobytes()

    assert generate(0) == generate(0)
    assert generate(0) != generate(1)


def test_seeded_generation():
    """Request generation with the same seed in two runs with `random_state=0`."""
    for _ in range(2):
        TextToImageGenerator(size=(4, 4), random_state=0, backend=RecordingBackend()).fit().predict(['a cat'])
    first, second = RecordingBackend.requests
    assert first.seed == second.seed


def test_text_to_image_sends_style_and_negative_prompt():
    """Send the style, the negative prompt and the reference images with every request."""
    references = [Image.new('RGB', (8, 8), 'red')]
    generator = TextToImageGenerator(
        size=(4, 4),
        style='flat',
        negative_prompt='blur',
        reference_images=references,
        backend=RecordingBackend(),
    )
    generator.fit().predict(['a cat', 'a dog'])
    assert [(request.style, request.negative_prompt) for request in RecordingBackend.requests] == [('flat', 'blur')] * 2
    assert all(len(request.reference_images) == 1 for request in RecordingBackend.requests)


def test_text_to_image_rejects_reference_images_without_support():
    """List the backends that take reference images when the backend does not."""

    class PlainBackend(DummyBackend):
        """Generate images without reference images."""

        supports_reference_images: ClassVar[bool] = False

    generator = TextToImageGenerator(reference_images=[PHOTO], backend=PlainBackend())
    with pytest.raises(UnsupportedBackendError, match=r"cannot take reference images.*\['dummy', 'openai'\]"):
        generator.fit()


def test_learned_style():
    """Learn a style from the style images, send it with every request, and let `style` override it."""
    watercolors = [Image.new('RGB', (8, 8), color) for color in ('red', 'blue', 'green')]
    generator = TextToImageGenerator(size=(4, 4), style_images=watercolors, backend=RecordingBackend()).fit()
    assert generator.style_
    assert generator.style_.count('A 8x8 image.') == 3
    RecordingBackend.requests = []
    generator.predict(['a cat', 'a dog'])
    assert [request.style for request in RecordingBackend.requests] == [generator.style_] * 2
    explicit = TextToImageGenerator(size=(4, 4), style='flat', style_images=watercolors, backend=RecordingBackend())
    explicit.fit().predict(['a cat'])
    assert RecordingBackend.requests[-1].style == 'flat'
    assert TextToImageGenerator(backend='dummy').fit().style_ is None


def test_style_images_need_image_description():
    """Reject style images on a backend that cannot describe images."""
    generator = TextToImageGenerator(style_images=[PHOTO], backend=ReplicateBackend(client=namespace()))
    with pytest.raises(UnsupportedBackendError, match='cannot describe images'):
        generator.fit()


def test_image_operations_keep_or_set_sizes():
    """Keep the size when editing, double it when upscaling and add transparency when removing the background."""
    edited = ImageToImageGenerator(operation='edit', instructions='add snow', backend='dummy').fit().predict([PHOTO])[0]
    upscaled = ImageToImageGenerator(operation='upscale', scale=2, backend='dummy').fit().predict([PHOTO])[0]
    cutout = ImageToImageGenerator(operation='remove_background', backend='dummy').fit().predict([PHOTO])[0]
    assert edited.size == (64, 48)
    assert upscaled.size == (128, 96)
    assert (cutout.mode, cutout.size) == ('RGBA', (64, 48))
    assert cutout.getpixel((0, 0))[3] == 0
    assert cutout.getpixel((32, 24))[3] == 255


def test_edit_keeps_every_input_size():
    """Return one edited image per input, of the input's size, with the instructions in every request."""
    editor = ImageToImageGenerator(instructions='add snow', backend=RecordingBackend()).fit()
    images = editor.predict([Image.new('RGB', (32, 24)), Image.new('RGB', (8, 8))])
    assert [image.size for image in images] == [(32, 24), (8, 8)]
    assert [request.instruction for request in RecordingBackend.requests] == ['add snow'] * 2


def test_mask_limits_the_edit():
    """Leave the image unchanged outside the mask."""
    image = Image.new('RGB', (8, 8), 'black')
    mask = Image.new('L', (4, 4), 0)
    mask.paste(255, (0, 0, 2, 4))
    edited = ImageToImageGenerator(instructions='add snow', mask=mask, backend='dummy').fit().predict([image])[0]
    assert edited.getpixel((7, 0)) == (0, 0, 0)
    assert edited.getpixel((0, 0)) != (0, 0, 0)


def test_edit_requires_instructions():
    """Reject an edit without `instructions` at `fit`, and accept other operations without them."""
    with pytest.raises(InvalidParameterError, match='instructions'):
        ImageToImageGenerator(backend='dummy').fit()
    ImageToImageGenerator(operation='upscale', backend='dummy').fit()


def test_edit_rejects_mask_without_support(monkeypatch):
    """List the backends that support masks when the backend does not."""
    install_module(monkeypatch, 'transformers', pipeline=None)
    generator = ImageToImageGenerator(instructions='x', mask=Image.new('L', (4, 4)), backend=HFBackend())
    with pytest.raises(UnsupportedBackendError, match=r"cannot edit inside a mask.*\['dummy', 'openai'\]"):
        generator.fit()


def test_collage_lays_images_out():
    """Lay a list of images out in a grid of square tiles, without a backend."""
    images = [Image.new('RGB', (40, 20), color) for color in ('red', 'green', 'blue')]
    square = ImageToImageGenerator(operation='collage', tile=16).fit()
    assert square.backend_ is None
    assert square.predict([images])[0].size == (32, 32)
    row = ImageToImageGenerator(operation='collage', columns=3, tile=16).fit().predict([images, images[:1]])
    assert [image.size for image in row] == [(48, 16), (48, 16)]
    assert row[0].getpixel((8, 8)) == (255, 0, 0)


def test_video_frame_and_thumbnail():
    """Return the frame at `at` seconds locally, and a sampled frame chosen by the backend."""
    frames = np.stack([np.full((6, 8, 3), value, dtype=np.uint8) for value in range(0, 160, 10)])
    video = Video(frames, fps=4)
    frame = VideoToImageGenerator(operation='frame', at=1.5).fit().predict([video])[0]
    assert frame.size == (8, 6)
    assert frame.getpixel((0, 0)) == (60, 60, 60)
    thumbnail = VideoToImageGenerator(n_frames=4, backend=RecordingBackend()).fit().predict([video])[0]
    assert thumbnail.size == (8, 6)
    (request,) = RecordingBackend.requests
    assert len(request.frames.frames) == 4
    assert any(np.array_equal(np.asarray(thumbnail), sampled) for sampled in request.frames.frames)


def test_edit_text_per_image():
    """Edit every image with the instructions and the mask of its own sample."""
    mask = Image.new('L', (64, 48), 0)
    mask.paste(255, (0, 0, 32, 48))
    generator = MultimodalToImageGenerator(operation='edit', backend=RecordingBackend()).fit()
    samples = [
        {'image': PHOTO, 'instructions': 'Write Γεια here', 'mask': mask},
        {'image': Image.new('RGB', (16, 8)), 'instructions': 'Add a hat'},
    ]
    images = generator.predict(samples)
    assert [image.size for image in images] == [(64, 48), (16, 8)]
    first, second = RecordingBackend.requests
    assert first.instruction == 'Write Γεια here'
    assert first.mask is not None
    assert np.array_equal(np.asarray(first.mask.convert('L')), np.asarray(mask))
    assert (second.instruction, second.mask) == ('Add a hat', None)
    assert images[0].getpixel((63, 0)) == PHOTO.getpixel((63, 0))


def test_edit_per_sample_needs_image_and_instructions():
    """Name the missing key and the index of the sample."""
    generator = MultimodalToImageGenerator(backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match=r'Sample 1.*`instructions`'):
        generator.predict([{'image': PHOTO, 'instructions': 'x'}, {'image': PHOTO}])


def test_consistent_characters():
    """Generate one reference image from the prompt at `fit` and send it with every request."""
    generator = TextToImageGenerator(
        size=(8, 8),
        reference_prompt='a girl with a red scarf',
        backend=RecordingBackend(),
    )
    generator.fit()
    assert len(generator.reference_images_) == 1
    assert isinstance(generator.reference_images_[0], Image.Image)
    RecordingBackend.requests = []
    generator.predict(['she reads', 'she sleeps'])
    assert len(RecordingBackend.requests) == 2
    assert all(request.reference_images == generator.reference_images_ for request in RecordingBackend.requests)


def test_explicit_reference_images_take_precedence():
    """Use the given reference images instead of generating them from the prompt."""
    references = [Image.new('RGB', (4, 4), 'red'), Image.new('RGB', (4, 4), 'blue')]
    generator = TextToImageGenerator(
        size=(8, 8),
        reference_images=references,
        reference_prompt='a girl',
        backend=RecordingBackend(),
    ).fit()
    assert len(generator.reference_images_) == 2
    assert [image.getpixel((0, 0)) for image in generator.reference_images_] == [(255, 0, 0), (0, 0, 255)]
    assert TextToImageGenerator(backend='dummy').fit().reference_images_ == ()
