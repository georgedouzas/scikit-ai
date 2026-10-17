"""Test the image descriptor."""

from PIL import Image
from sklearn.pipeline import make_pipeline

from skai.image import ImageDescriptor
from skai.text import TextClassifier

IMAGES = [Image.new('RGB', (size, size)) for size in (8, 16, 24, 32)]


def test_predict_describes_every_image():
    """Return one description string per image."""
    descriptions = ImageDescriptor(backend='dummy').fit().predict(IMAGES[:2])
    assert descriptions.dtype == object
    assert descriptions.tolist() == ['A brief description of a 8x8 image.', 'A brief description of a 16x16 image.']


def test_descriptions_feed_text_classifier():
    """Classify images through their descriptions in a pipeline."""
    pipeline = make_pipeline(ImageDescriptor(backend='dummy'), TextClassifier(backend='dummy'))
    predictions = pipeline.fit(IMAGES, ['small', 'small', 'large', 'large']).predict(IMAGES)
    assert len(predictions) == 4
    assert set(predictions) <= {'small', 'large'}
