"""Test the image animator."""

from PIL import Image

from skai.generation import ImageAnimator


def test_predict_animates_images():
    """Return one video of the requested duration whose first frame matches the image size."""
    animator = ImageAnimator(motion_prompt='slow zoom', duration=2, backend='dummy').fit()
    video = animator.predict([Image.new('RGB', (32, 24), 'teal')])[0]
    assert video.duration == 2.0
    assert video.frames[0].shape[:2] == (24, 32)
