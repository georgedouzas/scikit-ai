"""Components whose input is an image."""

from ._classifier import ImageClassifier
from ._descriptor import ImageDescriptor

__all__: list[str] = ['ImageClassifier', 'ImageDescriptor']
