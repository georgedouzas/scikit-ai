"""Components that assign labels to texts and images."""

from ._image import ImageClassifier
from ._text import TextClassifier

__all__: list[str] = ['ImageClassifier', 'TextClassifier']
