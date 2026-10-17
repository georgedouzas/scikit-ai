"""Components whose input is text."""

from ._classifier import TextClassifier
from ._embedder import TextEmbedder
from ._image_generator import ImageGenerator
from ._summarizer import TextSummarizer
from ._synthesizer import SpeechSynthesizer
from ._translator import TextTranslator

__all__: list[str] = [
    'ImageGenerator',
    'SpeechSynthesizer',
    'TextClassifier',
    'TextEmbedder',
    'TextSummarizer',
    'TextTranslator',
]
