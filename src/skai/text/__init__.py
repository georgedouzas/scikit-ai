"""Components that turn texts into other texts and records."""

from ._extractor import TextExtractor
from ._summarizer import TextSummarizer
from ._translator import TextTranslator

__all__: list[str] = [
    'TextExtractor',
    'TextSummarizer',
    'TextTranslator',
]
