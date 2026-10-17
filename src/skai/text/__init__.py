"""Components that turn texts into other texts."""

from ._summarizer import TextSummarizer
from ._translator import TextTranslator

__all__: list[str] = ['TextSummarizer', 'TextTranslator']
