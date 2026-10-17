"""Find the domain terms of a text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re

TOKEN = re.compile(r'[A-Za-z0-9][\w.+#-]*[\w+#]')
SENTENCE_END = re.compile(r'[.!?]\s+')


def _is_term(token: str, starts_sentence: bool) -> bool:
    """Decide whether a token looks like a name, a product or a technical term."""
    if any(character.isdigit() for character in token):
        return True
    if any(character.isupper() for character in token[1:]):
        return True
    return token[0].isupper() and not starts_sentence


def find_terms(text: str) -> list[str]:
    """Find the terms of a text that look like names, products or technical terms.

    A term is a token of two or more characters that contains a digit, has an upper-case letter after its first
    character, or starts with an upper-case letter in the middle of a sentence.

    Args:
        text:
            The text to search.

    Returns:
        The terms in order of first appearance, with repeats.

    Examples:
        >>> from skai.core import find_terms
        >>> find_terms('We deploy on Kubernetes with gRPC and Python 3.12.')
        ['Kubernetes', 'gRPC', 'Python', '3.12']
    """
    terms = []
    for sentence in SENTENCE_END.split(text):
        for position, match in enumerate(TOKEN.finditer(sentence)):
            if _is_term(match.group(), starts_sentence=position == 0):
                terms.append(match.group())
    return terms
