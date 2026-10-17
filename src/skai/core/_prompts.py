"""Build prompts for prompted backends and parse their answers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import dataclass

from PIL import Image

from ._errors import InvalidResponseError
from ._requests import (
    ClassificationRequest,
    DescriptionRequest,
    ImageGenerationRequest,
    SummaryRequest,
    TranslationRequest,
)

NO_LABEL = 'none'
SINGLE_LABEL_SYSTEM = 'You are a classifier. Answer with exactly one label from the list of labels and nothing else.'
MULTI_LABEL_SYSTEM = (
    'You are a multi-label classifier. Answer with every label from the list of labels that applies, separated by '
    f'commas, and nothing else. Answer `{NO_LABEL}` if no label applies.'
)
SUMMARY_SYSTEM = 'You summarize texts. Answer with the summary only.'
TRANSLATION_SYSTEM = 'You translate texts. Answer with the translation only.'
DESCRIPTION_SYSTEM = 'You describe images. Answer with the description only.'
DETAIL_INSTRUCTIONS = {
    'brief': 'Describe the image in one sentence.',
    'detailed': 'Describe the image in one detailed paragraph.',
}


@dataclass(frozen=True)
class Prompt:
    """Hold the messages of a prompt for a chat or vision model.

    Args:
        system:
            The system message.

        user:
            The user message. Images are referred to as `<image 1>`, `<image 2>` and so on.

        images:
            The images of the user message, in the order they are referred to.
    """

    system: str
    user: str
    images: tuple[Image.Image, ...] = ()


def _normalize(text: str) -> str:
    """Lower-case a label or an answer and strip quotes, spaces and final periods."""
    return text.strip().strip('`"\'').rstrip('.').strip().lower()


def _format_input(sample: str | Image.Image, images: list[Image.Image]) -> str:
    """Format a sample for a prompt, collecting it in `images` when it is an image."""
    if isinstance(sample, str):
        return sample
    images.append(sample)
    return f'<image {len(images)}>'


def _match_label(answer: str, labels: tuple[str, ...]) -> str:
    """Find the label that an answer names."""
    by_name = {_normalize(label): label for label in labels}
    normalized = _normalize(answer)
    if normalized in by_name:
        return by_name[normalized]
    found = [label for name, label in by_name.items() if re.search(rf'\b{re.escape(name)}\b', normalized)]
    if len(found) != 1:
        msg = f'The answer `{answer}` does not name exactly one of the labels {list(labels)}.'
        raise InvalidResponseError(msg)
    return found[0]


def build_classification_prompt(request: ClassificationRequest) -> Prompt:
    """Build the prompt that asks a model to classify a sample.

    Args:
        request:
            The classification request.

    Returns:
        The prompt.
    """
    images: list[Image.Image] = []
    lines = ['Labels:']
    lines += [
        f'- {label}: {request.descriptions[label]}' if label in request.descriptions else f'- {label}'
        for label in request.labels
    ]
    if request.instructions:
        lines += ['', request.instructions]
    for number, (sample, labels) in enumerate(request.examples, start=1):
        answer = ', '.join(labels) or NO_LABEL
        lines += ['', f'Example {number}:', f'Input: {_format_input(sample, images)}', f'Output: {answer}']
    lines += ['', f'Input: {_format_input(request.sample, images)}', 'Output:']
    system = MULTI_LABEL_SYSTEM if request.multi_label else SINGLE_LABEL_SYSTEM
    return Prompt(system, '\n'.join(lines), tuple(images))


def parse_labels(answer: str, labels: tuple[str, ...], multi_label: bool) -> dict[str, float]:
    """Turn the answer of a model into a score per label.

    Args:
        answer:
            The answer of the model.

        labels:
            The candidate labels.

        multi_label:
            Whether the answer can name several labels.

    Returns:
        A score of one for every named label and zero for the others.
    """
    scores = dict.fromkeys(labels, 0.0)
    if not multi_label:
        scores[_match_label(answer, labels)] = 1.0
        return scores
    parts = [part for part in re.split(r'[,\n]', answer) if _normalize(part)]
    for part in parts:
        if _normalize(part) != NO_LABEL:
            scores[_match_label(part, labels)] = 1.0
    return scores


def build_summary_prompt(request: SummaryRequest) -> Prompt:
    """Build the prompt that asks a model to summarize a text.

    Args:
        request:
            The summary request.

    Returns:
        The prompt.
    """
    style = f' Write it in this style: {request.style}.' if request.style else ''
    user = f'Summarize the following text in at most {request.max_length} words.{style}\n\n{request.text}'
    return Prompt(SUMMARY_SYSTEM, user)


def build_translation_prompt(request: TranslationRequest) -> Prompt:
    """Build the prompt that asks a model to translate a text.

    Args:
        request:
            The translation request.

    Returns:
        The prompt.
    """
    source = f' from {request.source_language}' if request.source_language else ''
    user = f'Translate the following text{source} into {request.target_language}.\n\n{request.text}'
    return Prompt(TRANSLATION_SYSTEM, user)


def build_description_prompt(request: DescriptionRequest) -> Prompt:
    """Build the prompt that asks a model to describe an image.

    Args:
        request:
            The description request.

    Returns:
        The prompt.
    """
    instructions = f' {request.instructions}' if request.instructions else ''
    user = f'<image 1>\n\n{DETAIL_INSTRUCTIONS[request.detail]}{instructions}'
    return Prompt(DESCRIPTION_SYSTEM, user, (request.image,))


def build_image_prompt(request: ImageGenerationRequest) -> str:
    """Build the text prompt of an image generation model from a request.

    Args:
        request:
            The image generation request.

    Returns:
        The prompt, with the style appended.
    """
    return f'{request.prompt}. Style: {request.style}' if request.style else request.prompt
