"""Build prompts for prompted backends and parse their answers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._errors import InvalidResponseError
from ._media import Audio
from ._requests import (
    ClassificationRequest,
    DescriptionRequest,
    ExtractionRequest,
    ImageGenerationRequest,
    Sample,
    SummaryRequest,
    TranslationRequest,
    VideoFrames,
)

NO_LABEL = 'none'
SINGLE_LABEL_SYSTEM = 'You are a classifier. Answer with exactly one label from the list of labels and nothing else.'
MULTI_LABEL_SYSTEM = (
    'You are a multi-label classifier. Answer with every label from the list of labels that applies, separated by '
    f'commas, and nothing else. Answer `{NO_LABEL}` if no label applies.'
)
SUMMARY_SYSTEM = 'You summarize texts. Answer with the summary only.'
TRANSLATION_SYSTEM = 'You translate texts. Answer with the translation only.'
DESCRIPTION_SYSTEM = 'You describe images, audio and videos. Answer with the description only.'
EXTRACTION_SYSTEM = (
    'You extract information from texts. Answer with one JSON object whose keys are exactly the requested fields. '
    'Use null for a field that the text does not contain.'
)
DETAIL_INSTRUCTIONS = {
    'brief': 'Describe the {kind} in one sentence.',
    'detailed': 'Describe the {kind} in one detailed paragraph.',
}
CODE_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$')


@dataclass(frozen=True)
class Prompt:
    """Hold the messages of a prompt for a chat or vision model.

    Args:
        system:
            The system message.

        user:
            The user message. Images are referred to as `<image 1>`, `<image 2>` and so on, and audio as `<audio 1>`.

        images:
            The images of the user message, in the order they are referred to.

        audio:
            The audio of the user message, in the order it is referred to.

        json_output:
            Whether the answer must be a JSON object.
    """

    system: str
    user: str
    images: tuple[Image.Image, ...] = ()
    audio: tuple[Audio, ...] = ()
    json_output: bool = False


def _normalize(text: str) -> str:
    """Lower-case a label or an answer and strip quotes, spaces and final periods."""
    return text.strip().strip('`"\'').rstrip('.').strip().lower()


def _format_input(sample: Sample, images: list[Image.Image], audio: list[Audio], include_audio: bool) -> str:
    """Format a sample for a prompt, collecting its images and audio."""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, Audio):
        audio.append(sample)
        return f'<audio {len(audio)}>'
    if isinstance(sample, VideoFrames):
        references = [_format_input(frame, images, audio, include_audio) for frame in sample.frames]
        text = f'video frames {" ".join(references)}'
        if not include_audio:
            return text
        return f'{text} with audio {_format_input(sample.audio, images, audio, include_audio)}'
    images.append(sample)
    return f'<image {len(images)}>'


def _format_pairs(pairs: tuple[tuple[str, str], ...], source: str, target: str) -> str:
    """Format example pairs of texts and their references for a prompt."""
    if not pairs:
        return ''
    examples = '\n\n'.join(f'{source}: {text}\n{target}: {reference}' for text, reference in pairs)
    return f' Follow the examples.\n\nExamples:\n\n{examples}'


def _media_kind(sample: Sample) -> str:
    """Name the kind of media of a sample."""
    if isinstance(sample, Audio):
        return 'audio'
    if isinstance(sample, VideoFrames):
        return 'video'
    return 'image'


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


def build_classification_prompt(request: ClassificationRequest, include_audio: bool = True) -> Prompt:
    """Build the prompt that asks a model to classify a sample.

    Args:
        request:
            The classification request.

        include_audio:
            Whether to include the audio track of sampled video frames.

    Returns:
        The prompt.
    """
    images: list[Image.Image] = []
    audio: list[Audio] = []
    lines = ['Labels:']
    lines += [
        f'- {label}: {request.descriptions[label]}' if label in request.descriptions else f'- {label}'
        for label in request.labels
    ]
    if request.instructions:
        lines += ['', request.instructions]
    for number, (sample, labels) in enumerate(request.examples, start=1):
        answer = ', '.join(labels) or NO_LABEL
        example = _format_input(sample, images, audio, include_audio)
        lines += ['', f'Example {number}:', f'Input: {example}', f'Output: {answer}']
    lines += ['', f'Input: {_format_input(request.sample, images, audio, include_audio)}', 'Output:']
    system = MULTI_LABEL_SYSTEM if request.multi_label else SINGLE_LABEL_SYSTEM
    return Prompt(system, '\n'.join(lines), tuple(images), tuple(audio))


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
    examples = _format_pairs(request.examples, 'Text', 'Summary')
    user = f'Summarize the following text in at most {request.max_length} words.{style}{examples}\n\n{request.text}'
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
    glossary = ''.join(f'\n- {term} → {translation}' for term, translation in request.glossary)
    glossary = f'\n\nTranslate these terms as given:{glossary}' if glossary else ''
    examples = _format_pairs(request.examples, 'Text', 'Translation')
    user = f'Translate the following text{source} into {request.target_language}.{glossary}{examples}\n\n{request.text}'
    return Prompt(TRANSLATION_SYSTEM, user)


def build_description_prompt(request: DescriptionRequest, include_audio: bool = True) -> Prompt:
    """Build the prompt that asks a model to describe an image, audio or sampled video frames.

    Args:
        request:
            The description request.

        include_audio:
            Whether to include the audio track of sampled video frames.

    Returns:
        The prompt.
    """
    images: list[Image.Image] = []
    audio: list[Audio] = []
    lines = []
    for number, (sample, description) in enumerate(request.examples, start=1):
        example = _format_input(sample, images, audio, include_audio)
        lines += [f'Example {number}: {example}', f'Description: {description}', '']
    reference = _format_input(request.sample, images, audio, include_audio)
    instructions = f' {request.instructions}' if request.instructions else ''
    detail = DETAIL_INSTRUCTIONS[request.detail].format(kind=_media_kind(request.sample))
    if request.examples:
        detail = f'{detail} Follow the style and length of the example descriptions.'
    user = '\n'.join([*lines, reference, '', f'{detail}{instructions}'])
    return Prompt(DESCRIPTION_SYSTEM, user, tuple(images), tuple(audio))


def build_extraction_prompt(request: ExtractionRequest) -> Prompt:
    """Build the prompt that asks a model to extract fields from a text as JSON.

    Args:
        request:
            The extraction request.

    Returns:
        The prompt, which asks for a JSON object.
    """
    fields = '\n'.join(f'- {name}: {description}' for name, description in request.fields)
    user = f'Fields:\n{fields}\n\nText:\n{request.text}'
    return Prompt(EXTRACTION_SYSTEM, user, json_output=True)


def parse_fields(answer: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Read the JSON answer of a model into the value of every field.

    Args:
        answer:
            The answer of the model.

        names:
            The field names.

    Returns:
        The value of every field, or `None` for a field that the answer leaves out. Unknown keys are dropped.

    Raises:
        InvalidResponseError:
            If the answer is not a JSON object.
    """
    try:
        data = json.loads(CODE_FENCE.sub('', answer.strip()))
    except json.JSONDecodeError as error:
        msg = f'The answer `{answer}` is not valid JSON ({error.msg}).'
        raise InvalidResponseError(msg) from error
    if not isinstance(data, dict):
        msg = f'The answer `{answer}` is not a JSON object.'
        raise InvalidResponseError(msg)
    return {name: data.get(name) for name in names}


def build_image_prompt(request: ImageGenerationRequest) -> str:
    """Build the text prompt of an image generation model from a request.

    Args:
        request:
            The image generation request.

    Returns:
        The prompt, with the style appended.
    """
    return f'{request.prompt}. Style: {request.style}' if request.style else request.prompt


@dataclass(frozen=True, eq=False)
class FineTuneRequest:
    """Ask a backend to train a model on prompts and their expected answers.

    Args:
        task:
            The task the model is trained for.

        examples:
            Pairs of a prompt and the answer the model must give.
    """

    task: str
    examples: tuple[tuple[Prompt, str], ...]


def build_task_prompt(
    task: str,
    request: ClassificationRequest | SummaryRequest | TranslationRequest | DescriptionRequest | ExtractionRequest,
) -> Prompt:
    """Build the prompt that a prompted backend sends for a task and a request.

    Args:
        task:
            The name of the task.

        request:
            The request of the task.

    Returns:
        The prompt.

    Raises:
        ValueError:
            If the task is not answered by prompting.
    """
    builders: dict[str, Callable[[Any], Prompt]] = {
        'classify_text': build_classification_prompt,
        'classify_image': build_classification_prompt,
        'classify_audio': build_classification_prompt,
        'classify_video': build_classification_prompt,
        'summarize': build_summary_prompt,
        'translate': build_translation_prompt,
        'describe_image': build_description_prompt,
        'describe_audio': build_description_prompt,
        'describe_video': build_description_prompt,
        'extract_fields': build_extraction_prompt,
    }
    if task not in builders:
        msg = f'The task `{task}` is not answered by prompting.'
        raise ValueError(msg)
    return builders[task](request)
