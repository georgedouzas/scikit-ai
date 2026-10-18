"""Build prompts for prompted backends and parse their answers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import base64
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._errors import InvalidResponseError
from ._io import to_data_url
from ._media import Audio
from ._requests import (
    ClassificationRequest,
    ExtractionRequest,
    ImageGenerationRequest,
    ImageTranscriptionRequest,
    Moment,
    MomentRequest,
    Region,
    RegionRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    VideoFrames,
)

NO_LABEL = 'none'
SINGLE_LABEL_SYSTEM = 'You are a classifier. Answer with exactly one label from the list of labels and nothing else.'
MULTI_LABEL_SYSTEM = (
    'You are a multi-label classifier. Answer with every label from the list of labels that applies, separated by '
    f'commas, and nothing else. Answer `{NO_LABEL}` if no label applies.'
)
GENERATION_SYSTEM = 'You follow instructions about the input you are given. Answer with the result only.'
EXTRACTION_SYSTEM = (
    'You extract information from the input. Answer with one JSON object that follows the template exactly, with '
    'the same keys. Use null for a value that the input does not contain, and an empty list for a list with no items.'
)
TRANSCRIPTION_SYSTEM = 'You transcribe the text written in images exactly, without correcting or summarizing it.'
DETECTION_SYSTEM = 'You locate regions in images. Answer with one JSON object and nothing else.'
MOMENT_SYSTEM = 'You find moments in audio and videos. Answer with one JSON object and nothing else.'
THUMBNAIL_SYSTEM = 'You pick the frame that best represents a video. Answer with the frame number only.'
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


def _format_frames(frames: VideoFrames, images: list[Image.Image], audio: list[Audio], include_audio: bool) -> str:
    """Format sampled video frames for a prompt, with their audio track when it is included."""
    references = [_format_input(frame, images, audio, include_audio) for frame in frames.frames]
    text = f'video frames {" ".join(references)}'
    if not include_audio or frames.audio is None:
        return text
    return f'{text} with audio {_format_input(frames.audio, images, audio, include_audio)}'


def _format_input(sample: Any, images: list[Image.Image], audio: list[Audio], include_audio: bool) -> str:
    """Format a sample for a prompt, collecting its images and audio, with the keys of a multimodal sample."""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        parts = [f'{key}: {_format_input(value, images, audio, include_audio)}' for key, value in sample.items()]
        return '\n'.join(parts)
    if isinstance(sample, list | tuple):
        return json.dumps(sample, ensure_ascii=False, default=str)
    if isinstance(sample, Audio):
        audio.append(sample)
        return f'<audio {len(audio)}>'
    if isinstance(sample, VideoFrames):
        return _format_frames(sample, images, audio, include_audio)
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


def _template(declaration: Any) -> Any:
    """Turn a field declaration into the JSON template a model fills in."""
    if isinstance(declaration, dict):
        return {name: _template(value) for name, value in declaration.items()}
    if isinstance(declaration, list):
        return [_template(declaration[0])] if declaration else []
    return f'<{declaration}>'


def build_extraction_prompt(request: ExtractionRequest, include_audio: bool = True) -> Prompt:
    """Build the prompt that asks a model to extract fields from a sample as JSON.

    Args:
        request:
            The extraction request.

        include_audio:
            Whether to include the audio track of sampled video frames.

    Returns:
        The prompt, which asks for a JSON object.
    """
    images: list[Image.Image] = []
    audio: list[Audio] = []
    template = json.dumps({name: _template(declaration) for name, declaration in request.fields}, ensure_ascii=False)
    examples = ''.join(
        f'\n\nExample input:\n{_format_input(sample, images, audio, include_audio)}\n'
        f'Example answer:\n{json.dumps(record, ensure_ascii=False, default=str)}'
        for sample, record in request.examples
    )
    guidance = f'Instructions: {request.instructions}\n\n' if request.instructions else ''
    sample = _format_input(request.sample, images, audio, include_audio)
    user = f'{guidance}Template:\n{template}{examples}\n\nInput:\n{sample}'
    return Prompt(EXTRACTION_SYSTEM, user, tuple(images), tuple(audio), json_output=True)


def build_text_generation_prompt(request: TextGenerationRequest, include_audio: bool = True) -> Prompt:
    """Build the prompt that asks a model to produce text from a sample by following instructions.

    Args:
        request:
            The text generation request.

        include_audio:
            Whether to include the audio track of sampled video frames.

    Returns:
        The prompt.
    """
    images: list[Image.Image] = []
    audio: list[Audio] = []
    lines = [f'Instructions: {request.instructions}']
    if request.max_length is not None:
        lines.append(f'Use at most {request.max_length} words.')
    if request.glossary:
        lines.append('Render these terms as given:')
        lines += [f'- {term} → {rendering}' for term, rendering in request.glossary]
    for number, (sample, output) in enumerate(request.examples, start=1):
        lines += ['', f'Example {number}:', f'Input: {_format_input(sample, images, audio, include_audio)}']
        lines.append(f'Output: {output}')
    lines += ['', f'Input: {_format_input(request.sample, images, audio, include_audio)}', 'Output:']
    return Prompt(GENERATION_SYSTEM, '\n'.join(lines), tuple(images), tuple(audio))


def build_image_transcription_prompt(request: ImageTranscriptionRequest) -> Prompt:
    """Build the prompt that asks a vision model to transcribe the text of an image.

    Args:
        request:
            The image transcription request.

    Returns:
        The prompt.
    """
    structure = ', keeping the structure of the page as Markdown headings, lists and tables' if request.markdown else ''
    user = f'<image 1>\n\nTranscribe all the text exactly as written{structure}. Answer with the transcription only.'
    return Prompt(TRANSCRIPTION_SYSTEM, user, (request.image,))


def build_detection_prompt(request: RegionRequest) -> Prompt:
    """Build the prompt that asks a vision model for the boxes of the regions to find.

    Args:
        request:
            The region request.

    Returns:
        The prompt, which asks for a JSON object.
    """
    width, height = request.image.size
    target = ', '.join(request.labels) if request.labels else request.prompt
    user = (
        f'<image 1>\n\nThe image is {width}x{height} pixels. Find every region showing: {target}. Answer with '
        '{"regions": [{"label": <label>, "box": [left, top, right, bottom], "score": <0 to 1>}]} in pixels.'
    )
    return Prompt(DETECTION_SYSTEM, user, (request.image,), json_output=True)


def build_moment_prompt(request: MomentRequest, include_audio: bool = True) -> Prompt:
    """Build the prompt that asks a model for the moments of audio or a video that match a description.

    Args:
        request:
            The moment request.

        include_audio:
            Whether to include the audio track of sampled video frames.

    Returns:
        The prompt, which asks for a JSON object.
    """
    images: list[Image.Image] = []
    audio: list[Audio] = []
    reference = _format_input(request.sample, images, audio, include_audio)
    if isinstance(request.sample, VideoFrames):
        times = ', '.join(f'frame {index + 1} at {time:.1f}s' for index, time in enumerate(request.sample.timestamps))
        reference = f'{reference}\n\nThe video lasts {request.sample.duration:.1f}s, with {times}.'
    else:
        reference = f'{reference}\n\nThe audio lasts {request.sample.duration:.1f}s.'
    user = (
        f'{reference}\n\nFind every moment that matches: {request.prompt}. Answer with '
        '{"moments": [{"start": <seconds>, "end": <seconds>, "score": <0 to 1>}]}.'
    )
    return Prompt(MOMENT_SYSTEM, user, tuple(images), tuple(audio), json_output=True)


def build_thumbnail_prompt(request: ThumbnailRequest) -> Prompt:
    """Build the prompt that asks a vision model for the frame that best represents a video.

    Args:
        request:
            The thumbnail request.

    Returns:
        The prompt.
    """
    images = request.frames.frames
    references = ' '.join(f'<image {number}>' for number in range(1, len(images) + 1))
    user = f'{references}\n\nWhich frame, from 1 to {len(images)}, best represents the video? Answer with its number.'
    return Prompt(THUMBNAIL_SYSTEM, user, tuple(images))


def _load_json(answer: str) -> Any:
    """Read the JSON object of an answer, without a Markdown code fence."""
    try:
        return json.loads(CODE_FENCE.sub('', answer.strip()))
    except json.JSONDecodeError as error:
        msg = f'The answer `{answer}` is not valid JSON ({error.msg}).'
        raise InvalidResponseError(msg) from error


def parse_regions(answer: str, size: tuple[int, int]) -> list[Region]:
    """Read the regions of a detection answer, clamping their boxes to the image.

    Args:
        answer:
            The answer of the model.

        size:
            The width and height of the image.

    Returns:
        The regions.

    Raises:
        InvalidResponseError:
            If the answer is not a JSON object with a list of regions.
    """
    data = _load_json(answer)
    if not isinstance(data, dict) or not isinstance(data.get('regions'), list):
        msg = f'The answer `{answer}` has no list of regions.'
        raise InvalidResponseError(msg)
    width, height = size
    regions = []
    for item in data['regions']:
        left, top, right, bottom = (round(float(value)) for value in item['box'])
        box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        regions.append(Region(str(item.get('label', '')), float(item.get('score', 1.0)), box))
    return regions


def parse_moments(answer: str, duration: float) -> list[Moment]:
    """Read the moments of an answer, clamping them to the length of the media.

    Args:
        answer:
            The answer of the model.

        duration:
            The length of the media in seconds.

    Returns:
        The moments, in time order.

    Raises:
        InvalidResponseError:
            If the answer is not a JSON object with a list of moments.
    """
    data = _load_json(answer)
    if not isinstance(data, dict) or not isinstance(data.get('moments'), list):
        msg = f'The answer `{answer}` has no list of moments.'
        raise InvalidResponseError(msg)
    moments = [
        Moment(max(0.0, float(item['start'])), min(duration, float(item['end'])), float(item.get('score', 1.0)))
        for item in data['moments']
    ]
    return sorted((moment for moment in moments if moment.end > moment.start), key=lambda moment: moment.start)


def parse_frame_number(answer: str, count: int) -> int:
    """Read the frame number of an answer as a zero-based index.

    Args:
        answer:
            The answer of the model.

        count:
            The number of frames.

    Returns:
        The index of the frame.

    Raises:
        InvalidResponseError:
            If the answer names no frame between 1 and `count`.
    """
    match = re.search(r'\d+', answer)
    if match is None or not 1 <= int(match.group()) <= count:
        msg = f'The answer `{answer}` names no frame between 1 and {count}.'
        raise InvalidResponseError(msg)
    return int(match.group()) - 1


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
    data = _load_json(answer)
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


def to_chat_messages(prompt: Prompt) -> list[dict[str, Any]]:
    """Turn a prompt into chat messages in the OpenAI format, with images as data URLs and audio as WAV.

    Args:
        prompt:
            The prompt.

    Returns:
        The system and user messages.
    """
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt.user}]
    content += [{'type': 'image_url', 'image_url': {'url': to_data_url(image)}} for image in prompt.images]
    content += [
        {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(audio.to_bytes()).decode(), 'format': 'wav'}}
        for audio in prompt.audio
    ]
    return [{'role': 'system', 'content': prompt.system}, {'role': 'user', 'content': content}]


def build_task_prompt(task: str, request: ClassificationRequest | TextGenerationRequest | ExtractionRequest) -> Prompt:
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
        'classify_multimodal': build_classification_prompt,
        'generate_text': build_text_generation_prompt,
        'extract_fields': build_extraction_prompt,
    }
    if task not in builders:
        msg = f'The task `{task}` is not answered by prompting.'
        raise ValueError(msg)
    return builders[task](request)
