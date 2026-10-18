"""Return components configured for common tasks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, Literal

from skai.classification import (
    AudioClassifier,
    ImageClassifier,
    MultimodalClassifier,
    TextClassifier,
    VideoClassifier,
)
from skai.core import BaseClassifier, BaseSupervisedGenerator
from skai.extraction import TextExtractor
from skai.generation import (
    AudioToAudioGenerator,
    AudioToTextGenerator,
    AudioToVideoGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    MultimodalToTextGenerator,
    MultimodalToVideoGenerator,
    TextToAudioGenerator,
    TextToTextGenerator,
    VideoToTextGenerator,
)
from skai.segmentation import AudioSegmenter, ImageSegmenter, VideoSegmenter

Kind = Literal['text', 'image', 'audio', 'video', 'multimodal']

INSTRUCTIONS: dict[str, str] = {
    'summarizer': 'Summarize the text, keeping its main points.',
    'translator': 'Translate the text into {target_language}, keeping its meaning, tone and formatting.',
    'rewriter': 'Rewrite the text in a {style} style, keeping its meaning.',
    'describer': 'Describe the {kind} in a few sentences.',
    'question_answerer': 'Answer this question about the {kind}: {question}',
    'script_writer': 'Write a {length} {format} script from the text, as the words to be spoken only.',
    'prompt_writer': 'Write one detailed prompt for an {target} generation model that shows the text.',
    'quiz_writer': 'Write {n_questions} multiple-choice questions, with their answers, that test the text.',
    'storyboarder': 'Split the text into {n_scenes} scenes, each with the narration and what the viewer sees.',
    'moderator': 'Decide whether the {kind} is appropriate for a general audience.',
}

_TEXT_GENERATORS: dict[str, type[BaseSupervisedGenerator]] = {
    'text': TextToTextGenerator,
    'image': ImageToTextGenerator,
    'audio': AudioToTextGenerator,
    'video': VideoToTextGenerator,
    'multimodal': MultimodalToTextGenerator,
}
_CLASSIFIERS: dict[str, type[BaseClassifier]] = {
    'text': TextClassifier,
    'image': ImageClassifier,
    'audio': AudioClassifier,
    'video': VideoClassifier,
    'multimodal': MultimodalClassifier,
}


def _instructed(kind: Kind, instructions: str, **kwargs: Any) -> BaseSupervisedGenerator:
    """Return the instructed text generator of an input kind."""
    generator = _TEXT_GENERATORS[kind]
    if kind in {'audio', 'video'}:
        kwargs = {'operation': 'instruct', **kwargs}
    return generator(instructions=instructions, **kwargs)


def summarizer(max_length: int | Literal['auto'] | None = 'auto', **kwargs: Any) -> TextToTextGenerator:
    """Return a text generator that summarizes texts.

    Args:
        max_length:
            The maximum number of words of every summary, or `'auto'` to learn it from reference summaries.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    return TextToTextGenerator(instructions=INSTRUCTIONS['summarizer'], max_length=max_length, **kwargs)


def translator(target_language: str, *, learn_terms: bool = True, **kwargs: Any) -> TextToTextGenerator:
    """Return a text generator that translates texts and learns a glossary from reference translations.

    Args:
        target_language:
            The language of the translations.

        learn_terms:
            Whether `fit` learns a glossary of consistently rendered terms.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    instructions = INSTRUCTIONS['translator'].format(target_language=target_language)
    return TextToTextGenerator(instructions=instructions, learn_terms=learn_terms, **kwargs)


def rewriter(style: str = 'clear and concise', **kwargs: Any) -> TextToTextGenerator:
    """Return a text generator that rewrites texts in a style.

    Args:
        style:
            The style of the rewritten texts.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    return TextToTextGenerator(instructions=INSTRUCTIONS['rewriter'].format(style=style), **kwargs)


def describer(kind: Kind = 'image', **kwargs: Any) -> BaseSupervisedGenerator:
    """Return a text generator that describes images, audio, videos or multimodal samples.

    Args:
        kind:
            The input kind.

        **kwargs:
            Other parameters of the generator.

    Returns:
        The configured `{Kind}ToTextGenerator`.
    """
    return _instructed(kind, INSTRUCTIONS['describer'].format(kind=kind), **kwargs)


def question_answerer(question: str, kind: Kind = 'text', **kwargs: Any) -> BaseSupervisedGenerator:
    """Return a text generator that answers one question about every sample.

    Args:
        question:
            The question.

        kind:
            The input kind.

        **kwargs:
            Other parameters of the generator.

    Returns:
        The configured `{Kind}ToTextGenerator`.
    """
    return _instructed(kind, INSTRUCTIONS['question_answerer'].format(kind=kind, question=question), **kwargs)


def script_writer(
    format: str = 'video narration',  # noqa: A002 - the name of the kind of script
    length: str = 'one-minute',
    **kwargs: Any,
) -> TextToTextGenerator:
    """Return a text generator that writes a spoken script from every text.

    Args:
        format:
            The kind of script, such as `'video narration'` or `'podcast'`.

        length:
            The length of the script, in words a model understands, such as `'one-minute'`.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    instructions = INSTRUCTIONS['script_writer'].format(format=format, length=length)
    return TextToTextGenerator(instructions=instructions, **kwargs)


def prompt_writer(target: Literal['image', 'video', 'music'] = 'image', **kwargs: Any) -> TextToTextGenerator:
    """Return a text generator that writes a generation prompt from every text.

    Args:
        target:
            The kind of generation model the prompt is for.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    return TextToTextGenerator(instructions=INSTRUCTIONS['prompt_writer'].format(target=target), **kwargs)


def quiz_writer(n_questions: int = 5, **kwargs: Any) -> TextToTextGenerator:
    """Return a text generator that writes a multiple-choice quiz from every text.

    Args:
        n_questions:
            The number of questions.

        **kwargs:
            Other parameters of `TextToTextGenerator`.

    Returns:
        The configured generator.
    """
    return TextToTextGenerator(instructions=INSTRUCTIONS['quiz_writer'].format(n_questions=n_questions), **kwargs)


def storyboarder(n_scenes: int | str = 'a few', **kwargs: Any) -> TextExtractor:
    """Return an extractor that splits every text into scenes with narration and visuals.

    Args:
        n_scenes:
            The number of scenes, or words such as `'a few'`.

        **kwargs:
            Other parameters of `TextExtractor`.

    Returns:
        The configured extractor, whose only field `scenes` is a list of records with `narration` and `visual`.
    """
    fields = {'scenes': [{'narration': 'the words spoken in the scene', 'visual': 'what the viewer sees'}]}
    instructions = INSTRUCTIONS['storyboarder'].format(n_scenes=n_scenes)
    return TextExtractor(fields=fields, instructions=instructions, **kwargs)


def moderator(labels: list[str] | None = None, kind: Kind = 'multimodal', **kwargs: Any) -> BaseClassifier:
    """Return a classifier that labels samples as safe or unsafe.

    Args:
        labels:
            The labels, `['safe', 'unsafe']` when it is `None`.

        kind:
            The input kind.

        **kwargs:
            Other parameters of the classifier.

    Returns:
        The configured `{Kind}Classifier`.
    """
    instructions = INSTRUCTIONS['moderator'].format(kind='sample' if kind == 'multimodal' else kind)
    return _CLASSIFIERS[kind](labels=labels or ['safe', 'unsafe'], instructions=instructions, **kwargs)


def ocr(**kwargs: Any) -> ImageToTextGenerator:
    """Return a text generator that transcribes the text of images as plain text.

    Args:
        **kwargs:
            Other parameters of `ImageToTextGenerator`.

    Returns:
        The configured generator.
    """
    return ImageToTextGenerator(operation='transcribe', **kwargs)


def document_parser(**kwargs: Any) -> ImageToTextGenerator:
    """Return a text generator that transcribes page images as Markdown that keeps their structure.

    Args:
        **kwargs:
            Other parameters of `ImageToTextGenerator`.

    Returns:
        The configured generator.
    """
    return ImageToTextGenerator(operation='transcribe', format='markdown', **kwargs)


def diarizer(**kwargs: Any) -> AudioSegmenter:
    """Return a segmenter that splits audio into speaker turns.

    Args:
        **kwargs:
            Other parameters of `AudioSegmenter`.

    Returns:
        The configured segmenter.
    """
    return AudioSegmenter(by='speaker', **kwargs)


def object_detector(labels: list[str], **kwargs: Any) -> ImageSegmenter:
    """Return a segmenter that finds the boxes of objects in images.

    Args:
        labels:
            The kinds of objects to find.

        **kwargs:
            Other parameters of `ImageSegmenter`.

    Returns:
        The configured segmenter.
    """
    return ImageSegmenter(by='object', labels=labels, **kwargs)


def shot_detector(**kwargs: Any) -> VideoSegmenter:
    """Return a segmenter that splits videos into shots locally.

    Args:
        **kwargs:
            Other parameters of `VideoSegmenter`.

    Returns:
        The configured segmenter.
    """
    return VideoSegmenter(by='shot', **kwargs)


def background_remover(**kwargs: Any) -> ImageToImageGenerator:
    """Return a generator that removes the background of images.

    Args:
        **kwargs:
            Other parameters of `ImageToImageGenerator`.

    Returns:
        The configured generator.
    """
    return ImageToImageGenerator(operation='remove_background', **kwargs)


def upscaler(scale: int = 2, **kwargs: Any) -> ImageToImageGenerator:
    """Return a generator that upscales images.

    Args:
        scale:
            The factor of upscaling.

        **kwargs:
            Other parameters of `ImageToImageGenerator`.

    Returns:
        The configured generator.
    """
    return ImageToImageGenerator(operation='upscale', scale=scale, **kwargs)


def music_generator(duration: float = 10.0, **kwargs: Any) -> TextToAudioGenerator:
    """Return a generator that creates music from descriptions.

    Args:
        duration:
            The length of the music in seconds.

        **kwargs:
            Other parameters of `TextToAudioGenerator`.

    Returns:
        The configured generator.
    """
    return TextToAudioGenerator(operation='music', duration=duration, **kwargs)


def voice_converter(voice: Any, **kwargs: Any) -> AudioToAudioGenerator:
    """Return a generator that says the speech of audio in another voice.

    Args:
        voice:
            One recording or a list of recordings of the target voice.

        **kwargs:
            Other parameters of `AudioToAudioGenerator`.

    Returns:
        The configured generator.
    """
    return AudioToAudioGenerator(operation='convert_voice', voice=voice, **kwargs)


def separator(stem: Literal['vocals', 'accompaniment'] = 'vocals', **kwargs: Any) -> AudioToAudioGenerator:
    """Return a generator that separates the vocals or the accompaniment of audio.

    Args:
        stem:
            The stem to keep.

        **kwargs:
            Other parameters of `AudioToAudioGenerator`.

    Returns:
        The configured generator.
    """
    return AudioToAudioGenerator(operation='separate', stem=stem, **kwargs)


def lip_syncer(**kwargs: Any) -> MultimodalToVideoGenerator:
    """Return a generator that makes the `video` of every sample speak its `audio`.

    Args:
        **kwargs:
            Other parameters of `MultimodalToVideoGenerator`.

    Returns:
        The configured generator.
    """
    return MultimodalToVideoGenerator(operation='lip_sync', **kwargs)


def talking_head(avatar: Any, **kwargs: Any) -> AudioToVideoGenerator:
    """Return a generator that makes a portrait speak audio.

    Args:
        avatar:
            The portrait.

        **kwargs:
            Other parameters of `AudioToVideoGenerator`.

    Returns:
        The configured generator.
    """
    return AudioToVideoGenerator(avatar=avatar, **kwargs)


def animator(instructions: str | None = None, duration: float = 2.0, **kwargs: Any) -> ImageToVideoGenerator:
    """Return a generator that turns images into short videos.

    Args:
        instructions:
            A description of the motion.

        duration:
            The length of the videos in seconds.

        **kwargs:
            Other parameters of `ImageToVideoGenerator`.

    Returns:
        The configured generator.
    """
    return ImageToVideoGenerator(instructions=instructions, duration=duration, **kwargs)
