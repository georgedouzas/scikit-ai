"""Return components configured for common tasks.

Every preset is a function that returns an ordinary component with its instructions and operation set, and passes
extra keyword arguments, such as `backend`, to the component. `INSTRUCTIONS` holds the instructions of the presets
that are text generators, extractors or classifiers.

Examples:
    >>> from skai import presets
    >>> summarizer = presets.summarizer(max_length=50, backend='dummy')
    >>> type(summarizer).__name__, summarizer.max_length
    ('TextToTextGenerator', 50)
"""

from ._presets import (
    INSTRUCTIONS,
    animator,
    background_remover,
    describer,
    diarizer,
    document_parser,
    lip_syncer,
    moderator,
    music_generator,
    object_detector,
    ocr,
    prompt_writer,
    question_answerer,
    quiz_writer,
    rewriter,
    script_writer,
    separator,
    shot_detector,
    storyboarder,
    summarizer,
    talking_head,
    translator,
    upscaler,
    voice_converter,
)

__all__: list[str] = [
    'INSTRUCTIONS',
    'animator',
    'background_remover',
    'describer',
    'diarizer',
    'document_parser',
    'lip_syncer',
    'moderator',
    'music_generator',
    'object_detector',
    'ocr',
    'prompt_writer',
    'question_answerer',
    'quiz_writer',
    'rewriter',
    'script_writer',
    'separator',
    'shot_detector',
    'storyboarder',
    'summarizer',
    'talking_head',
    'translator',
    'upscaler',
    'voice_converter',
]
