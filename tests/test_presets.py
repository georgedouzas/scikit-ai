"""Test the component presets of `skai.presets`."""

import inspect

import numpy as np
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from skai import presets
from skai.classification import ImageClassifier, MultimodalClassifier, TextClassifier
from skai.core import Audio
from skai.extraction import TextExtractor
from skai.generation import (
    AudioToAudioGenerator,
    AudioToTextGenerator,
    AudioToVideoGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    MultimodalToVideoGenerator,
    TextToAudioGenerator,
    TextToTextGenerator,
    VideoToTextGenerator,
)
from skai.segmentation import AudioSegmenter, ImageSegmenter, VideoSegmenter

VOICE = Audio(np.zeros(16000, dtype=np.float32), 16000)
PORTRAIT = Image.new('RGB', (16, 16))
PRESETS = {
    'summarizer': ({}, TextToTextGenerator, {}),
    'translator': ({'target_language': 'Greek'}, TextToTextGenerator, {'learn_terms': True}),
    'rewriter': ({}, TextToTextGenerator, {}),
    'describer': ({}, ImageToTextGenerator, {}),
    'question_answerer': ({'question': 'Who is there?'}, TextToTextGenerator, {}),
    'script_writer': ({}, TextToTextGenerator, {}),
    'prompt_writer': ({}, TextToTextGenerator, {}),
    'quiz_writer': ({}, TextToTextGenerator, {}),
    'storyboarder': ({}, TextExtractor, {}),
    'moderator': ({}, MultimodalClassifier, {'labels': ['safe', 'unsafe']}),
    'ocr': ({}, ImageToTextGenerator, {'operation': 'transcribe', 'format': 'text'}),
    'document_parser': ({}, ImageToTextGenerator, {'operation': 'transcribe', 'format': 'markdown'}),
    'diarizer': ({}, AudioSegmenter, {'by': 'speaker'}),
    'object_detector': ({'labels': ['cat']}, ImageSegmenter, {'by': 'object', 'labels': ['cat']}),
    'shot_detector': ({}, VideoSegmenter, {'by': 'shot'}),
    'background_remover': ({}, ImageToImageGenerator, {'operation': 'remove_background'}),
    'upscaler': ({}, ImageToImageGenerator, {'operation': 'upscale', 'scale': 2}),
    'music_generator': ({}, TextToAudioGenerator, {'operation': 'music'}),
    'voice_converter': ({'voice': VOICE}, AudioToAudioGenerator, {'operation': 'convert_voice'}),
    'separator': ({}, AudioToAudioGenerator, {'operation': 'separate', 'stem': 'vocals'}),
    'lip_syncer': ({}, MultimodalToVideoGenerator, {'operation': 'lip_sync'}),
    'talking_head': ({'avatar': PORTRAIT}, AudioToVideoGenerator, {}),
    'animator': ({}, ImageToVideoGenerator, {}),
}


def test_available_presets():
    """List at least every documented preset as a function of `skai.presets`."""
    functions = {
        name
        for name, value in vars(presets).items()
        if inspect.isfunction(value) and value.__module__.startswith('skai.presets')
    }
    assert set(PRESETS) <= functions


@pytest.mark.parametrize(('name', 'case'), PRESETS.items())
def test_preset_returns_documented_class(name, case):
    """Return the documented component with its configuration, passing extra keyword arguments through."""
    arguments, component_class, expected = case
    component = getattr(presets, name)(**arguments, backend='dummy', random_state=3)
    assert type(component) is component_class
    assert component.backend == 'dummy'
    assert component.random_state == 3
    params = component.get_params()
    assert {key: params[key] for key in expected} == expected
    clone(component).fit()


def test_summarizer_preset():
    """Return a summarizing text generator with the given length and backend."""
    summarizer = presets.summarizer(max_length=50, backend='dummy')
    assert isinstance(summarizer, TextToTextGenerator)
    assert summarizer.instructions == presets.INSTRUCTIONS['summarizer']
    assert 'Summarize' in summarizer.instructions
    assert summarizer.max_length == 50
    assert summarizer.backend == 'dummy'


def test_instructions_are_filled_in():
    """Fill the instructions of instructed presets with their arguments."""
    assert 'Greek' in presets.translator('Greek').instructions
    assert 'formal' in presets.rewriter(style='formal').instructions
    assert 'Who is there?' in presets.question_answerer('Who is there?').instructions
    assert '7 multiple-choice' in presets.quiz_writer(n_questions=7).instructions
    assert '4 scenes' in presets.storyboarder(n_scenes=4).instructions
    assert 'podcast' in presets.script_writer(format='podcast').instructions
    assert 'video' in presets.prompt_writer(target='video').instructions
    assert not any(
        '{' in instructions for instructions in [presets.describer().instructions, presets.moderator().instructions]
    )


@pytest.mark.parametrize(
    ('kind', 'describer_class', 'moderator_class'),
    [
        ('text', TextToTextGenerator, TextClassifier),
        ('image', ImageToTextGenerator, ImageClassifier),
        ('audio', AudioToTextGenerator, None),
        ('video', VideoToTextGenerator, None),
    ],
)
def test_presets_follow_the_input_kind(kind, describer_class, moderator_class):
    """Return the generator and classifier of the requested input kind."""
    assert type(presets.describer(kind=kind)) is describer_class
    assert type(presets.question_answerer('Why?', kind=kind)) is describer_class
    if moderator_class is not None:
        assert type(presets.moderator(kind=kind)) is moderator_class
    if kind == 'audio':
        assert presets.describer(kind=kind).operation == 'instruct'


def test_storyboarder_fields():
    """Extract a list of scenes with narration and visuals."""
    storyboarder = presets.storyboarder(backend='dummy')
    assert set(storyboarder.fields) == {'scenes'}
    assert set(storyboarder.fields['scenes'][0]) == {'narration', 'visual'}
    (record,) = storyboarder.fit().predict(['A boat leaves the harbor at dawn.'])
    assert all(set(scene) == {'narration', 'visual'} for scene in record['scenes'])


def test_presets_are_ordinary_estimators():
    """Clone, chain and tune a preset like the component it returns."""
    summarizer = presets.summarizer(max_length=50, backend='dummy')
    copy = clone(summarizer)
    assert type(copy) is TextToTextGenerator
    assert copy.get_params() == summarizer.get_params()
    texts = [f'text number {index} about many things' for index in range(6)]
    pipeline = make_pipeline(
        presets.translator('Greek', backend='dummy'),
        presets.summarizer(max_length=2, backend='dummy'),
    )
    assert [len(output.split()) for output in pipeline.fit(None).predict(texts)] == [2] * 6

    def length(references, outputs):
        """Return the negative gap between output and reference lengths."""
        pairs = zip(references, outputs, strict=True)
        return -np.mean([abs(len(reference.split()) - len(output.split())) for reference, output in pairs])

    search = GridSearchCV(summarizer, {'max_length': [2, 4]}, cv=2, scoring=make_scorer(length))
    search.fit(texts, ['one two three four'] * 6)
    assert search.best_params_ == {'max_length': 4}
