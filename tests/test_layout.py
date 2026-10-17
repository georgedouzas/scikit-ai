"""Test where components are imported from and how they are named."""

import importlib

import pytest

import skai.classification
import skai.description
import skai.embedding
import skai.generation
import skai.text
import skai.transcription

LAYOUT = {
    skai.classification: ['ImageClassifier', 'TextClassifier'],
    skai.description: ['ImageDescriptor'],
    skai.transcription: ['SpeechTranscriber'],
    skai.text: ['TextSummarizer', 'TextTranslator'],
    skai.embedding: ['TextEmbedder'],
    skai.generation: ['ImageGenerator', 'SpeechSynthesizer', 'TalkingAvatarGenerator'],
}
INPUT_TYPES = ('Text', 'Image', 'Speech', 'Audio', 'Video')


def test_components_are_grouped_by_task_family():
    """Export exactly the components of each task family."""
    for module, names in LAYOUT.items():
        assert sorted(module.__all__) == names
        assert all(hasattr(module, name) for name in names)


def test_components_live_in_one_module():
    """Export every component name from a single module."""
    names = [name for module in LAYOUT for name in module.__all__]
    assert len(names) == len(set(names))


def test_component_names_start_with_input_type():
    """Start the name of every non-generator component with its input type."""
    for module, names in LAYOUT.items():
        if module is not skai.generation:
            assert all(name.startswith(INPUT_TYPES) for name in names)


@pytest.mark.parametrize(('module', 'name'), [('skai.image', 'ImageDescriptor'), ('skai.audio', 'SpeechTranscriber')])
def test_input_type_modules_are_removed(module, name):
    """Fail to import the removed input-type modules."""
    with pytest.raises(ImportError):
        getattr(importlib.import_module(module), name)


def test_text_module_exports_only_text_to_text_components():
    """Fail to import non text-to-text components from `skai.text`."""
    with pytest.raises(ImportError):
        exec('from skai.text import TextClassifier', {})  # noqa: S102 # nosec B102 - checks a fixed import statement
