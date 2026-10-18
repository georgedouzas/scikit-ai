"""Test where components are imported from and how they are named."""

import importlib
import inspect
import re

import pytest

import skai.classification
import skai.embedding
import skai.extraction
import skai.generation
import skai.pipeline
import skai.segmentation

INPUT = '(Text|Image|Audio|Video|Multimodal)'
OUTPUT = '(Text|Image|Audio|Video)'
ROLES = {
    skai.classification: 'Classifier',
    skai.extraction: 'Extractor',
    skai.embedding: 'Embedder',
    skai.segmentation: 'Segmenter',
}
MODULES = [*ROLES, skai.generation, skai.pipeline]
REMOVED = [
    'TextSummarizer',
    'TextTranslator',
    'ImageDescriptor',
    'AudioDescriptor',
    'VideoDescriptor',
    'SpeechTranscriber',
    'VideoTranscriber',
    'SpeechSynthesizer',
    'ImageGenerator',
    'ImageEditor',
    'ImageAnimator',
    'VideoGenerator',
    'TalkingAvatarGenerator',
]


def test_import_by_task_family():
    """Import one component of every role module."""
    from skai.classification import TextClassifier  # noqa: PLC0415
    from skai.embedding import AudioEmbedder  # noqa: PLC0415
    from skai.extraction import VideoExtractor  # noqa: PLC0415
    from skai.generation import VideoToTextGenerator  # noqa: PLC0415
    from skai.pipeline import ForEach  # noqa: PLC0415
    from skai.segmentation import AudioSegmenter  # noqa: PLC0415

    assert all(inspect.isclass(cls) for cls in (TextClassifier, VideoExtractor, AudioEmbedder, AudioSegmenter))
    assert all(inspect.isclass(cls) for cls in (VideoToTextGenerator, ForEach))


def test_every_public_name_is_exported():
    """Export only classes that the module defines under their names."""
    for module in MODULES:
        assert module.__all__ == sorted(module.__all__)
        assert all(inspect.isclass(getattr(module, name)) for name in module.__all__)


def test_pipeline_primitives():
    """Export the primitives from `skai.pipeline`."""
    assert {'MediaUnion', 'Field', 'ForEach', 'TopK'} <= set(skai.pipeline.__all__)
    assert set(skai.pipeline.__all__) <= {'MediaUnion', 'Field', 'ForEach', 'TopK', 'Retriever'}


def test_single_location_per_component():
    """Export every component name from one module only."""
    names = [name for module in MODULES for name in module.__all__]
    assert len(names) == len(set(names))


def test_input_type_in_the_name():
    """Name analyzers `{Input}{Role}` and generators `{Input}To{Output}Generator`."""
    for module, role in ROLES.items():
        assert module.__all__
        assert all(re.fullmatch(f'{INPUT}{role}', name) for name in module.__all__), module.__all__
    assert all(re.fullmatch(f'{INPUT}To{OUTPUT}Generator', name) for name in skai.generation.__all__)


def test_generator_grid():
    """Provide a text generator for every input and the media generators of the spec."""
    names = set(skai.generation.__all__)
    assert {f'{source}ToTextGenerator' for source in ('Text', 'Image', 'Audio', 'Video', 'Multimodal')} <= names
    media = {
        'TextToImageGenerator',
        'ImageToImageGenerator',
        'VideoToImageGenerator',
        'MultimodalToImageGenerator',
        'TextToAudioGenerator',
        'AudioToAudioGenerator',
        'VideoToAudioGenerator',
        'MultimodalToAudioGenerator',
        'TextToVideoGenerator',
        'ImageToVideoGenerator',
        'AudioToVideoGenerator',
        'VideoToVideoGenerator',
        'MultimodalToVideoGenerator',
    }
    assert media <= names


@pytest.mark.parametrize('module', ['skai.image', 'skai.audio', 'skai.text', 'skai.description', 'skai.transcription'])
def test_input_type_modules_removed(module):
    """Fail to import the removed modules."""
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_old_import_path():
    """Fail to import the 0.2 names from the old and new modules."""
    with pytest.raises(ImportError):
        exec('from skai.text import TextSummarizer', {})  # noqa: S102 # nosec B102 - checks a fixed import statement
    with pytest.raises(ImportError):
        exec('from skai.generation import ImageGenerator', {})  # noqa: S102 # nosec B102 - a fixed import
    for name in REMOVED:
        assert all(not hasattr(module, name) for module in [*MODULES, skai])
