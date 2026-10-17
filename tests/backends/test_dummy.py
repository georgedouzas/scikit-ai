"""Test the offline dummy backend."""

import numpy as np
import pytest
from PIL import Image

from skai.backends import DummyBackend
from skai.classification import ImageClassifier, TextClassifier
from skai.core import (
    TASKS,
    AnimationRequest,
    ClassificationRequest,
    DescriptionRequest,
    EmbeddingRequest,
    ExtractionRequest,
    ImageEditRequest,
    Video,
    VideoGenerationRequest,
    sample_frames,
)
from skai.description import ImageDescriptor
from skai.embedding import TextEmbedder
from skai.generation import ImageGenerator, SpeechSynthesizer, TalkingAvatarGenerator
from skai.text import TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber

COMPONENTS = [
    (TextClassifier(labels=['a', 'b'], backend='dummy'), ['x', 'y']),
    (TextSummarizer(backend='dummy'), ['x y z']),
    (TextTranslator(target_language='German', backend='dummy'), ['hello']),
    (TextEmbedder(backend='dummy'), ['hello']),
    (SpeechSynthesizer(backend='dummy', random_state=0), ['hello']),
    (ImageGenerator(size=(8, 8), backend='dummy', random_state=0), ['a cat']),
    (ImageDescriptor(backend='dummy'), [Image.new('RGB', (8, 8))]),
    (ImageClassifier(labels=['cat', 'dog'], backend='dummy'), [Image.new('RGB', (8, 8))]),
    (SpeechTranscriber(backend='dummy'), [(np.zeros(1600), 16000)]),
    (TalkingAvatarGenerator(avatar=Image.new('RGB', (8, 8)), backend='dummy'), [(np.zeros(1600), 16000)]),
]


def outputs_of(component, X):
    """Return the outputs of a fitted component as comparable values."""
    output = component.transform(X) if isinstance(component, TextEmbedder) else component.predict(X)
    return [np.asarray(item.frames if hasattr(item, 'frames') else getattr(item, 'samples', item)) for item in output]


def test_supported_tasks_cover_every_task():
    """Support every task."""
    assert DummyBackend.supported_tasks() == frozenset(TASKS)


@pytest.mark.parametrize(('component', 'X'), COMPONENTS)
def test_outputs_are_deterministic(component, X):
    """Return identical outputs for identical requests without network access."""
    first = outputs_of(component.fit(X if isinstance(component, TextEmbedder) else None), X)
    second = outputs_of(component, X)
    assert len(first) == len(X)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object))


def test_avatar_duration_matches_audio():
    """Return a video as long as its audio."""
    generator = TalkingAvatarGenerator(avatar=Image.new('RGB', (128, 96)), backend='dummy').fit()
    video = generator.predict([(np.zeros(24000), 16000)])[0]
    assert video.duration == pytest.approx(1.5, abs=1 / 25)
    assert max(video.frames.shape[1:3]) <= 64


def test_new_tasks_return_typed_outputs(image, audio):
    """Return outputs of the documented types for the audio, video, extraction and generation tasks."""
    backend = DummyBackend()
    frames = sample_frames(Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=5), 4)
    proba = backend.classify_video(ClassificationRequest(frames, ('a', 'b')))
    assert sum(proba.values()) == pytest.approx(1)
    assert backend.classify_audio(ClassificationRequest(audio, ('a', 'b'))) == backend.classify_audio(
        ClassificationRequest(audio, ('a', 'b')),
    )
    assert backend.describe_audio(DescriptionRequest(audio)) == 'A brief description of 1.00 seconds of audio.'
    assert backend.describe_video(DescriptionRequest(frames)).startswith('A brief description of 4 frames')
    fields = backend.extract_fields(ExtractionRequest('name: Ada', (('name', 'person'), ('city', 'city'))))
    assert set(fields) == {'name', 'city'}
    assert fields['city'] is None
    assert backend.embed_image(EmbeddingRequest(image, 16)).shape == (16,)
    assert backend.embed_audio(EmbeddingRequest(audio)).shape == (8,)
    video = backend.generate_video(VideoGenerationRequest('a boat', duration=2, size=(32, 24), fps=8))
    assert (video.duration, video.frames.shape[1:3]) == (2.0, (24, 32))
    mask = Image.new('L', image.size)
    assert backend.edit_image(ImageEditRequest(image, 'add snow', mask=mask)).tobytes() == image.tobytes()
    assert backend.edit_image(ImageEditRequest(image, 'add snow')).size == image.size
    animation = backend.animate_image(AnimationRequest(image, duration=1.5, fps=4))
    assert animation.frames.shape[0] == 6
