"""Test the offline dummy backend."""

import numpy as np
import pytest
from PIL import Image

from skai.audio import SpeechTranscriber, TalkingAvatarGenerator
from skai.backends import DummyBackend
from skai.core import TASKS
from skai.image import ImageClassifier, ImageDescriptor
from skai.text import ImageGenerator, SpeechSynthesizer, TextClassifier, TextEmbedder, TextSummarizer, TextTranslator

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
