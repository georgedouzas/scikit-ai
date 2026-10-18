"""Test what components learn in `fit` and how pipelines pass targets."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from skai.backends import DummyBackend
from skai.classification import TextClassifier
from skai.core import TextGenerationRequest, UnusedFitDataWarning
from skai.generation import AudioToTextGenerator, ImageToTextGenerator, TextToTextGenerator
from tests.fakes import ScriptedBackend

TEXTS = [f'text number {index} about topic {index % 3}' for index in range(12)]
SUMMARIES = [f'topic {index % 3}' for index in range(12)]
SUMMARIZE = 'Summarize the text.'


def summarizer(**params):
    """Return a text generator that summarizes."""
    return TextToTextGenerator(instructions=SUMMARIZE, **params)


class RecordingBackend(DummyBackend):
    """Keep every request the backend receives."""

    requests: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_requests():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []


def test_fitted_state_changes_requests():
    """Send different requests after a fit with data than after a fit without it."""
    summarizer(backend=RecordingBackend()).fit().predict(['new text'])
    summarizer(backend=RecordingBackend()).fit(TEXTS, SUMMARIES).predict(['new text'])
    zero_shot, few_shot = RecordingBackend.requests
    assert zero_shot.examples == ()
    assert few_shot.examples
    assert zero_shot != few_shot


def test_examples_are_capped_and_reproducible():
    """Pass at most `max_examples` pairs, the same ones for the same `random_state`."""
    first = summarizer(max_examples=3, random_state=0, backend='dummy').fit(TEXTS, SUMMARIES).examples_
    second = summarizer(max_examples=3, random_state=0, backend='dummy').fit(TEXTS, SUMMARIES).examples_
    assert len(first) == 3
    assert first == second
    assert summarizer(k_shot=0, backend='dummy').fit(TEXTS, SUMMARIES).examples_ == ()
    assert summarizer(k_shot=[1, 4], backend='dummy').fit(TEXTS, SUMMARIES).examples_[1] == (TEXTS[4], SUMMARIES[4])


def test_examples_are_rendered_in_prompts():
    """Render example pairs, glossary terms and image examples in the prompts."""
    backend = ScriptedBackend(answer='ok')
    backend.default_models = {'generate_text': 'm'}
    backend.generate_text(TextGenerationRequest('long text', SUMMARIZE, examples=(('a text', 'a summary'),)))
    assert 'Example 1:\nInput: a text\nOutput: a summary' in backend.prompts[-1].user
    request = TextGenerationRequest('hi', 'Translate into French.', (('a', 'b'),), glossary=(('SkaiBox', 'SkaiBox'),))
    backend.generate_text(request)
    assert '- SkaiBox → SkaiBox' in backend.prompts[-1].user
    assert 'Input: a\nOutput: b' in backend.prompts[-1].user
    image = Image.new('RGB', (4, 4))
    backend.generate_text(TextGenerationRequest(image, 'Describe the image.', examples=((image, 'A black square.'),)))
    prompt = backend.prompts[-1]
    assert len(prompt.images) == 2
    assert 'Example 1:\nInput: <image 1>\nOutput: A black square.' in prompt.user
    assert 'Input: <image 2>' in prompt.user


def test_mismatched_targets_are_rejected():
    """Reject references whose length differs from the samples."""
    with pytest.raises(ValueError, match='3 samples but `y` has 2 references'):
        summarizer(backend='dummy').fit(TEXTS[:3], SUMMARIES[:2])


def test_pipeline_target_is_not_read_as_references(audio):
    """Learn no vocabulary from class labels in an intermediate transcriber."""
    pipeline = make_pipeline(AudioToTextGenerator(backend='dummy'), TextClassifier(backend='dummy'))
    pipeline.fit([audio, audio], ['Kubernetes', 'Docker'])
    assert pipeline[0].vocabulary_ == ()
    assert pipeline[1].classes_.tolist() == ['Docker', 'Kubernetes']


def test_supervised_final_step_learns_from_references(audio):
    """Learn from references in the final step and not in the intermediate one."""
    pipeline = make_pipeline(AudioToTextGenerator(backend='dummy'), summarizer(backend='dummy'))
    pipeline.fit([audio, audio], ['First Summary.', 'Second Summary.'])
    assert pipeline[0].vocabulary_ == ()
    assert [reference for _, reference in pipeline[1].examples_] == ['First Summary.', 'Second Summary.']


def test_use_y_overrides_the_rule(audio):
    """Learn in `fit_transform` with `use_y=True` and never with `use_y=False`."""
    transcriber = AudioToTextGenerator(use_y=True, backend='dummy')
    transcriber.fit_transform([audio], ['We use Kubernetes.'])
    assert transcriber.vocabulary_ == ('Kubernetes',)
    assert AudioToTextGenerator(use_y=False, backend='dummy').fit([audio], ['We use Kubernetes.']).vocabulary_ == ()


def test_references_are_scored_in_cross_validation():
    """Learn from training references and score against test references in every fold."""

    def overlap(references, summaries):
        """Return the share of summaries that contain their reference's topic number."""
        pairs = zip(references, summaries, strict=True)
        return np.mean([reference.split()[-1] in summary for reference, summary in pairs])

    scores = cross_val_score(summarizer(backend='dummy'), TEXTS, SUMMARIES, cv=3, scoring=make_scorer(overlap))
    assert scores.shape == (3,)


def test_classifier_selects_one_example_per_class():
    """Select one example of every class by default and none with `k_shot=0`."""
    y = [index % 3 for index in range(12)]
    classifier = TextClassifier(backend='dummy').fit(TEXTS, y)
    assert sorted(labels for _, labels in classifier.examples_) == [('0',), ('1',), ('2',)]
    assert TextClassifier(k_shot=0, backend='dummy').fit(TEXTS, y).examples_ == ()


def test_summarizer_learns_length():
    """Learn the output length from the references when `max_length='auto'`."""
    references = [' '.join(['word'] * length) for length in (8, 9, 10, 11, 12)]
    generator = summarizer(max_length='auto', backend='dummy').fit(TEXTS[:5], references)
    assert 8 <= generator.max_length_ <= 12
    long_text = ' '.join(['token'] * 50)
    assert len(generator.predict([long_text])[0].split()) <= generator.max_length_
    assert summarizer(max_length='auto', backend='dummy').fit().max_length_ is None
    assert summarizer(backend='dummy').fit(TEXTS[:5], references).max_length_ is None


def test_translator_learns_glossary():
    """Map a product name that keeps its translation in two references."""
    sources = ['Open SkaiBox now.', 'SkaiBox is ready.', 'Hello there.']
    references = ['Ouvrez SkaiBox.', 'SkaiBox est prête.', 'Bonjour.']
    translator = TextToTextGenerator(instructions='Translate into French.', learn_terms=True)
    translator.set_params(backend=RecordingBackend())
    translator.fit(sources, references)
    assert translator.glossary_ == (('SkaiBox', 'SkaiBox'),)
    assert TextToTextGenerator(instructions='Translate.', backend='dummy').fit(sources, references).glossary_ == ()
    translator.predict(['Close SkaiBox.'])
    assert RecordingBackend.requests[-1].glossary == (('SkaiBox', 'SkaiBox'),)


def test_descriptor_sends_style_examples(image):
    """Send `k_shot` image and description pairs with every request."""
    images = [image, Image.new('RGB', (8, 8)), Image.new('RGB', (16, 16))]
    descriptions = ['A red block.', 'A black block.', 'A larger black block.']
    descriptor = ImageToTextGenerator(instructions='Describe.', k_shot=2, random_state=0, backend=RecordingBackend())
    descriptor.fit(images, descriptions)
    descriptor.predict([image])
    assert len(RecordingBackend.requests[-1].examples) == 2


def test_transcriber_learns_vocabulary(audio):
    """Learn domain terms and send them with every request."""
    transcriber = AudioToTextGenerator(backend=RecordingBackend()).fit([audio], ['We run Kubernetes over gRPC.'])
    assert set(transcriber.vocabulary_) == {'Kubernetes', 'gRPC'}
    transcriber.predict([audio])
    assert RecordingBackend.requests[-1].vocabulary == transcriber.vocabulary_


def test_transcriber_warns_without_context(audio):
    """Warn when the backend cannot use the learned vocabulary."""

    class NoContextBackend(DummyBackend):
        """Take no context."""

        accepts_context = False

    with pytest.warns(UnusedFitDataWarning, match='vocabulary is not used'):
        AudioToTextGenerator(backend=NoContextBackend()).fit([audio], ['We run Kubernetes.'])
