"""Test fine-tuning in `fit`."""

from typing import ClassVar

import pytest
from PIL import Image

from skai.backends import DummyBackend
from skai.classification import TextClassifier
from skai.core import UnsupportedBackendError
from skai.description import ImageDescriptor
from skai.text import TextSummarizer, TextTranslator
from tests.fakes import ScriptedBackend

TEXTS = [f'review number {index}' for index in range(20)]
LABELS = ['positive' if index % 2 else 'negative' for index in range(20)]


class TuningBackend(DummyBackend):
    """Keep the fine-tuning requests and every later request."""

    tunings: ClassVar[list] = []
    requests: ClassVar[list] = []

    def fine_tune(self, request):
        """Keep the request and fine-tune like the dummy backend."""
        TuningBackend.tunings.append(request)
        return super().fine_tune(request)

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        TuningBackend.requests.extend(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_records():
    """Start every test with no recorded requests."""
    TuningBackend.tunings = []
    TuningBackend.requests = []


def test_classifier_uses_fine_tuned_model():
    """Train on every sample and send later requests to the new model without examples."""
    classifier = TextClassifier(adaptation='finetune', backend=TuningBackend()).fit(TEXTS, LABELS)
    assert classifier.model_.startswith('dummy-finetuned-')
    assert classifier.backend_.model == classifier.model_
    assert classifier.examples_ == ()
    tuning = TuningBackend.tunings[-1]
    assert tuning.task == 'classify_text'
    assert len(tuning.examples) == 20
    prompt, answer = tuning.examples[1]
    assert answer == 'positive'
    assert prompt.user.endswith('Input: review number 1\nOutput:')
    assert set(classifier.predict(TEXTS[:3])) <= {'negative', 'positive'}
    assert TuningBackend.requests[-1].examples == ()


def test_generators_fine_tune_on_references():
    """Fine-tune the summarizer, translator and descriptor on their references."""
    summaries = [text.split()[-1] for text in TEXTS]
    summarizer = TextSummarizer(adaptation='finetune', backend=TuningBackend()).fit(TEXTS, summaries)
    assert summarizer.model_.startswith('dummy-finetuned-')
    assert TuningBackend.tunings[-1].examples[0][1] == summaries[0]
    translator = TextTranslator(target_language='Greek', adaptation='finetune', backend=TuningBackend())
    assert translator.fit(TEXTS, summaries).backend_.model == translator.model_
    images = [Image.new('RGB', (4, 4), (index, 0, 0)) for index in range(20)]
    descriptor = ImageDescriptor(adaptation='finetune', backend=TuningBackend()).fit(images, summaries)
    assert len(TuningBackend.tunings[-1].examples[0][0].images) == 1
    assert descriptor.predict(images[:1])[0]


def test_fine_tuning_requires_a_capable_backend():
    """List the backends that fine-tune when the backend cannot."""
    backend = ScriptedBackend(answer='positive')
    backend.default_models = {'classify_text': 'scripted'}
    with pytest.raises(UnsupportedBackendError, match=r"cannot fine-tune.*\['dummy', 'openai'\]"):
        TextClassifier(adaptation='finetune', backend=backend).fit(TEXTS, LABELS)


def test_fine_tuning_requires_enough_samples():
    """State the minimum when there are too few samples."""
    with pytest.raises(ValueError, match='at least 10 samples, but 5'):
        TextClassifier(adaptation='finetune', backend='dummy').fit(TEXTS[:5], LABELS[:5])


def test_fine_tuning_requires_targets():
    """Ask for targets or references."""
    with pytest.raises(ValueError, match='needs the targets'):
        TextClassifier(labels=['a', 'b'], adaptation='finetune', backend='dummy').fit()
    with pytest.raises(ValueError, match='needs reference outputs'):
        TextSummarizer(adaptation='finetune', backend='dummy').fit(TEXTS)
