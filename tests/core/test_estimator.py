"""Test the scikit-learn contract that every component follows."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.utils._param_validation import InvalidParameterError

from skai.audio import SpeechTranscriber, TalkingAvatarGenerator
from skai.backends import DummyBackend
from skai.core import SampleError, SampleFailureWarning
from skai.image import ImageClassifier, ImageDescriptor
from skai.text import ImageGenerator, SpeechSynthesizer, TextClassifier, TextEmbedder, TextSummarizer, TextTranslator

AVATAR = Image.new('RGB', (8, 8))
COMPONENTS = [
    TextClassifier(labels=['a', 'b'], backend='dummy', batch_size=3),
    TextSummarizer(max_length=5, backend='dummy'),
    TextTranslator(target_language='German', backend='dummy'),
    TextEmbedder(dimensions=4, backend='dummy'),
    SpeechSynthesizer(voice='alloy', backend='dummy'),
    ImageGenerator(size=(8, 8), backend='dummy'),
    ImageDescriptor(detail='detailed', backend='dummy'),
    ImageClassifier(labels=['cat', 'dog'], backend='dummy'),
    SpeechTranscriber(timestamps='segment', backend='dummy'),
    TalkingAvatarGenerator(avatar=AVATAR, backend='dummy'),
]
GENERATORS = [
    (TextSummarizer(backend='dummy'), ['a b c']),
    (TextTranslator(target_language='German', backend='dummy'), ['hello']),
    (ImageDescriptor(backend='dummy'), [AVATAR]),
    (SpeechTranscriber(backend='dummy'), [(np.zeros(800), 16000)]),
]
TEXTS = [f'text number {index}' for index in range(10)]


class FlakyBackend(DummyBackend):
    """Fail on texts that contain the word fail."""

    def summarize(self, request):
        """Fail or summarize like the dummy backend."""
        if 'fail' in request.text:
            msg = 'provider error'
            raise RuntimeError(msg)
        return super().summarize(request)

    def classify_text(self, request):
        """Fail or classify like the dummy backend."""
        if 'fail' in request.sample:
            msg = 'provider error'
            raise RuntimeError(msg)
        return super().classify_text(request)


class SeedRecordingBackend(DummyBackend):
    """Record the seed of every image request."""

    seeds: ClassVar[list[int]] = []

    def generate_image(self, request):
        """Record the seed and generate like the dummy backend."""
        SeedRecordingBackend.seeds.append(request.seed)
        return super().generate_image(request)


@pytest.mark.parametrize('component', COMPONENTS, ids=lambda component: type(component).__name__)
def test_clone_preserves_parameters(component):
    """Clone to an unfitted component with equal parameters."""
    cloned = clone(clone(component).fit())
    assert cloned.get_params() == component.get_params()
    assert not hasattr(cloned, 'backend_')


@pytest.mark.parametrize('component', COMPONENTS, ids=lambda component: type(component).__name__)
def test_predict_before_fit_raises(component):
    """Raise `NotFittedError` before `fit`."""
    component = clone(component)
    method = component.transform if isinstance(component, TextEmbedder) else component.predict
    with pytest.raises(NotFittedError):
        method(['x'])


@pytest.mark.parametrize('component', COMPONENTS, ids=lambda component: type(component).__name__)
def test_constructor_takes_keyword_parameters_only(component):
    """Accept hyperparameters as keywords only."""
    with pytest.raises(TypeError):
        type(component)('positional')


def test_invalid_parameter_is_reported_at_fit():
    """Accept any value at construction and report invalid ones at `fit`."""
    summarizer = TextSummarizer(batch_size=0, backend='dummy')
    with pytest.raises(InvalidParameterError, match='batch_size'):
        summarizer.fit()


@pytest.mark.parametrize(('component', 'X'), GENERATORS)
def test_stateless_fit_ignores_data(component, X):
    """Return the component from `fit` with or without data."""
    assert component.fit() is component
    assert component.fit(X, ['ignored'] * len(X)) is component
    assert len(component.predict(X)) == len(X)


@pytest.mark.parametrize(('component', 'X'), GENERATORS)
def test_predict_equals_transform(component, X):
    """Return identical results from `predict` and `transform`."""
    component.fit()
    np.testing.assert_array_equal(component.predict(X), component.transform(X))


def test_converter_feeds_classifier_in_pipeline():
    """Feed the transcripts of the transcriber to the classifier in a pipeline."""
    audio = [(np.zeros(size), 16000) for size in (1600, 3200, 4800, 6400)]
    pipeline = make_pipeline(SpeechTranscriber(backend='dummy'), TextClassifier(backend='dummy'))
    predictions = pipeline.fit(audio, ['x', 'y', 'x', 'y']).predict(audio)
    assert predictions.shape == (4,)
    assert set(predictions) <= {'x', 'y'}


def test_classifier_learns_classes_from_targets():
    """Learn sorted classes from string targets and predict only those."""
    classifier = TextClassifier(backend='dummy').fit(['a', 'b', 'c'], ['spam', 'ham', 'spam'])
    assert classifier.classes_.tolist() == ['ham', 'spam']
    assert set(classifier.predict(TEXTS)) <= {'ham', 'spam'}
    assert 0 <= classifier.score(TEXTS[:2], ['ham', 'spam']) <= 1


def test_classifier_probabilities_sum_to_one():
    """Return probabilities of shape `(n, k)` whose rows sum to one."""
    classifier = TextClassifier(backend='dummy').fit(['a', 'b', 'c'], [0, 1, 2])
    proba = classifier.predict_proba(TEXTS)
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(classifier.predict(TEXTS), classifier.classes_[proba.argmax(axis=1)])


def test_classifier_uses_labels_without_targets():
    """Classify zero-shot over `labels` when `y` is `None`."""
    classifier = TextClassifier(labels={'pos': 'positive tone', 'neg': 'negative tone'}, backend='dummy').fit()
    assert classifier.classes_.tolist() == ['neg', 'pos']
    assert classifier.examples_ == ()


def test_classifier_requires_targets_or_labels():
    """Ask for targets or labels when both are missing."""
    with pytest.raises(ValueError, match='Provide the targets'):
        TextClassifier(backend='dummy').fit(TEXTS)


def test_classifier_rejects_mismatched_lengths():
    """Reject inputs and targets of different lengths."""
    with pytest.raises(ValueError, match='3 samples but `y` has 2'):
        TextClassifier(backend='dummy').fit(['a', 'b', 'c'], [0, 1])


def test_classifier_selects_examples_reproducibly():
    """Select the same examples for the same `k_shot` and `random_state`."""
    y = [index % 2 for index in range(10)]
    first = TextClassifier(k_shot=4, random_state=0, backend='dummy').fit(TEXTS, y).examples_
    second = TextClassifier(k_shot=4, random_state=0, backend='dummy').fit(TEXTS, y).examples_
    assert first == second
    assert len(first) == 4


def test_classifier_selects_examples_by_index():
    """Select the examples given as indices."""
    classifier = TextClassifier(k_shot=[0, 3], backend='dummy').fit(TEXTS, list('ababababab'))
    assert classifier.examples_ == (('text number 0', ('a',)), ('text number 3', ('b',)))


def test_classifier_rejects_too_many_examples():
    """Reject a `k_shot` larger than the number of samples."""
    with pytest.raises(ValueError, match='`k_shot` is 20'):
        TextClassifier(k_shot=20, backend='dummy').fit(TEXTS, [0] * 10)


def test_classifier_supports_multilabel_targets():
    """Learn from and predict a binary indicator matrix."""
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    classifier = TextClassifier(labels=['a', 'b', 'c'], k_shot=2, random_state=1, backend='dummy').fit(TEXTS[:3], y)
    assert classifier.multilabel_
    assert classifier.predict(TEXTS).shape == (10, 3)
    assert all(len(labels) >= 0 for _, labels in classifier.examples_)
    assert set(np.unique(classifier.predict(TEXTS))) <= {0, 1}


def test_classifier_rejects_labels_that_do_not_match_columns():
    """Reject `labels` whose length differs from the columns of a multilabel `y`."""
    with pytest.raises(ValueError, match='2 labels but `y` has 3 columns'):
        TextClassifier(labels=['a', 'b'], backend='dummy').fit(TEXTS[:2], np.eye(2, 3, dtype=int))


def test_embeddings_feed_scikit_learn_model():
    """Use the embeddings of `TextEmbedder` in a scikit-learn pipeline."""
    y = [index % 2 for index in range(10)]
    pipeline = make_pipeline(TextEmbedder(backend='dummy'), LogisticRegression())
    predictions = pipeline.fit(TEXTS, y).predict(TEXTS)
    assert predictions.shape == (10,)
    assert 0 <= accuracy_score(y, predictions) <= 1


def test_embedder_finds_dimensions_and_names_features():
    """Find the dimensions from a probe and name the output features."""
    embedder = TextEmbedder(normalize=True, backend='dummy').fit()
    assert embedder.n_features_out_ == 8
    assert embedder.get_feature_names_out().tolist()[:2] == ['textembedder0', 'textembedder1']
    np.testing.assert_allclose(np.linalg.norm(embedder.transform(TEXTS), axis=1), 1)


def test_batched_outputs_keep_input_order():
    """Return output `i` for input `i` with batches and concurrency."""
    summarizer = TextSummarizer(max_length=3, batch_size=3, backend=DummyBackend(max_concurrency=4)).fit()
    assert summarizer.predict(TEXTS).tolist() == TEXTS


def test_ignored_failures_are_reported_in_one_warning():
    """Return `None` for failed samples and name each of them in one warning."""
    summarizer = TextSummarizer(backend=FlakyBackend(), on_error='ignore').fit()
    with pytest.warns(SampleFailureWarning, match=r'2 samples failed.*sample 1: provider error; sample 2'):
        summaries = summarizer.predict(['fine', 'fail', 'fail again'])
    assert summaries.tolist() == ['fine', None, None]


def test_raised_failure_names_sample_index():
    """Raise `SampleError` naming the failed sample when `on_error='raise'`."""
    with pytest.raises(SampleError, match='Sample 1 failed: provider error'):
        TextSummarizer(backend=FlakyBackend()).fit().predict(['fine', 'fail'])


def test_classifier_marks_failed_samples():
    """Return NaN probabilities and `None` labels for failed samples."""
    classifier = TextClassifier(labels=['a', 'b'], backend=FlakyBackend(), on_error='ignore').fit()
    with pytest.warns(SampleFailureWarning):
        proba = classifier.predict_proba(['ok', 'fail'])
    assert np.isnan(proba[1]).all()
    with pytest.warns(SampleFailureWarning):
        assert classifier.predict(['ok', 'fail'])[1] is None


def test_multilabel_classifier_marks_failed_samples():
    """Return a row of NaN for a failed multilabel sample."""
    y = np.array([[1, 0], [0, 1]])
    classifier = TextClassifier(backend=FlakyBackend(), on_error='ignore').fit(['a', 'b'], y)
    with pytest.warns(SampleFailureWarning):
        predictions = classifier.predict(['ok', 'fail'])
    assert np.isnan(predictions[1]).all()


def test_embedder_marks_failed_samples():
    """Return a row of NaN for a failed embedding."""

    class FailingEmbedder(DummyBackend):
        """Fail on every embedding except the probe."""

        def embed_text(self, request):
            """Fail on texts that contain the word fail."""
            if 'fail' in request.text:
                msg = 'provider error'
                raise RuntimeError(msg)
            return super().embed_text(request)

    embedder = TextEmbedder(backend=FailingEmbedder(), on_error='ignore').fit()
    with pytest.warns(SampleFailureWarning):
        embeddings = embedder.transform(['ok', 'fail'])
    assert np.isnan(embeddings[1]).all()
    assert not np.isnan(embeddings[0]).any()


def test_random_state_controls_seeds():
    """Send the same seeds for the same `random_state`."""
    SeedRecordingBackend.seeds.clear()
    for _ in range(2):
        ImageGenerator(size=(4, 4), random_state=0, backend=SeedRecordingBackend()).fit().predict(['a cat'])
    assert SeedRecordingBackend.seeds[0] == SeedRecordingBackend.seeds[1]


def test_tags_declare_input_kind():
    """Declare string input for text components and non-deterministic outputs for all."""
    assert TextSummarizer().__sklearn_tags__().input_tags.string
    assert not ImageDescriptor().__sklearn_tags__().input_tags.string
    assert ImageDescriptor().__sklearn_tags__().non_deterministic
    assert TextClassifier().__sklearn_tags__().classifier_tags.multi_label
