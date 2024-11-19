"""Test OpenAI models."""

from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin

from skai.llm import OpenAIClassifier


def test_openai_classifier_init() -> None:
    """Tests the OpenAI classifier initialization."""
    classifier = OpenAIClassifier()
    assert isinstance(classifier, BaseEstimator | ClassifierMixin | MultiOutputMixin)
