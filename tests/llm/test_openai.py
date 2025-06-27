"""Test OpenAI models."""

import numpy as np
import openai
import pytest
from sklearn.exceptions import NotFittedError

from skai.llm import OpenAIClassifier

X = ['An animal', 'A vehicle', 'Sports car', 'Car of blue color', 'Friendly animal']
y = ['cat', 'car', 'car', 'car', 'cat']


def test_default_classifier_init() -> None:
    """Tests the initialization with default parameters."""
    k_shot = None
    prompt = None
    openai_client = None
    responses_kwargs = None
    classes = None
    classifier = OpenAIClassifier()
    assert classifier.k_shot is k_shot
    assert classifier.prompt is prompt
    assert classifier.openai_client is openai_client
    assert classifier.responses_kwargs is responses_kwargs
    assert classifier.classes is classes


def test_classifier_init() -> None:
    """Tests the initialization with parameters."""
    k_shot = 5
    prompt = 'Classify this text into one of these categories'
    openai_client = 'api_key'
    responses_kwargs = {'temperature': 0.7, 'max_tokens': 50}
    classes = np.array(['positive', 'negative', 'neutral'])
    classifier = OpenAIClassifier(
        k_shot=k_shot,
        prompt=prompt,
        openai_client=openai_client,
        responses_kwargs=responses_kwargs,
        classes=classes,
    )
    assert classifier.k_shot == k_shot
    assert classifier.prompt == prompt
    assert classifier.openai_client == openai_client
    assert classifier.responses_kwargs == responses_kwargs
    assert np.array_equal(classifier.classes, classes)


@pytest.mark.parametrize('k_shot', [-5, 0.4, 'zero', openai.AsyncOpenAI, ['one', 1]])
def test_classifier_fit_wrong_k_shot(k_shot) -> None:
    """Tests the fit method with wrong k-shot value."""
    classifier = OpenAIClassifier(k_shot=k_shot)
    with pytest.raises(TypeError, match='The \'k_shot\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('prompt', [5, ['prompt'], openai.AsyncOpenAI])
def test_classifier_fit_wrong_prompt(prompt) -> None:
    """Tests the fit method with wrong prompt."""
    classifier = OpenAIClassifier(prompt=prompt)
    with pytest.raises(TypeError, match='The \'prompt\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('openai_client', [True, ['key'], openai.AsyncOpenAI])
def test_classifier_fit_wrong_openai_client(openai_client) -> None:
    """Tests the fit method with wrong openai_client."""
    classifier = OpenAIClassifier(openai_client=openai_client)
    with pytest.raises(TypeError, match='The \'openai_client\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('classes', [True, 'class_label'])
def test_classifier_fit_wrong_classes(classes) -> None:
    """Tests the fit method with wrong classes."""
    classifier = OpenAIClassifier(classes=classes)
    with pytest.raises(TypeError, match='The \'classes\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('classes', [None, ['Cat', 'Car']])
@pytest.mark.parametrize('y', [None, y])
def test_classifier_fit_wrong_classes_targets(classes, y) -> None:
    """Tests the fit method with wrong classes and targets."""
    classifier = OpenAIClassifier(classes=classes)
    if classes is not None and y is not None:
        with pytest.raises(ValueError, match='Parameter `classes` must be None when `y` is provided.'):
            classifier.fit(X, y)
    elif classes is None and y is not None:
        classifier.fit(X, y)
        assert np.array_equal(classifier.classes_, np.unique(y))
    elif classes is not None and y is None:
        classifier.fit(X, y)
        assert np.array_equal(classifier.classes_, np.sort(classes))
    else:
        with pytest.raises(ValueError, match='Parameter `classes` must be provided when `y` is `None`'):
            classifier.fit(X, y)


@pytest.mark.parametrize('X', [None, X])
@pytest.mark.parametrize('y', [None, y])
def test_default_classifier_fit(X, y) -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier()
    if y is None:
        classifier.set_params(classes=np.unique(y))
    classifier.fit(X, y)
    assert isinstance(classifier.openai_client_, openai.AsyncOpenAI)
    assert classifier.responses_kwargs_ == {}
    assert np.array_equal(classifier.classes_, np.unique(y))
    assert classifier.instructions_ == classifier.DEFAULT_INSTRUCTIONS.format(classifier.classes_)
    assert classifier.k_shot_ == 0
    assert classifier.k_shot_examples_ is None
    assert classifier.prompt_ == classifier.DEFAULT_PROMPT


@pytest.mark.parametrize('X', [None, X])
@pytest.mark.parametrize('y', [None, y])
@pytest.mark.parametrize('prompt', [OpenAIClassifier.DEFAULT_PROMPT, 'Please classify the input.', None])
@pytest.mark.parametrize('openai_client', ['api_key', openai.OpenAI(), openai.AsyncOpenAI(), None])
def test_zero_shot_classifier_fit(X, y, prompt, openai_client) -> None:
    """Tests the fit method of a zero shot classifier."""

    # Fit classifier
    classifier = OpenAIClassifier(k_shot=0, prompt=prompt, openai_client=openai_client)
    if y is None:
        classifier.set_params(classes=np.unique(y))
    classifier.fit(X, y)

    # OpenAI client
    if isinstance(openai_client, str | openai.AsyncOpenAI | None):
        assert isinstance(classifier.openai_client_, openai.AsyncOpenAI)
        if isinstance(openai_client, str):
            assert classifier.openai_client_.api_key == openai_client
    else:
        assert isinstance(classifier.openai_client_, openai.OpenAI)

    # Responses kwargs
    assert classifier.responses_kwargs_ == {}

    # Classes
    assert np.array_equal(classifier.classes_, np.unique(y))

    # Instructions
    assert classifier.instructions_ == classifier.DEFAULT_INSTRUCTIONS.format(classifier.classes_)

    # K-shot
    assert classifier.k_shot_ == 0

    # K-shot examples
    assert classifier.k_shot_examples_ is None

    # Prompt
    if prompt is None:
        prompt = OpenAIClassifier.DEFAULT_PROMPT
    assert classifier.prompt_ == prompt


@pytest.mark.parametrize('k_shot', [1, 3, 4])
@pytest.mark.parametrize('prompt', [OpenAIClassifier.DEFAULT_PROMPT, 'Please classify the input.'])
@pytest.mark.parametrize('openai_client', ['api_key', openai.OpenAI(), openai.AsyncOpenAI(), None])
def test_few_shot_classifier_fit(k_shot, prompt, openai_client) -> None:
    """Tests the fit method of a few shot classifier."""

    # Fit classifier
    classifier = OpenAIClassifier(k_shot=k_shot, prompt=prompt, openai_client=openai_client).fit(X, y)

    # OpenAI client
    if isinstance(openai_client, str | openai.AsyncOpenAI | None):
        assert isinstance(classifier.openai_client_, openai.AsyncOpenAI)
        if isinstance(openai_client, str):
            assert classifier.openai_client_.api_key == openai_client
    else:
        assert isinstance(classifier.openai_client_, openai.OpenAI)

    # Responses kwargs
    assert classifier.responses_kwargs_ == {}

    # Classes
    assert np.array_equal(classifier.classes_, np.unique(y))

    # Instructions
    assert classifier.instructions_ == classifier.DEFAULT_INSTRUCTIONS.format(classifier.classes_)

    # K-shot
    assert classifier.k_shot_ == k_shot

    # K-shot examples
    assert len(classifier.k_shot_examples_) == classifier.k_shot_

    # Prompt
    for i, example in enumerate(classifier.k_shot_examples_):
        prompt += f'\n\nExample {i + 1}:\nInput: {example[0]}\nOutput: {example[1]}'
    assert classifier.prompt_ == prompt


def test_classifier_predict_not_fitted() -> None:
    """Tests the predict method of classifier when is not fitted."""
    classifier = OpenAIClassifier()
    with pytest.raises(NotFittedError, match='This OpenAIClassifier instance is not fitted yet.'):
        classifier.predict(X)
