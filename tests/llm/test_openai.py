"""Test OpenAI models."""

import numpy as np
import openai
import pytest
from sklearn.exceptions import NotFittedError

from skai.llm import OpenAIClassifier
from tests.conftest import (
    CLASSES,
    CLASSES_MAPPING,
    K_SHOT_NONE_CLASSES_MISSING_DATA,
    K_SHOT_ZERO_CLASSES_MISSING_DATA,
    X,
    X_test,
    y,
    y_test,
)


def test_default_classifier_init() -> None:
    """Tests the initialization with default parameters."""
    model = None
    k_shot = None
    prompt = None
    openai_client = None
    responses_kwargs = None
    classes = None
    classifier = OpenAIClassifier(
        model=model,
        k_shot=k_shot,
        prompt=prompt,
        openai_client=openai_client,
        responses_kwargs=responses_kwargs,
        classes=classes,
    )
    assert classifier.model is model
    assert classifier.k_shot is k_shot
    assert classifier.prompt is prompt
    assert classifier.openai_client is openai_client
    assert classifier.responses_kwargs is responses_kwargs
    assert classifier.classes is classes


def test_classifier_init() -> None:
    """Tests the initialization with parameters."""
    model = 'gpt-3.5-turbo'
    k_shot = 5
    prompt = 'Classify this text into one of these categories'
    openai_client = 'api_key'
    responses_kwargs = {'temperature': 0.7, 'max_tokens': 50}
    classes = np.array(['positive', 'negative', 'neutral'])
    classifier = OpenAIClassifier(
        model=model,
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


@pytest.mark.parametrize('model', [-5, 0.4, []])
def test_classifier_fit_type_error_model(model) -> None:
    """Tests the fit method with wrong model value."""
    classifier = OpenAIClassifier(model=model)
    with pytest.raises(TypeError, match='The \'model\' parameter of'):
        classifier.fit(X, y)


def test_classifier_fit_error_model() -> None:
    """Tests the fit method with wrong model value."""
    classifier = OpenAIClassifier(model='model')
    with pytest.raises(ValueError, match='Parameter `model` must be one'):
        classifier.fit(X, y)


@pytest.mark.parametrize('k_shot', [-5, 0.4, 'zero', openai.AsyncOpenAI, ['one', 1], []])
def test_classifier_fit_error_k_shot(k_shot) -> None:
    """Tests the fit method with wrong k-shot value."""
    classifier = OpenAIClassifier(k_shot=k_shot)
    with pytest.raises(TypeError, match='The \'k_shot\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('prompt', [5, ['prompt'], openai.AsyncOpenAI])
def test_classifier_fit_error_prompt(prompt) -> None:
    """Tests the fit method with wrong prompt."""
    classifier = OpenAIClassifier(prompt=prompt)
    with pytest.raises(TypeError, match='The \'prompt\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('openai_client', [True, ['key'], openai.AsyncOpenAI])
def test_classifier_fit_error_openai_client(openai_client) -> None:
    """Tests the fit method with wrong openai_client."""
    classifier = OpenAIClassifier(openai_client=openai_client)
    with pytest.raises(TypeError, match='The \'openai_client\' parameter of'):
        classifier.fit(X, y)


@pytest.mark.parametrize('classes', [True, 'class_label'])
def test_classifier_fit_error_classes(classes) -> None:
    """Tests the fit method with wrong classes."""
    classifier = OpenAIClassifier(classes=classes)
    with pytest.raises(TypeError, match='The \'classes\' parameter of'):
        classifier.fit(X, y)


def test_classifier_fit_error_no_classes_no_targets() -> None:
    """Tests the fit method when no classes and no targets are provided."""
    classifier = OpenAIClassifier()
    with pytest.raises(ValueError, match='Parameter `classes` must be provided when `y` is `None`'):
        classifier.fit(X=X)


@pytest.mark.parametrize('classes', [[0, 1], ['a', 'b', 'c']])
def test_classifier_fit_error_type_classes_targets(classes) -> None:
    """Tests the fit method when wrong classes and targets are provided."""
    classifier = OpenAIClassifier(classes=classes)
    with pytest.raises(TypeError, match='Parameter `classes` must be a dictonary of class'):
        classifier.fit(X, y)


@pytest.mark.parametrize('classes', [{}, {0: 'a'}, {1: 'a', 2: 'b'}])
def test_classifier_fit_error_value_classes_targets(classes) -> None:
    """Tests the fit method when wrong classes and targets are provided."""
    classifier = OpenAIClassifier(classes=classes)
    with pytest.raises(ValueError, match='The unique labels in `y` must be equal to the'):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    ('classes', 'X', 'y'),
    K_SHOT_NONE_CLASSES_MISSING_DATA,
)
def test_classifier_fit_k_shot_none_missing_data(classes, X, y) -> None:
    """Tests the fit method and k_shot attribute when data are missing."""
    classifier = OpenAIClassifier(classes=classes)
    classifier.fit(X=X, y=y)
    assert classifier.k_shot_ == 0
    assert classifier.k_shot_examples_ is None


@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_k_shot_none(classes) -> None:
    """Tests the fit method and k_shot attribute when data are provided."""
    classifier = OpenAIClassifier(classes=classes)
    classifier.fit(X=X, y=y)
    assert classifier.k_shot_ == np.unique(y).size
    assert np.array_equal(
        np.array([cls for _, cls in classifier.k_shot_examples_]),
        np.unique(y),
    )


@pytest.mark.parametrize(
    ('classes', 'X', 'y'),
    K_SHOT_ZERO_CLASSES_MISSING_DATA,
)
def test_classifier_fit_k_shot_zero(classes, X, y) -> None:
    """Tests the fit method and k_shot attribute for zero k_shot."""
    classifier = OpenAIClassifier(
        k_shot=0,
        classes=classes,
    )
    classifier.fit(X=X, y=y)
    assert classifier.k_shot_ == 0
    assert classifier.k_shot_examples_ is None


@pytest.mark.parametrize('k_shot', [3, 5])
@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_k_shot_positive(k_shot, classes) -> None:
    """Tests the fit method and k_shot attribute for positive k_shot."""
    classifier = OpenAIClassifier(
        k_shot=k_shot,
        classes=classes,
    )
    classifier.fit(X=X, y=y)
    assert classifier.k_shot_ == len(classifier.k_shot_examples_) == k_shot


@pytest.mark.parametrize('k_shot', [[0, 1], [4, 2]])
@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_k_shot_array(k_shot, classes) -> None:
    """Tests the fit method and k_shot attribute for list k_shot."""
    classifier = OpenAIClassifier(
        k_shot=k_shot,
        classes=classes,
    )
    classifier.fit(X=X, y=y)
    assert classifier.k_shot_ == len(k_shot)
    assert classifier.k_shot_examples_ == list(zip(np.array(X)[k_shot], y[k_shot], strict=False))


@pytest.mark.parametrize('prompt', [None, 'Please classify the input.', 'Provide a classification of the input.'])
@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_prompt_none_k_shot_zero(prompt, classes) -> None:
    """Tests the fit method and prompt attribute."""
    classifier = OpenAIClassifier(
        k_shot=0,
        prompt=prompt,
        classes=classes,
    )
    classifier.fit(X=X, y=y)
    if prompt is None:
        prompt = classifier.DEFAULT_PROMPT
    assert classifier.prompt_ == prompt


@pytest.mark.parametrize('k_shot', [None, 3, 1, [0, 1]])
@pytest.mark.parametrize('prompt', [None, 'Please classify the input.', 'Provide a classification of the input.'])
@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_prompt_none_k_shot_non_zero(k_shot, prompt, classes) -> None:
    """Tests the fit method and prompt attribute."""
    classifier = OpenAIClassifier(
        k_shot=k_shot,
        prompt=prompt,
        classes=classes,
    )
    classifier.fit(X=X, y=y)
    if prompt is None:
        prompt = classifier.DEFAULT_PROMPT
    for i, example in enumerate(classifier.k_shot_examples_):
        output = example[1] if classes != CLASSES_MAPPING else CLASSES_MAPPING[example[1]]
        prompt += f'\n\nExample {i + 1}:\nInput: {example[0]}\nOutput: {output}'
    assert classifier.prompt_ == prompt


@pytest.mark.parametrize('openai_client', [None, 'api_key', openai.AsyncOpenAI()])
def test_classifier_fit_openai_client_async(openai_client) -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(openai_client=openai_client)
    classifier.fit(X=X, y=y)
    assert isinstance(classifier.openai_client_, openai.AsyncOpenAI)


def test_classifier_fit_openai_client_sync() -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(openai_client=openai.OpenAI())
    classifier.fit(X=X, y=y)
    assert isinstance(classifier.openai_client_, openai.OpenAI)


def test_classifier_fit_responses_kwargs_none() -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier()
    classifier.fit(X=X, y=y)
    assert classifier.responses_kwargs_ == {}


@pytest.mark.parametrize('responses_kwargs', [{}, {'temperature': 0.5}])
def test_classifier_fit_responses_kwargs(responses_kwargs) -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(responses_kwargs=responses_kwargs)
    classifier.fit(X=X, y=y)
    assert classifier.responses_kwargs_ == responses_kwargs


def test_classifier_fit_classes_none() -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier()
    classifier.fit(X=X, y=y)
    assert np.array_equal(classifier.classes_, np.unique(y))


def test_classifier_fit_classes_list() -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(classes=CLASSES)
    classifier.fit(X=X)
    assert np.array_equal(classifier.classes_, np.sort(CLASSES))


def test_classifier_fit_classes_dict() -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(classes=CLASSES_MAPPING)
    classifier.fit(X=X, y=y)
    assert np.array_equal(classifier.classes_, np.sort(list(CLASSES_MAPPING.values())))


@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_fit_instructions(classes) -> None:
    """Tests the fit method with default parameters."""
    classifier = OpenAIClassifier(classes=classes)
    classifier.fit(X=X, y=y)
    classes_repr = ', '.join([str(cls) for cls in classifier.classes_])
    assert classifier.instructions_ == classifier.DEFAULT_INSTRUCTIONS.format(classes_repr)


def test_classifier_predict_not_fitted() -> None:
    """Tests the predict method of classifier when is not fitted."""
    classifier = OpenAIClassifier()
    with pytest.raises(NotFittedError, match='This OpenAIClassifier instance is not fitted yet.'):
        classifier.predict(X_test)


@pytest.mark.parametrize('k_shot', [0, 3, [0, 2]])
@pytest.mark.parametrize('classes', [None, CLASSES_MAPPING])
def test_classifier_predict(k_shot, classes) -> None:
    """Tests the predict method of classifier when is not fitted."""
    classifier = OpenAIClassifier(k_shot=k_shot, classes=classes)
    classifier.fit(X=X, y=y)
    y_pred = classifier.predict(X_test)
    assert y_pred.dtype == y_test.dtype
    assert set(y_pred).issubset(y)
