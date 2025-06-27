"""Implementation of OpenAI classes."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import asyncio
import warnings
from numbers import Integral
from typing import ClassVar, Self

import numpy as np
import openai
from dotenv import Any, load_dotenv
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    Tags,
    TargetTags,
    _fit_context,
    check_array,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import ClassifierTags
from sklearn.utils._param_validation import Interval, InvalidParameterError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

load_dotenv()


class OpenAIClassifier(ClassifierMixin, MultiOutputMixin, BaseEstimator):
    """OpenAI classifier.

    A classifier that is based on the OpenAI API.

    Args:
        k_shot:
            Number of examples to include into the prompt. If `None` then no examples
            are included. If `int` then it is equal to the number of example that are
            selected randomly. If an array of indices is provided then it is used
            to select the examples.
        prompt:
            Prompt for the OpenAI API.
        openai_client:
            OpenAI client.
        responses_kwargs:
            Keyword arguments for the OpenAI API.
        classes:
            Class labels.

    Attributes:
        k_shot_ (int | dict[int, ArrayLike | np.random.RandomState] | None):
            Number of examples per class.
        prompt_ (str | None):
            Prompt for the OpenAI API.
        openai_client_ (openai.OpenAI | openai.AsyncOpenAI):
            OpenAI client.
        responses_kwargs_ (dict | None):
            Keyword arguments for the OpenAI API.
        classes_ (np.ndarray | list[np.ndarray] | None):
            Class labels.
        instructions_ (str):
            Instructions for the OpenAI API.
    """

    _estimator_type: ClassVar[str] = 'classifier'
    _parameter_constraints: ClassVar[dict] = {
        'k_shot': [Interval(Integral, 0, None, closed='left'), 'array-like', None],
        'prompt': [str, None],
        'responses_kwargs': [dict, None],
        'openai_client': [str, object, None],
        'classes': ['array-like', None],
    }
    DEFAULT_PROMPT = 'Please classify the following input into one of the available classes.'
    DEFAULT_INSTRUCTIONS = (
        'You are a machine learning classifier. You should only provide an answer that '
        'is a single word or a number from the available class labels {}. '
        'Do not provide any additional information.'
    )

    def __init__(
        self: Self,
        k_shot: int | ArrayLike | None = None,
        prompt: str | None = None,
        openai_client: str | openai.OpenAI | openai.AsyncOpenAI | None = None,
        responses_kwargs: dict | None = None,
        classes: np.ndarray | list[np.ndarray] | None = None,
    ) -> None:
        self.k_shot = k_shot
        self.prompt = prompt
        self.responses_kwargs = responses_kwargs
        self.openai_client = openai_client
        self.classes = classes

    def __sklearn_tags__(self: Self) -> Tags:
        """Classifier tags."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(
            poor_score=False,
            multi_class=True,
            multi_label=True,
        )
        tags.target_tags = TargetTags(
            required=True,
            multi_output=True,
            single_output=True,
        )
        return tags

    def _validate_openai_client(self: Self) -> openai.OpenAI | openai.AsyncOpenAI:
        if isinstance(self.openai_client, openai.OpenAI | openai.AsyncOpenAI):
            return self.openai_client
        elif isinstance(self.openai_client, str):
            return openai.AsyncOpenAI(api_key=self.openai_client)
        elif self.openai_client is None:
            return openai.AsyncOpenAI()
        else:
            error_msg = (
                'The \'openai_client\' parameter of OpenAIClassifier must be a '
                'string, openai.OpenAI or openai.AsyncOpenAI instance.'
            )
            raise TypeError(error_msg)

    def _validate_responses_kwargs(self: Self) -> dict[str, Any]:
        if self.responses_kwargs is not None:
            return self.responses_kwargs
        return {}

    def _validate_prompt(self: Self) -> str:
        if self.prompt is not None:
            return self.prompt
        return self.DEFAULT_PROMPT

    def _validate_data(self: Self, X: ArrayLike | None, y: ArrayLike | None) -> tuple[NDArray | None, NDArray | None]:
        if X is not None and y is not None:
            X, y = validate_data(
                self,
                X,
                y,
                validate_separately=({'ensure_2d': False, 'dtype': str}, {'ensure_2d': False, 'dtype': None}),
            )
        elif X is None and y is not None:
            y = check_array(y, ensure_2d=False, dtype=None)
        elif y is None and X is not None:
            X = check_array(X, ensure_2d=False, dtype=str)
        if y is not None:
            ndim_vector = 2
            if y.ndim == 1 or (y.ndim == ndim_vector and y.shape[1] == 1):
                if y.ndim != 1:
                    warnings.warn(
                        (
                            "A column-vector y was passed when a "
                            "1d array was expected. Please change "
                            "the shape of y to (n_samples,), for "
                            "example using ravel()."
                        ),
                        DataConversionWarning,
                        stacklevel=2,
                    )

                self.outputs_2d_ = False
                y = y.reshape((-1, 1))
            else:
                self.outputs_2d_ = True
            check_classification_targets(y)
        return X, y

    def _validate_instructions(self: Self) -> str:
        return self.DEFAULT_INSTRUCTIONS.format(self.classes_)

    def _validate_classes(self: Self, X: NDArray | None, y: NDArray | None) -> NDArray:
        if self.classes is not None and y is not None:
            error_msg = 'Parameter `classes` must be None when `y` is provided.'
            raise ValueError(error_msg)
        elif self.classes is None and y is None:
            error_msg = 'Parameter `classes` must be provided when `y` is `None`.'
            raise ValueError(error_msg)
        elif self.classes is None and y is not None:
            classes = []
            for k in range(y.shape[1]):
                classes.append(np.unique(y[:, k]))
            if not self.outputs_2d_:
                classes = classes[0]
        elif self.classes is not None and y is None:
            classes = np.sort(check_array(self.classes, ensure_2d=False, dtype=None))
        return classes

    def _validate_k_shot(self, X: np.typing.NDArray | None, y: np.typing.NDArray | None) -> int:
        if isinstance(self.k_shot, np.ndarray | list | tuple):
            if not all(isinstance(k, Integral) for k in self.k_shot):
                error_msg = 'The \'k_shot\' parameter of OpenAIClassifier must be an array-like object of integers.'
                raise InvalidParameterError(error_msg)
            k_shot = check_array(self.k_shot, ensure_2d=False, dtype=int).shape[0]
            return k_shot
        if (X is None or y is None) and (self.k_shot is not None and self.k_shot != 0):
            error_msg = 'Parameter `k_shot` must be `None` or `0` when input data `X` or labels `y` is None.'
            raise ValueError(error_msg)
        if self.k_shot is None:
            return 0
        if X is not None and self.k_shot > X.shape[0]:
            error_msg = 'Parameter `k_shot` must be less than or equal to the number of examples in `X`.'
            raise ValueError(error_msg)
        return self.k_shot

    def _select_k_shot_examples(self: Self, X: ArrayLike, y: ArrayLike) -> Self:
        self.k_shot_examples_ = None
        if self.k_shot is None:
            return self
        elif isinstance(self.k_shot, int):
            if self.k_shot > 0:
                indices = np.random.choice(X.shape[0], size=self.k_shot, replace=False)
                self.k_shot_examples_ = list(zip(X[indices], y[indices], strict=False))
            return self
        indices = np.array(self.k_shot)
        self.k_shot_examples_ = list(zip(X[indices], y[indices], strict=False))
        return self

    def _add_k_shot_examples(self: Self) -> Self:
        if self.k_shot_examples_ is not None:
            for i, example in enumerate(self.k_shot_examples_):
                self.prompt_ += f'\n\nExample {i + 1}:\nInput: {example[0]}\nOutput: {example[1]}'
        return self

    def _fit(self, X: ArrayLike | None, y: ArrayLike | None) -> Self:

        # Validate client
        self.openai_client_ = self._validate_openai_client()

        # Validate responses kwargs
        self.responses_kwargs_ = self._validate_responses_kwargs()

        # Validate prompt
        self.prompt_ = self._validate_prompt()

        # Validate data
        X, y = self._validate_data(X, y)

        # Validate classes
        self.classes_ = self._validate_classes(X, y)

        # Validate instructions
        self.instructions_ = self._validate_instructions()

        # Validate k_shot
        self.k_shot_ = self._validate_k_shot(X, y)

        # Select k_shot examples and add them to prompt
        self._select_k_shot_examples(X, y)._add_k_shot_examples()

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Fit the classifier to the training dataset.

        Args:
            X:
                Input data.

            y:
                Target values.

        Returns:
            self:
                The fitted OpenAI classifier.
        """
        return self._fit(X, y)

    def _get_api_call_args(self: Self, x: str) -> dict:
        prompt = f'{self.prompt_}\n\nInput: {x}\nOutput:'
        return {
            'model': 'gpt-3.5-turbo',
            'input': prompt,
            'instructions': self.instructions_,
            **self.responses_kwargs_,
        }

    async def _predict_async(self, X: list[str]) -> list[str]:

        async def _predict_single(x: str) -> str:
            response = await self.openai_client_.responses.create(**self._get_api_call_args(x))
            return response.output_text

        predictions = [_predict_single(x) for x in X]
        return await asyncio.gather(*predictions)

    def _predict_sync(self, X: ArrayLike) -> NDArray:
        predictions = []
        for x in X:
            response = self.openai_client_.responses.create(**self._get_api_call_args(x))
            predictions.append(response.output_text)
        return np.array(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the class labels for the provided data.

        Args:
            X:
                Input data.

        Returns:
            The predicted class labels.
        """
        check_is_fitted(
            self,
            ('k_shot_', 'openai_client_', 'responses_kwargs_', 'prompt_', 'classes_', 'instructions_'),
        )
        X = check_array(X, ensure_2d=False, dtype=str)
        if isinstance(self.openai_client_, openai.OpenAI):
            predictions = self._predict_sync(X)
        else:
            predictions = asyncio.run(self._predict_async(X.tolist()))
        return np.array(predictions)
