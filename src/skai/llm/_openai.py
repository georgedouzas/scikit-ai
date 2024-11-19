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
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin, Tags, TargetTags, _fit_context, check_array
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import ClassifierTags
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

load_dotenv()


class OpenAIClassifier(ClassifierMixin, MultiOutputMixin, BaseEstimator):

    _instructions = (
        'You are a machine learning classifier. You should only provide an answer that '
        'is a comma seperated sequence of single words or number from the available class labels {}. '
        'Do not provide any additional information.'
    )
    _parameter_constraints: ClassVar[dict] = {
        'k_shot': [Interval(Integral, 0, None, closed='left'), None],
        'prompt': [str, None],
        'prompt_optimization': [bool],
        'responses_kwargs': [dict, None],
        'openai_client': [str, object, None],
        'classes': ['array-like', None],
    }

    def __init__(
        self: Self,
        k_shot: int | dict[int, ArrayLike | np.random.RandomState] | None = None,
        prompt: str | None = None,
        prompt_optimization: bool = False,
        openai_client: str | openai.OpenAI | openai.AsyncOpenAI | None = None,
        responses_kwargs: dict | None = None,
        classes: np.ndarray | list[np.ndarray] | None = None,
    ) -> None:
        self.k_shot = k_shot
        self.prompt = prompt
        self.prompt_optimization = prompt_optimization
        self.responses_kwargs = responses_kwargs
        self.openai_client = openai_client
        self.classes = classes

    def _validate_openai_client(self: Self) -> openai.OpenAI | openai.AsyncOpenAI:
        if isinstance(self.openai_client, openai.OpenAI | openai.AsyncOpenAI):
            return self.openai_client
        elif isinstance(self.openai_client, str):
            return openai.AsyncOpenAI(api_key=self.openai_client)
        elif self.openai_client is None:
            return openai.AsyncOpenAI()
        else:
            error_msg = 'Parameter `openai_client` must be a string, openai.OpenAI or openai.AsyncOpenAI instance.'
            raise TypeError(error_msg)

    def _validate_prompt(self: Self) -> str:
        if isinstance(self.prompt, str):
            return self.prompt
        elif self.prompt is None:
            return 'Please classify the following input into one of the available classes.'
        error_msg = 'Parameter `prompt` must be a string.'
        raise TypeError(error_msg)

    def _optimize_prompt(self: Self, X: ArrayLike, y: ArrayLike) -> str:
        error_msg = 'Prompt optimization is not implemented.'
        raise NotImplementedError(error_msg)

    def _select_k_shot_examples(self: Self, X: ArrayLike, y: ArrayLike) -> ArrayLike | None:
        if isinstance(self.k_shot, int) and self.k_shot > 0:
            if self.k_shot > X.shape[0]:
                error_msg = 'Parameter `k_shot` must be less than or equal to the number of examples in `X`.'
                raise ValueError(error_msg)
            random_state = np.random.default_rng()
            return random_state.choice(X, size=self.k_shot, replace=False)
        elif isinstance(self.k_shot, dict):
            Xs, ys = [], []
            for k, v in self.k_shot.items():
                if k > X.shape[0]:
                    error_msg = 'Parameter `k_shot` must be less than or equal to the number of examples in `X`.'
                    raise ValueError(error_msg)
                if isinstance(v, np.random.RandomState):
                    indices = random_state.choice(v, size=k, replace=False)
                    Xs.append(X[indices])
                    ys.append(y[indices])
                elif isinstance(v, ArrayLike):
                    if v.shape[0] != X.shape[0]:
                        error_msg = 'Parameter `k_shot` must be less than or equal to the number of examples in `X`.'
                        raise ValueError(error_msg)
                    Xs.append(X[v])
                    ys.append(y[v])
                else:
                    error_msg = 'Parameter `k_shot` must be an integer or a dictionary of integers and array-likes.'
                    raise TypeError(error_msg)
            return np.concatenate(Xs), np.concatenate(ys)
        return None

    def _add_k_shot_examples(self: Self, X_k_shot: ArrayLike, y_k_shot: ArrayLike) -> Self:
        for i in range(X_k_shot.shape[0]):
            self.prompt_ += f'\n\nExample {i + 1}:\nInput: {X_k_shot[i]}\nOutput: {y_k_shot[i]}'
        return self

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

    def _fit(self, X: ArrayLike | None, y: ArrayLike | None) -> Self:
        X, y = validate_data(
            self,
            X,
            y,
            validate_separately=({'ensure_2d': False, 'dtype': str}, {'ensure_2d': False, 'dtype': None}),
        )
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
        self.classes_ = []
        for k in range(y.shape[1]):
            classes = np.unique(y[:, k])
            self.classes_.append(classes)
        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
        self.openai_client_ = self._validate_openai_client()
        self.prompt_ = self._validate_prompt()
        if self.prompt_optimization:
            self.prompt_ = self._optimize_prompt(X, y)
        if self.k_shot is not None:
            k_shot_examples = self._select_k_shot_examples(X, y)
            if k_shot_examples is not None:
                X_k_shot, y_k_shot = k_shot_examples
                self.prompt_ = self._add_k_shot_examples(X_k_shot, y_k_shot)
        if self.responses_kwargs is None:
            self.responses_kwargs_: dict[str, Any] = {}
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike | None, y: ArrayLike | None) -> Self:
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
            'instructions': self._instructions.format(self.classes_),
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
        X = check_array(X, ensure_2d=False, dtype=str)
        if isinstance(self.openai_client_, openai.OpenAI):
            predictions = self._predict_sync(X)
        else:
            predictions = asyncio.run(self._predict_async(X.tolist()))
        return np.array(predictions)
