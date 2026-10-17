"""Define the scikit-learn base classes of the media components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from abc import abstractmethod
from collections.abc import Callable
from numbers import Integral
from os import PathLike
from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from ._backend import Backend
from ._errors import MediaReadError, SampleError, SampleFailureWarning, UnsupportedBackendError
from ._io import MediaKind, check_text, read_audio, read_image, read_video, to_object_array, to_samples, write_outputs
from ._prompts import NO_LABEL, FineTuneRequest, build_task_prompt
from ._registry import list_backends, resolve_backend
from ._requests import ClassificationRequest, EmbeddingRequest

MAX_SEED = 2**31 - 1
PROBE_TEXT = 'dimension probe'
MULTILABEL_THRESHOLD = 0.5
DEFAULT_MAX_EXAMPLES = 8
READERS: dict[str, Callable[[Any, int], Any]] = {
    'text': check_text,
    'image': read_image,
    'audio': read_audio,
    'video': read_video,
}


def _select_indices(
    k_shot: int | ArrayLike | None,
    groups: list[Any],
    random_state: int | np.random.RandomState | None,
    max_examples: int | None,
) -> NDArray[np.int_]:
    """Select the indices of the examples among samples whose group labels are given.

    `k_shot=None` takes one random sample per group, capped at `max_examples`. `0` takes none, an integer takes that
    many at random, and an array takes those indices.
    """
    rng = check_random_state(random_state)
    if k_shot is None:
        by_group: dict[Any, list[int]] = {}
        for index, group in enumerate(groups):
            by_group.setdefault(group, []).append(index)
        indices = np.array(sorted(int(rng.choice(members)) for members in by_group.values()), dtype=int)
        if max_examples is not None and indices.shape[0] > max_examples:
            indices = np.sort(rng.choice(indices, size=max_examples, replace=False))
        return indices
    if isinstance(k_shot, Integral):
        count = int(k_shot)
        if count > len(groups):
            msg = f'`k_shot` is {count} but only {len(groups)} samples were given.'
            raise ValueError(msg)
        return np.sort(rng.choice(len(groups), size=count, replace=False))
    return np.asarray(k_shot, dtype=int)


class BaseMediaEstimator(BaseEstimator):
    """Resolve the backend, batch samples and handle failed samples for media components."""

    task: ClassVar[str]
    input_kind: ClassVar[Literal['text', 'image', 'audio', 'video']]
    default_backend: ClassVar[str] = 'hf'
    _parameter_constraints: ClassVar[dict] = {
        'backend': [str, Backend, None],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'on_error': [StrOptions({'raise', 'ignore'})],
        'random_state': ['random_state'],
    }

    def __sklearn_tags__(self) -> Tags:
        """Declare the input type and the non-deterministic output of the component."""
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.input_tags.two_d_array = False
        tags.input_tags.string = self.input_kind == 'text'
        return tags

    def _connect(self) -> None:
        """Validate the parameters and resolve the backend."""
        self._validate_params()
        backend = self.default_backend if self.backend is None else self.backend
        self.backend_ = resolve_backend(backend, self.task)

    def _fine_tune(self, samples: list[Any], answers: list[str]) -> None:
        """Train a model on the samples and their answers, and send later requests to it without examples."""
        if 'fine_tune' not in type(self.backend_).supported_tasks():
            backends = list_backends('fine_tune')
            msg = f'{type(self.backend_).__name__} cannot fine-tune. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        minimum = self.backend_.min_finetune_examples
        if len(samples) < minimum:
            msg = f'Fine-tuning needs at least {minimum} samples, but {len(samples)} were given.'
            raise ValueError(msg)
        self.examples_: tuple[tuple[Any, Any], ...] = ()
        prompts = [
            build_task_prompt(self.task, self._build_request(self._read(sample, index), 0))
            for index, sample in enumerate(samples)
        ]
        self.model_ = self.backend_.fine_tune(FineTuneRequest(self.task, tuple(zip(prompts, answers, strict=True))))
        tuned = clone(self.backend_)
        tuned.set_params(model=self.model_)
        self.backend_ = tuned.connect()

    def _read(self, sample: Any, index: int) -> Any:
        """Read one sample as the input kind of the component."""
        return READERS[self.input_kind](sample, index)

    def _report_failures(self, failures: dict[int, Exception]) -> None:
        """Raise the first failure or warn about all of them, depending on `on_error`."""
        if not failures:
            return
        if self.on_error == 'raise':
            index, error = min(failures.items())
            msg = f'Sample {index} failed: {error}'
            raise SampleError(msg) from error
        details = '; '.join(f'sample {index}: {error}' for index, error in sorted(failures.items()))
        msg = f'{len(failures)} samples failed and were replaced by missing values ({details}).'
        warnings.warn(msg, SampleFailureWarning, stacklevel=4)

    def _build_requests(self, samples: list[Any], indices: range, seeds: list[int]) -> tuple[dict[int, Any], dict]:
        """Read the samples of one batch and build their requests."""
        requests: dict[int, Any] = {}
        failures: dict[int, Exception] = {}
        for index in indices:
            try:
                requests[index] = self._build_request(self._read(samples[index], index), seeds[index])
            except MediaReadError as error:
                if self.on_error == 'raise':
                    raise
                failures[index] = error
        return requests, failures

    def _run_task(self, X: ArrayLike) -> list[Any]:
        """Run the task of the component on every sample, in batches, keeping the input order."""
        check_is_fitted(self, 'backend_')
        samples = to_samples(X)
        seeds = check_random_state(self.random_state).randint(0, MAX_SEED, size=len(samples)).tolist()
        results: list[Any] = [None] * len(samples)
        failures: dict[int, Exception] = {}
        for start in range(0, len(samples), self.batch_size):
            indices = range(start, min(start + self.batch_size, len(samples)))
            requests, read_failures = self._build_requests(samples, indices, seeds)
            failures.update(read_failures)
            outcomes = self.backend_.run(self.task, list(requests.values()))
            for index, outcome in zip(requests, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    failures[index] = outcome
                else:
                    results[index] = outcome
        self._report_failures(failures)
        return results

    @abstractmethod
    def _build_request(self, sample: Any, seed: int) -> Any:
        """Build the backend request of one sample."""


class BaseGenerator(TransformerMixin, BaseMediaEstimator):
    """Generate one output per sample, returned by both `predict` and `transform`."""

    def _prepare(self) -> None:
        """Validate and derive the state of the component after the backend is resolved."""

    def _finalize(self, results: list[Any]) -> NDArray:
        """Turn the backend results into the output array."""
        return to_object_array(results)

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Validate the parameters and resolve the backend.

        Args:
            X:
                Ignored. The component learns nothing from data.

            y:
                Ignored.

        Returns:
            The fitted component.
        """
        self._connect()
        self._prepare()
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Generate one output per sample.

        Args:
            X:
                The samples.

        Returns:
            The outputs, in input order.
        """
        return self._finalize(self._run_task(X))

    def transform(self, X: ArrayLike) -> NDArray:
        """Generate one output per sample, so the component can be an intermediate pipeline step.

        Args:
            X:
                The samples.

        Returns:
            The outputs, in input order.
        """
        return self.predict(X)


class BaseSupervisedGenerator(BaseGenerator):
    """Generate outputs and learn from reference outputs given as `y` in `fit`.

    `fit(X, y)` learns from the references. `fit_transform(X, y)`, which pipelines use for intermediate steps, ignores
    `y` unless `use_y=True`, because `y` is then the target of the final step.
    """

    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'k_shot': [Interval(Integral, 0, None, closed='left'), 'array-like', None],
        'max_examples': [Interval(Integral, 1, None, closed='left'), None],
        'use_y': [StrOptions({'auto'}), 'boolean'],
    }

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Learn the fitted state from samples and their references, which are empty without `y`."""
        indices = _select_indices(self.k_shot, list(range(len(samples))), self.random_state, self.max_examples)
        self.examples_ = tuple((self._read(samples[index], index), references[index]) for index in indices)

    def _fit(self, X: ArrayLike | None, y: ArrayLike | None) -> Self:
        """Validate the parameters, resolve the backend and learn from the references."""
        self._connect()
        self._prepare()
        samples = to_samples(X) if y is not None else []
        references = to_samples(y) if y is not None else []
        if len(samples) != len(references):
            msg = f'`X` has {len(samples)} samples but `y` has {len(references)} references.'
            raise ValueError(msg)
        self._learn(samples, references)
        if getattr(self, 'adaptation', 'examples') == 'finetune':
            if not references:
                msg = 'Fine-tuning needs reference outputs as `y`.'
                raise ValueError(msg)
            self._fine_tune(samples, [str(reference) for reference in references])
        return self

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Learn from reference outputs.

        Args:
            X:
                The training samples.

            y:
                The reference output of every sample, such as its summary or transcript. Without it, the component
                predicts zero-shot.

        Returns:
            The fitted component.
        """
        return self._fit(X, None if self.use_y is False else y)

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any) -> NDArray:
        """Fit and return the outputs, ignoring `y` unless `use_y=True`.

        Pipelines fit intermediate steps through this method and pass them the target of the final step, so `y` is
        not read as references by default.

        Args:
            X:
                The samples.

            y:
                The references, used only when `use_y=True`.

            **fit_params:
                Ignored.

        Returns:
            The outputs, in input order.
        """
        return self._fit(X, y if self.use_y is True else None).transform(X)


class BaseMediaGenerator(BaseGenerator):
    """Generate images, audio or video and return them in the requested output format."""

    output_kind: ClassVar[MediaKind]
    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'output_format': [StrOptions({'object', 'bytes', 'path'})],
        'output_dir': [str, PathLike, None],
    }

    def _finalize(self, results: list[Any]) -> NDArray:
        """Return the generated media in the requested output format."""
        prefix = type(self).__name__.lower()
        return write_outputs(results, self.output_kind, self.output_format, self.output_dir, prefix)


class BaseClassifier(ClassifierMixin, BaseMediaEstimator):
    """Assign labels with zero-shot or few-shot prediction."""

    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaEstimator._parameter_constraints,
        'labels': [list, dict, None],
        'instructions': [str, None],
        'k_shot': [Interval(Integral, 0, None, closed='left'), 'array-like', None],
        'adaptation': [StrOptions({'examples', 'finetune'})],
    }

    def __sklearn_tags__(self) -> Tags:
        """Declare that the classifier supports multilabel targets."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags

    def _fit_labels(self) -> None:
        """Set the classes from the `labels` parameter for zero-shot classification."""
        if self.labels is None:
            msg = 'Provide the targets `y` or the `labels` parameter.'
            raise ValueError(msg)
        self.classes_ = np.unique(list(self.labels))
        self.multilabel_ = False

    def _fit_targets(self, samples: list[Any], y: ArrayLike) -> NDArray:
        """Set the classes from the targets and return the validated targets."""
        y = np.asarray(y)
        if len(samples) != y.shape[0]:
            msg = f'`X` has {len(samples)} samples but `y` has {y.shape[0]} targets.'
            raise ValueError(msg)
        self.multilabel_ = type_of_target(y) == 'multilabel-indicator'
        if self.multilabel_:
            self.classes_ = np.asarray(list(self.labels)) if self.labels is not None else np.arange(y.shape[1])
            if self.classes_.shape[0] != y.shape[1]:
                msg = f'`labels` has {self.classes_.shape[0]} labels but `y` has {y.shape[1]} columns.'
                raise ValueError(msg)
            return y
        check_classification_targets(y)
        y = y.ravel()
        self.classes_ = np.unique(y)
        if self.labels is not None:
            unknown = sorted(set(self.classes_.tolist()) - set(self.labels))
            if unknown:
                msg = f'The targets contain values that are not in `labels`: {unknown}.'
                raise ValueError(msg)
            self.classes_ = np.unique(list(self.labels))
        return y

    def _label_names_of(self, target: Any) -> tuple[str, ...]:
        """Return the label names of one target."""
        if self.multilabel_:
            return tuple(str(label) for label, value in zip(self.classes_, target, strict=True) if value)
        return (str(target),)

    def _select_examples(self, samples: list[Any], y: NDArray | None) -> tuple[tuple[Any, tuple[str, ...]], ...]:
        """Select the labeled examples that are passed to the backend, one per class by default."""
        if y is None:
            return ()
        groups = [self._label_names_of(target) for target in y]
        indices = _select_indices(self.k_shot, groups, self.random_state, None)
        return tuple((self._read(samples[index], index), groups[index]) for index in indices)

    def _build_request(self, sample: Any, seed: int) -> ClassificationRequest:
        """Build the classification request of one sample."""
        labels = self.labels if isinstance(self.labels, dict) else {}
        descriptions = {str(label): text for label, text in labels.items()}
        return ClassificationRequest(
            sample=sample,
            labels=tuple(str(label) for label in self.classes_),
            descriptions=descriptions,
            instructions=self.instructions,
            examples=self.examples_,
            multi_label=self.multilabel_,
        )

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Learn the classes and select the few-shot examples.

        Args:
            X:
                The training samples. They can be `None` for zero-shot classification with `labels`.

            y:
                The targets, as labels or as a binary indicator matrix for multilabel classification. They can be
                `None` for zero-shot classification with `labels`.

        Returns:
            The fitted classifier.

        Raises:
            ValueError:
                If `adaptation='finetune'` and no targets are given.
        """
        self._connect()
        samples = to_samples(X)
        targets = None
        if y is None:
            self._fit_labels()
        else:
            targets = self._fit_targets(samples, y)
        if self.adaptation == 'finetune':
            if targets is None:
                msg = 'Fine-tuning needs the targets `y`.'
                raise ValueError(msg)
            answers = [', '.join(self._label_names_of(target)) or NO_LABEL for target in targets]
            self._fine_tune(samples, answers)
            return self
        self.examples_ = self._select_examples(samples, targets)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Score every sample against every class.

        Args:
            X:
                The samples.

        Returns:
            An array of shape `(n_samples, n_classes)`. Rows sum to one, except for multilabel classification where
            every column is an independent probability. A failed sample has a row of NaN.
        """
        check_is_fitted(self, 'classes_')
        names = [str(label) for label in self.classes_]
        proba = np.full((len(to_samples(X)), len(names)), np.nan)
        for row, scores in enumerate(self._run_task(X)):
            if scores is not None:
                proba[row] = [scores.get(name, 0.0) for name in names]
        if self.multilabel_:
            return proba
        totals = proba.sum(axis=1, keepdims=True)
        proba = np.where(totals == 0, 1.0 / len(names), proba)
        return proba / np.where(totals == 0, 1.0, totals)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the labels of every sample.

        Args:
            X:
                The samples.

        Returns:
            The predicted labels, or a binary indicator matrix for multilabel classification. A failed sample is
            `None`, or a row of NaN for multilabel classification.
        """
        proba = self.predict_proba(X)
        failed = np.isnan(proba).any(axis=1)
        if self.multilabel_:
            indicator = (proba >= MULTILABEL_THRESHOLD).astype(int)
            return np.where(np.isnan(proba), np.nan, indicator) if failed.any() else indicator
        labels = self.classes_[np.nan_to_num(proba, nan=0.0).argmax(axis=1)]
        if not failed.any():
            return labels
        labels = labels.astype(object)
        labels[failed] = None
        return labels


class BaseEmbedder(TransformerMixin, BaseMediaEstimator):
    """Map every sample to a fixed-size vector, learning centering and a projection from the samples in `fit`."""

    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaEstimator._parameter_constraints,
        'dimensions': [Interval(Integral, 1, None, closed='left'), None],
        'center': ['boolean'],
        'normalize': ['boolean'],
    }

    def _build_request(self, sample: Any, seed: int) -> EmbeddingRequest:
        """Build the embedding request of one sample, asking for fewer dimensions only when the backend can."""
        native = self.dimensions if self.backend_.shortens_embeddings else None
        return EmbeddingRequest(sample, native)

    def _embed(self, X: ArrayLike) -> list[NDArray[np.float64] | None]:
        """Return the raw embedding of every sample, or `None` for a failed sample."""
        return [None if embedding is None else np.asarray(embedding, dtype=float) for embedding in self._run_task(X)]

    def _to_vector(self, embedding: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project or shorten, center and normalize one raw embedding."""
        if self.components_ is not None:
            vector = (embedding - self.mean_) @ self.components_.T
        else:
            vector = embedding[: self.n_features_out_]
            vector = vector - self.mean_ if self.center else vector
        if not self.normalize:
            return vector
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _learn(self, embeddings: NDArray[np.float64]) -> None:
        """Learn the mean and, when the backend cannot shorten embeddings, a PCA projection to `dimensions`."""
        if self.dimensions is not None and not self.backend_.shortens_embeddings:
            if embeddings.shape[0] < self.dimensions:
                msg = f'A projection to {self.dimensions} dimensions needs at least {self.dimensions} samples.'
                raise ValueError(msg)
            self.mean_ = embeddings.mean(axis=0)
            _, _, vt = np.linalg.svd(embeddings - self.mean_, full_matrices=False)
            self.components_: NDArray[np.float64] | None = vt[: self.dimensions]
            self.n_features_out_ = self.dimensions
            return
        shortened = embeddings[:, : self.dimensions]
        self.mean_ = shortened.mean(axis=0)
        self.components_ = None
        self.n_features_out_ = shortened.shape[1]

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Embed the samples and learn their mean and, when needed, a projection.

        With `X`, every sample is embedded, the mean is learned for centering, and a PCA projection to `dimensions`
        is learned when the backend cannot shorten embeddings. Without `X`, a short probe text is embedded to find the
        number of dimensions, and nothing is centered or projected.

        Args:
            X:
                The samples.

            y:
                Ignored.

        Returns:
            The fitted embedder.
        """
        self._connect()
        embeddings = [embedding for embedding in self._embed(to_samples(X)) if embedding is not None]
        if embeddings:
            self._learn(np.stack(embeddings))
            return self
        probe = self._embed([PROBE_TEXT])[0]
        assert probe is not None
        self.n_features_out_ = probe[: self.dimensions].shape[0]
        self.mean_ = np.zeros(self.n_features_out_)
        self.components_ = None
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Embed every sample.

        Args:
            X:
                The samples.

        Returns:
            An array of shape `(n_samples, n_features_out_)`. A failed sample has a row of NaN.
        """
        results = self._embed(X)
        embeddings = np.full((len(results), self.n_features_out_), np.nan)
        for row, embedding in enumerate(results):
            if embedding is not None:
                embeddings[row] = self._to_vector(embedding)
        return embeddings

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Return the names of the embedding dimensions.

        Args:
            input_features:
                Ignored.

        Returns:
            The names `<class name><index>`.
        """
        check_is_fitted(self, 'n_features_out_')
        prefix = type(self).__name__.lower()
        return np.asarray([f'{prefix}{index}' for index in range(self.n_features_out_)], dtype=object)
