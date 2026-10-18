"""Define the scikit-learn base classes of the media components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import warnings
from collections import Counter
from numbers import Integral
from os import PathLike
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from ._backend import Backend
from ._errors import (
    InvalidSampleError,
    MediaReadError,
    SampleError,
    SampleFailureWarning,
    UnsupportedBackendError,
    UnusedFitDataWarning,
)
from ._io import InputKind, MediaKind, read_input, read_multimodal, to_object_array, to_samples, write_outputs
from ._media import Audio, Video
from ._prompts import NO_LABEL, FineTuneRequest, build_task_prompt
from ._registry import list_backends, resolve_backend
from ._requests import ClassificationRequest, EmbeddingRequest, ExtractionRequest
from ._terms import find_terms
from ._utils import draw_seeds

PROBE_TEXT = 'dimension probe'
PROBE_SIZE = 32
PROBE_SAMPLE_RATE = 16000
MULTILABEL_THRESHOLD = 0.5
DEFAULT_MAX_EXAMPLES = 8
PROBE_FRAMES = 4


class MultiRequest(tuple):
    """Hold several backend requests of one sample, whose results are returned together as a list."""

    __slots__ = ()


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
    """Resolve the backend, batch samples and handle failed samples for media components.

    A component declares its input kind and its operations as class variables. `operations` maps every operation to the
    backend task that runs it, or to `None` for an operation that runs locally without a backend. The first operation is
    the default, and the parameter named by `operation_param` selects one at `fit`. `default_backends` overrides
    `default_backend` for some operations.
    """

    input_kind: ClassVar[InputKind]
    operations: ClassVar[dict[str, str | None]]
    operation_param: ClassVar[str] = 'operation'
    required_keys: ClassVar[dict[str, tuple[str, ...]]] = {}
    list_operations: ClassVar[frozenset[str]] = frozenset()
    default_backend: ClassVar[str] = 'hf'
    default_backends: ClassVar[dict[str, str]] = {}
    _parameter_constraints: ClassVar[dict] = {
        'backend': [str, Backend, None],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'on_error': [StrOptions({'raise', 'ignore'})],
        'random_state': ['random_state'],
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate the operation parameter of a component against its operations."""
        super().__init_subclass__(**kwargs)
        operations = getattr(cls, 'operations', None)
        if operations and cls.operation_param in inspect.signature(cls.__init__).parameters:
            constraint = [StrOptions(set(operations))]
            cls._parameter_constraints = {**cls._parameter_constraints, cls.operation_param: constraint}

    def __sklearn_tags__(self) -> Tags:
        """Declare the input type and the non-deterministic output of the component."""
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.input_tags.two_d_array = False
        tags.input_tags.string = self.input_kind == 'text'
        return tags

    def _operation(self) -> str:
        """Return the selected operation."""
        return getattr(self, self.operation_param, next(iter(self.operations)))

    def _connect(self) -> None:
        """Validate the parameters, resolve the task of the operation and resolve the backend."""
        self._validate_params()
        self.task_: str | None = self.operations[self._operation()]
        if self.task_ is None:
            self.backend_: Backend | None = None
            return
        default = self.default_backends.get(self._operation(), self.default_backend)
        backend = default if self.backend is None else self.backend
        self.backend_ = resolve_backend(backend, self.task_, self.input_kind)

    def _fine_tune(self, samples: list[Any], answers: list[str]) -> None:
        """Train a model on the samples and their answers, and send later requests to it without examples."""
        assert self.backend_ is not None
        assert self.task_ is not None
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
            build_task_prompt(self.task_, self._build_request(self._read(sample, index), 0))
            for index, sample in enumerate(samples)
        ]
        self.model_ = self.backend_.fine_tune(FineTuneRequest(self.task_, tuple(zip(prompts, answers, strict=True))))
        tuned = clone(self.backend_)
        tuned.set_params(model=self.model_)
        self.backend_ = tuned.connect()

    def _samples(self, X: ArrayLike | None) -> list[Any]:
        """Convert the input into a list of samples, reading DataFrame rows as multimodal samples."""
        return to_samples(X, rows=self.input_kind == 'multimodal')

    def _read_one(self, sample: Any, index: int) -> Any:
        """Read one sample, or one item of a list-valued sample, as the input kind of the component."""
        n_frames = getattr(self, 'n_frames', None)
        if self.input_kind == 'multimodal':
            return read_multimodal(sample, index, n_frames, self.required_keys.get(self._operation(), ()))
        return read_input(self.input_kind, sample, index, n_frames)

    def _read(self, sample: Any, index: int) -> Any:
        """Read one sample, accepting a list of samples only for list operations."""
        if isinstance(sample, list):
            if self._operation() not in self.list_operations:
                msg = (
                    f'Sample {index} is a list, which {type(self).__name__} does not accept. Wrap the step in '
                    '`ForEach` to apply it to every item.'
                )
                raise InvalidSampleError(msg)
            return [self._read_one(item, index) for item in sample]
        return self._read_one(sample, index)

    def _report_failures(self, failures: dict[int, Exception]) -> None:
        """Raise the first failure or warn about all of them, depending on `on_error`."""
        if not failures:
            return
        if getattr(self, 'on_error', 'raise') == 'raise':
            index, error = min(failures.items())
            msg = f'Sample {index} failed: {error}'
            raise SampleError(msg) from error
        details = '; '.join(f'sample {index}: {error}' for index, error in sorted(failures.items()))
        msg = f'{len(failures)} samples failed and were replaced by missing values ({details}).'
        warnings.warn(msg, SampleFailureWarning, stacklevel=4)

    def _empty_result(self) -> Any:
        """Return the result of a sample whose request builder asks to skip the backend."""

    def _complete(self, result: Any, sample: Any) -> Any:
        """Turn the backend result of one sample into the result of the component, with the read sample at hand."""
        return result

    def _run_local(self, sample: Any, seed: int) -> Any:
        """Run a local operation on one read sample."""
        raise NotImplementedError

    def _read_batch(self, samples: list[Any], indices: range) -> tuple[dict[int, Any], dict[int, Exception]]:
        """Read the samples of one batch, collecting the samples that cannot be read."""
        read: dict[int, Any] = {}
        failures: dict[int, Exception] = {}
        for index in indices:
            try:
                read[index] = self._read(samples[index], index)
            except MediaReadError as error:
                if getattr(self, 'on_error', 'raise') == 'raise':
                    raise
                failures[index] = error
        return read, failures

    def _run_backend(self, read: dict[int, Any], seeds: list[int]) -> dict[int, Any]:
        """Build the requests of read samples, run them on the backend and return every outcome by index."""
        assert self.backend_ is not None
        assert self.task_ is not None
        outcomes: dict[int, Any] = {}
        requests: dict[int, Any] = {}
        for index, sample in read.items():
            request = self._build_request(sample, seeds[index])
            if request is None:
                outcomes[index] = self._empty_result()
            else:
                requests[index] = request
        flat = [
            item
            for request in requests.values()
            for item in (request if isinstance(request, MultiRequest) else [request])
        ]
        results = iter(self.backend_.run(self.task_, flat))
        for index, request in requests.items():
            if not isinstance(request, MultiRequest):
                outcomes[index] = next(results)
                continue
            parts = [next(results) for _ in request]
            errors = [part for part in parts if isinstance(part, Exception)]
            outcomes[index] = errors[0] if errors else parts
        return outcomes

    def _run_task(self, X: ArrayLike) -> list[Any]:
        """Run the operation of the component on every sample, in batches, keeping the input order."""
        check_is_fitted(self, 'backend_')
        samples = self._samples(X)
        random_state = getattr(self, 'random_state', None)
        seeds = draw_seeds(random_state, len(samples))
        results: list[Any] = [None] * len(samples)
        failures: dict[int, Exception] = {}
        batch_size = getattr(self, 'batch_size', max(1, len(samples)))
        for start in range(0, len(samples), batch_size):
            read, read_failures = self._read_batch(samples, range(start, min(start + batch_size, len(samples))))
            failures.update(read_failures)
            if self.task_ is None:
                outcomes = {index: self._call_local(sample, seeds[index]) for index, sample in read.items()}
            else:
                outcomes = self._run_backend(read, seeds)
            for index, outcome in outcomes.items():
                if isinstance(outcome, Exception):
                    failures[index] = outcome
                    continue
                if self.task_ is None:
                    results[index] = outcome
                    continue
                try:
                    results[index] = self._complete(outcome, read[index])
                except (MediaReadError, ValueError) as error:
                    failures[index] = error
        self._report_failures(failures)
        return results

    def _call_local(self, sample: Any, seed: int) -> Any:
        """Run a local operation, returning its exception unless the sample has the wrong type."""
        try:
            return self._run_local(sample, seed)
        except InvalidSampleError:
            raise
        except Exception as error:  # noqa: BLE001 - failures are returned per sample so components can apply `on_error`
            return error

    def _build_request(self, sample: Any, seed: int) -> Any:
        """Build the backend request of one sample, `None` to skip the backend, or a `MultiRequest`."""
        raise NotImplementedError


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
        if not references or getattr(self, 'k_shot', 0) == 0:
            self.examples_ = ()
            return
        max_examples = getattr(self, 'max_examples', DEFAULT_MAX_EXAMPLES)
        indices = _select_indices(self.k_shot, list(range(len(samples))), self.random_state, max_examples)
        self.examples_ = tuple((self._read(samples[index], index), references[index]) for index in indices)

    def _learn_vocabulary(self, references: list[Any]) -> None:
        """Learn the most frequent domain terms of reference transcripts, warning when the backend takes no context."""
        counts = Counter(term for reference in references for term in find_terms(str(reference)))
        self.vocabulary_ = tuple(term for term, _ in counts.most_common(self.max_vocabulary))
        if self.vocabulary_ and self.backend_ is not None and not self.backend_.accepts_context:
            msg = f'{type(self.backend_).__name__} takes no context, so the learned vocabulary is not used.'
            warnings.warn(msg, UnusedFitDataWarning, stacklevel=5)

    def _fit(self, X: ArrayLike | None, y: ArrayLike | None) -> Self:
        """Validate the parameters, resolve the backend and learn from the references."""
        self._connect()
        self._prepare()
        samples = self._samples(X) if y is not None else []
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
        return self._fit(X, None if getattr(self, 'use_y', 'auto') is False else y)

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
        return self._fit(X, y if getattr(self, 'use_y', 'auto') is True else None).transform(X)


class BaseMediaGenerator(BaseGenerator):
    """Generate images, audio or video and return them in the requested output format.

    Local operations need the `video` extra for videos, and no backend.
    """

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
        samples = self._samples(X)
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
        proba = np.full((len(self._samples(X)), len(names)), np.nan)
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

    def _build_request(self, sample: Any, seed: int) -> EmbeddingRequest | MultiRequest:
        """Build the embedding request of one sample, asking for fewer dimensions only when the backend can."""
        assert self.backend_ is not None
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
        assert self.backend_ is not None
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
        is learned when the backend cannot shorten embeddings. Without `X`, a small probe sample is embedded to find the
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
        embeddings = [embedding for embedding in self._embed(self._samples(X)) if embedding is not None]
        if embeddings:
            self._learn(np.stack(embeddings))
            return self
        probes = {
            'text': PROBE_TEXT,
            'image': Image.new('RGB', (PROBE_SIZE, PROBE_SIZE)),
            'audio': Audio(np.zeros(PROBE_SAMPLE_RATE, dtype=np.float32), PROBE_SAMPLE_RATE),
            'video': Video(np.zeros((PROBE_FRAMES, PROBE_SIZE, PROBE_SIZE, 3), dtype=np.uint8), fps=PROBE_FRAMES),
        }
        probe = self._embed([probes[self.input_kind]])[0]
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


class BaseExtractor(TransformerMixin, BaseMediaEstimator):
    """Extract a record of declared fields from every sample, learning from reference records in `fit`.

    `predict` returns one dict per sample, and `transform` one column per top-level field.
    """

    operations: ClassVar[dict[str, str | None]] = {'extract': 'extract_fields'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaEstimator._parameter_constraints,
        'fields': [dict],
        'instructions': [str, None],
        'k_shot': [Interval(Integral, 0, None, closed='left'), 'array-like', None],
        'max_examples': [Interval(Integral, 1, None, closed='left'), None],
    }

    def _names(self) -> tuple[str, ...]:
        """Return the top-level field names in order."""
        return tuple(str(name) for name in self.fields)

    def _build_request(self, sample: Any, seed: int) -> ExtractionRequest:
        """Build the extraction request of one sample."""
        fields = tuple((str(name), declaration) for name, declaration in self.fields.items())
        return ExtractionRequest(sample, fields, self.instructions, self.examples_)

    def _complete(self, result: Any, sample: Any) -> dict[str, Any]:
        """Keep exactly the declared fields, with `None` for absent ones."""
        return {name: result.get(name) for name in self._names()}

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Resolve the backend and select example pairs of samples and their reference records.

        Args:
            X:
                The training samples. They can be `None` to extract zero-shot.

            y:
                The reference record of every sample, as dicts keyed by field name.

        Returns:
            The fitted extractor.

        Raises:
            ValueError:
                If `X` and `y` have different lengths.
        """
        self._connect()
        samples = self._samples(X) if y is not None else []
        references = list(y) if y is not None else []
        if len(samples) != len(references):
            msg = f'`X` has {len(samples)} samples but `y` has {len(references)} records.'
            raise ValueError(msg)
        self.examples_: tuple[tuple[Any, Any], ...] = ()
        if references and self.k_shot != 0:
            indices = _select_indices(self.k_shot, list(range(len(samples))), self.random_state, self.max_examples)
            self.examples_ = tuple((self._read(samples[index], index), dict(references[index])) for index in indices)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        """Extract the record of every sample.

        Args:
            X:
                The samples.

        Returns:
            One dict per sample whose keys are the field names, or `None` for a failed sample.
        """
        return to_object_array(self._run_task(X))

    def transform(self, X: ArrayLike) -> NDArray[np.object_]:
        """Extract the fields of every sample as columns.

        Args:
            X:
                The samples.

        Returns:
            An object array of shape `(n_samples, n_fields)`. A failed sample has a row of `None`.
        """
        names = self._names()
        records = self._run_task(X)
        table = np.empty((len(records), len(names)), dtype=object)
        for row, record in enumerate(records):
            table[row] = [None if record is None else record[name] for name in names]
        return table

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Return the top-level field names.

        Args:
            input_features:
                Ignored.

        Returns:
            The field names, in the order of `fields`.
        """
        check_is_fitted(self, 'backend_')
        return np.asarray(self._names(), dtype=object)


class BaseSegmenter(BaseSupervisedGenerator):
    """Split every sample into a list of region records, chosen with the `by` parameter.

    Every region is a dict with at least `label`, `score` and `media`, the cropped piece of the sample.
    """

    operation_param: ClassVar[str] = 'by'

    def _finalize(self, results: list[Any]) -> NDArray:
        """Return one list of regions per sample."""
        return to_object_array(results)
