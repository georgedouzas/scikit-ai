# Spec Delta

## Purpose

Defines the scikit-learn contract every scikit-ai media component follows, so that any component can be
constructed, fitted, cloned, tuned and composed in pipelines the same way regardless of task or backend.

## ADDED Requirements

### Requirement: Components follow scikit-learn estimator conventions
Every component SHALL accept only keyword hyperparameters in its constructor, store them unchanged as attributes, defer all validation to `fit`, and support `get_params`, `set_params` and `sklearn.base.clone`.

#### Scenario: Cloning preserves parameters
- **WHEN** a component constructed with non-default parameters is passed to `sklearn.base.clone`
- **THEN** the clone has equal `get_params()` output and is unfitted

#### Scenario: Invalid parameter is reported at fit time
- **WHEN** a component is constructed with an invalid parameter value
- **THEN** construction succeeds and `fit` raises `sklearn.utils._param_validation.InvalidParameterError` naming the parameter

### Requirement: Components must be fitted before use
Every component SHALL require `fit(X, y=None)` before `predict`, `predict_proba` or `transform`. Components without a learnable state SHALL still validate parameters and resolve their backend in `fit`.

#### Scenario: Predict before fit
- **WHEN** `predict` or `transform` is called on an unfitted component
- **THEN** `sklearn.exceptions.NotFittedError` is raised

#### Scenario: Stateless fit ignores data
- **WHEN** a generator component is fitted with `X=None` or any valid `X`
- **THEN** `fit` returns the component itself and subsequent `predict` works

### Requirement: Generator components expose predict and transform
Components whose output is generated text or media SHALL implement `predict(X)` and `transform(X)`, returning identical results, so they can be used as the final step or an intermediate step of a `Pipeline`.

#### Scenario: Converter inside a pipeline
- **WHEN** a pipeline of `SpeechTranscriber` followed by `TextClassifier` is fitted on audio inputs and labels and then predicts
- **THEN** the transcriber's `transform` output feeds the classifier and the pipeline returns one label per audio input

#### Scenario: Predict equals transform
- **WHEN** `predict(X)` and `transform(X)` are called on a fitted generator with a deterministic backend
- **THEN** both return equal arrays

### Requirement: Classifier components expose predict and predict_proba
Classifier components SHALL implement `fit(X, y)`, `predict(X)` and `predict_proba(X)`, expose `classes_` after fitting, support binary, multiclass and multilabel targets, and be scorable with scikit-learn metrics.

#### Scenario: Classes learned from targets
- **WHEN** a classifier is fitted with string labels `["spam", "ham", "spam"]`
- **THEN** `classes_` equals `["ham", "spam"]` and `predict` returns values from `classes_`

#### Scenario: Probabilities shape
- **WHEN** `predict_proba` is called on `n` samples for a classifier with `k` classes
- **THEN** it returns an array of shape `(n, k)` whose rows sum to 1

#### Scenario: Labels without training data
- **WHEN** a classifier is constructed with `labels=[...]` and fitted with `y=None`
- **THEN** it performs zero-shot classification over those labels and sets `classes_` to them

### Requirement: Classifiers support few-shot examples
Classifier components SHALL accept a `k_shot` parameter selecting how many fitted samples are passed to the backend as examples: `None` or `0` for zero-shot, an integer for a random selection controlled by `random_state`, or an array of indices.

#### Scenario: Reproducible example selection
- **WHEN** a classifier is fitted twice with the same `k_shot=4` and `random_state=0`
- **THEN** the same four examples are selected both times

### Requirement: Embedders expose transform
Embedding components SHALL implement `transform(X)` and `fit_transform(X)` returning a float array of shape `(n_samples, n_dimensions)`, and expose `get_feature_names_out`.

#### Scenario: Embeddings feed a scikit-learn model
- **WHEN** a pipeline of `TextEmbedder` followed by `LogisticRegression` is fitted on texts and labels
- **THEN** fitting succeeds and `predict` returns one label per text

### Requirement: Output order and length match input
Every component SHALL return exactly one output per input sample, in input order, regardless of internal batching or concurrency.

#### Scenario: Batched concurrent calls keep order
- **WHEN** 10 inputs are processed with `batch_size=3`
- **THEN** output `i` corresponds to input `i` for all `i`

### Requirement: Per-sample failure handling
Every component SHALL accept `on_error` with values `"raise"` (default) and `"ignore"`. With `"ignore"`, a sample whose backend call fails yields `None` (or a row of NaN for numeric outputs), and every such failure is reported in one warning naming each failed index and its error.

#### Scenario: Ignored failure
- **WHEN** one of three inputs fails at the backend and `on_error="ignore"`
- **THEN** the result has three entries, the failed one is `None` or NaN, and a warning names its index

#### Scenario: Raised failure
- **WHEN** a backend call fails and `on_error="raise"`
- **THEN** the exception propagates with the failing sample index in its message

### Requirement: Reproducibility control
Every component SHALL accept `random_state` and pass a derived seed to backends that support seeding. The documentation of each backend SHALL state whether its outputs are deterministic.

#### Scenario: Seeded generation
- **WHEN** `ImageGenerator` runs twice with `random_state=0` on a backend that supports seeding
- **THEN** both runs request generation with the same seed
