# Spec Delta

## MODIFIED Requirements

### Requirement: Components must be fitted before use
Every component SHALL require `fit` before `predict`, `predict_proba` or `transform`. A component fitted without data SHALL still validate its parameters and resolve its backend, and SHALL then predict zero-shot.

#### Scenario: Predict before fit
- **WHEN** `predict` or `transform` is called on an unfitted component
- **THEN** `sklearn.exceptions.NotFittedError` is raised

#### Scenario: Stateless fit ignores data
- **WHEN** a generator component is fitted with no samples and no references
- **THEN** `fit` returns the component itself and subsequent `predict` works zero-shot

### Requirement: Classifiers support few-shot examples
Classifier components SHALL accept a `k_shot` parameter selecting how many fitted samples are passed to the backend as examples: `None` for one example per class, `0` for zero-shot, an integer for a random selection controlled by `random_state`, or an array of indices.

#### Scenario: Reproducible example selection
- **WHEN** a classifier is fitted twice with the same `k_shot=4` and `random_state=0`
- **THEN** the same four examples are selected both times

#### Scenario: One example per class by default
- **WHEN** a classifier with `k_shot=None` is fitted on samples of three classes
- **THEN** `examples_` holds exactly one sample of each class

#### Scenario: Zero-shot with targets
- **WHEN** a classifier with `k_shot=0` is fitted with targets
- **THEN** `examples_` is empty and only `classes_` is learned

## ADDED Requirements

### Requirement: Fit learns from the data it is given
Every component SHALL derive fitted state from the samples and references passed to `fit` and use that state in `predict` or `transform`. Generators SHALL ignore `y`, as scikit-learn transformers do, because a pipeline passes its final target to every step.

#### Scenario: Fitted state changes predictions
- **WHEN** a component is fitted once without data and once with data, on the deterministic `dummy` backend
- **THEN** the backend receives different requests for the same input

#### Scenario: Generators ignore the target
- **WHEN** a generator is fitted with a non-`None` `y` and no references
- **THEN** its fitted state equals that of a fit without `y`

### Requirement: Generators take references through named fit parameters
Generators SHALL take reference data through named keyword parameters of `fit`, such as `summaries` or `transcripts`, never through `y`. These parameters SHALL support scikit-learn metadata routing, so a pipeline can send them to one step.

#### Scenario: Pipeline target is not read as references
- **WHEN** a pipeline of `SpeechTranscriber` and `TextClassifier` is fitted on recordings and class labels
- **THEN** the transcriber learns no vocabulary from the labels

#### Scenario: Routed references
- **WHEN** metadata routing is enabled, the transcriber requests `transcripts`, and the pipeline is fitted with `transcripts=references`
- **THEN** the transcriber learns its vocabulary from `references` and the classifier still learns from the labels

### Requirement: Fine-tuning adaptation
Components that support fine-tuning SHALL accept `adaptation` with values `'examples'` (default) and `'finetune'`. With `'finetune'`, `fit` SHALL train a model on all of its data, store the model id in `model_`, and use it at prediction time instead of examples.

#### Scenario: Fine-tuned model is used
- **WHEN** a `TextClassifier(adaptation='finetune')` is fitted on a backend that supports fine-tuning
- **THEN** `model_` holds the new model id, and `predict` sends requests to that model with no examples

#### Scenario: Unsupported backend
- **WHEN** `adaptation='finetune'` is set on a backend that cannot fine-tune
- **THEN** `UnsupportedBackendError` is raised at `fit`, listing the backends that can

#### Scenario: Too little data
- **WHEN** `adaptation='finetune'` is fitted on fewer samples than the backend's minimum
- **THEN** `ValueError` is raised at `fit`, stating the minimum
