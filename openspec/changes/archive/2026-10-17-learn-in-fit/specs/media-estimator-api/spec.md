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
Every component SHALL keep the signature `fit(X, y=None)`, derive fitted state from the data passed to it, and use that state in `predict` or `transform`. For a supervised generator, `y` SHALL hold its own targets, one per sample of `X`, in the type of its output.

#### Scenario: Fitted state changes predictions
- **WHEN** a component is fitted once without data and once with data, on the deterministic `dummy` backend
- **THEN** the backend receives different requests for the same input

#### Scenario: References scored as targets
- **WHEN** `cross_val_score` runs on a `TextSummarizer` with texts, reference summaries and a scorer that compares summaries
- **THEN** every fold learns from its training references and is scored against its test references

### Requirement: Generators ignore y in fit_transform
A generator SHALL learn from `y` in `fit` and SHALL ignore `y` in `fit_transform`, because pipelines and feature unions fit intermediate steps through `fit_transform` and pass them the final target. A `use_y` parameter SHALL override this rule: `'auto'` (default) follows it, `True` learns from `y` in both methods, and `False` never does.

#### Scenario: Pipeline target is not read as references
- **WHEN** a pipeline of `SpeechTranscriber` and `TextClassifier` is fitted on recordings and class labels
- **THEN** the transcriber learns no vocabulary from the labels and the classifier learns from them

#### Scenario: Supervised final step
- **WHEN** a pipeline of `SpeechTranscriber` and `TextSummarizer` is fitted on recordings and reference summaries
- **THEN** the summarizer learns from the references and the transcriber does not

#### Scenario: Override
- **WHEN** a transcriber with `use_y=True` is an intermediate pipeline step fitted with reference transcripts
- **THEN** it learns its vocabulary from them

#### Scenario: Mismatched targets
- **WHEN** a supervised generator is fitted with a `y` whose length differs from `X`
- **THEN** `ValueError` is raised naming both lengths

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
