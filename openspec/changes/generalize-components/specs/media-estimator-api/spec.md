# Spec Delta

## ADDED Requirements

### Requirement: Records as samples
A sample MAY be a record, a dict of named values of any input type, wherever a component takes `input='record'`, and components that return records SHALL return dicts with stable keys.

#### Scenario: Records between steps
- **WHEN** a step that returns records is followed by a step with `input='record'`
- **THEN** the second step receives every record unchanged

### Requirement: List-valued samples
Components that split samples SHALL return one list per sample, and only `ForEach`, `Select` and `Composer` SHALL accept list-valued samples.

#### Scenario: List reaches a component
- **WHEN** a list-valued sample reaches a component other than `ForEach`, `Select` or `Composer`
- **THEN** `InvalidSampleError` names the index and suggests wrapping the step in `ForEach`

## MODIFIED Requirements

### Requirement: Generator components expose predict and transform
Components whose output is generated text or media SHALL implement `predict(X)` and `transform(X)`, returning identical results, so they can be used as the final step or an intermediate step of a `Pipeline`.

#### Scenario: Converter inside a pipeline
- **WHEN** a pipeline of `Transcriber` followed by `Classifier` is fitted on audio inputs and labels and then predicts
- **THEN** the transcriber's `transform` output feeds the classifier and the pipeline returns one label per audio input

#### Scenario: Predict equals transform
- **WHEN** `predict(X)` and `transform(X)` are called on a fitted generator with a deterministic backend
- **THEN** both return equal arrays

### Requirement: Embedders expose transform
Embedding components SHALL implement `transform(X)` and `fit_transform(X)` returning a float array of shape `(n_samples, n_dimensions)`, and expose `get_feature_names_out`.

#### Scenario: Embeddings feed a scikit-learn model
- **WHEN** a pipeline of `Embedder` followed by `LogisticRegression` is fitted on texts and labels
- **THEN** fitting succeeds and `predict` returns one label per text

### Requirement: Fit learns from the data it is given
Every component SHALL keep the signature `fit(X, y=None)`, derive fitted state from the data passed to it, and use that state in `predict` or `transform`. For a supervised generator, `y` SHALL hold its own targets, one per sample of `X`, in the type of its output.

#### Scenario: Fitted state changes predictions
- **WHEN** a component is fitted once without data and once with data, on the deterministic `dummy` backend
- **THEN** the backend receives different requests for the same input

#### Scenario: References scored as targets
- **WHEN** `cross_val_score` runs on a `TextGenerator(instructions='Summarize the text.')` with texts, reference summaries and a scorer that compares summaries
- **THEN** every fold learns from its training references and is scored against its test references

### Requirement: Generators ignore y in fit_transform
A generator SHALL learn from `y` in `fit` and SHALL ignore `y` in `fit_transform`, because pipelines and feature unions fit intermediate steps through `fit_transform` and pass them the final target. A `use_y` parameter SHALL override this rule: `'auto'` (default) follows it, `True` learns from `y` in both methods, and `False` never does.

#### Scenario: Pipeline target is not read as references
- **WHEN** a pipeline of `Transcriber` and `Classifier` is fitted on recordings and class labels
- **THEN** the transcriber learns no vocabulary from the labels and the classifier learns from them

#### Scenario: Supervised final step
- **WHEN** a pipeline of `Transcriber` and `TextGenerator(instructions='Summarize the text.')` is fitted on recordings and reference summaries
- **THEN** the text generator learns from the references and the transcriber does not

#### Scenario: Override
- **WHEN** a transcriber with `use_y=True` is an intermediate pipeline step fitted with reference transcripts
- **THEN** it learns its vocabulary from them

#### Scenario: Mismatched targets
- **WHEN** a supervised generator is fitted with a `y` whose length differs from `X`
- **THEN** `ValueError` is raised naming both lengths

### Requirement: Fine-tuning adaptation
Components that support fine-tuning SHALL accept `adaptation` with values `'examples'` (default) and `'finetune'`. With `'finetune'`, `fit` SHALL train a model on all of its data, store the model id in `model_`, and use it at prediction time instead of examples.

#### Scenario: Fine-tuned model is used
- **WHEN** a `Classifier(adaptation='finetune')` is fitted on a backend that supports fine-tuning
- **THEN** `model_` holds the new model id, and `predict` sends requests to that model with no examples

#### Scenario: Unsupported backend
- **WHEN** `adaptation='finetune'` is set on a backend that cannot fine-tune
- **THEN** `UnsupportedBackendError` is raised at `fit`, listing the backends that can

#### Scenario: Too little data
- **WHEN** `adaptation='finetune'` is fitted on fewer samples than the backend's minimum
- **THEN** `ValueError` is raised at `fit`, stating the minimum
