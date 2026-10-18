# Spec Delta

## ADDED Requirements

### Requirement: Multimodal samples
Wherever a class name starts with `Multimodal`, a sample SHALL be a dict of named texts, images, audio, videos or transcripts, or a DataFrame row of them, and every operation that needs specific keys SHALL raise `InvalidSampleError` naming the missing key and the sample index.

#### Scenario: Missing key
- **WHEN** a multimodal generator whose operation needs `video` and `audio` receives a sample without `audio`
- **THEN** `InvalidSampleError` names `audio` and the index

#### Scenario: DataFrame rows
- **WHEN** a multimodal component receives a DataFrame with the columns `image` and `caption`
- **THEN** every row is read as one sample with those keys

### Requirement: List-valued samples
Segmenters SHALL return one list per sample. Only `ForEach`, `TopK` and the list operations of generators (`collage` and `concatenate`) SHALL accept list-valued samples.

#### Scenario: List reaches a component
- **WHEN** a list-valued sample reaches any other component
- **THEN** `InvalidSampleError` names the index and suggests wrapping the step in `ForEach`

## MODIFIED Requirements

### Requirement: Generator components expose predict and transform
Components whose output is generated text or media SHALL implement `predict(X)` and `transform(X)`, returning identical results, so they can be used as the final step or an intermediate step of a `Pipeline`.

#### Scenario: Converter inside a pipeline
- **WHEN** a pipeline of `AudioToTextGenerator` followed by `TextClassifier` is fitted on audio inputs and labels and then predicts
- **THEN** the transcriber's `transform` output feeds the classifier and the pipeline returns one label per audio input

#### Scenario: Predict equals transform
- **WHEN** `predict(X)` and `transform(X)` are called on a fitted generator with a deterministic backend
- **THEN** both return equal arrays

### Requirement: Reproducibility control
Every component SHALL accept `random_state` and pass a derived seed to backends that support seeding. The documentation of each backend SHALL state whether its outputs are deterministic.

#### Scenario: Seeded generation
- **WHEN** `TextToImageGenerator` runs twice with `random_state=0` on a backend that supports seeding
- **THEN** both runs request generation with the same seed

### Requirement: Fit learns from the data it is given
Every component SHALL keep the signature `fit(X, y=None)`, derive fitted state from the data passed to it, and use that state in `predict` or `transform`. For a supervised generator, `y` SHALL hold its own targets, one per sample of `X`, in the type of its output.

#### Scenario: Fitted state changes predictions
- **WHEN** a component is fitted once without data and once with data, on the deterministic `dummy` backend
- **THEN** the backend receives different requests for the same input

#### Scenario: References scored as targets
- **WHEN** `cross_val_score` runs on a `TextToTextGenerator(instructions='Summarize the text.')` with texts, reference summaries and a scorer that compares summaries
- **THEN** every fold learns from its training references and is scored against its test references

### Requirement: Generators ignore y in fit_transform
A generator SHALL learn from `y` in `fit` and SHALL ignore `y` in `fit_transform`, because pipelines and feature unions fit intermediate steps through `fit_transform` and pass them the final target. A `use_y` parameter SHALL override this rule: `'auto'` (default) follows it, `True` learns from `y` in both methods, and `False` never does.

#### Scenario: Pipeline target is not read as references
- **WHEN** a pipeline of `AudioToTextGenerator` and `TextClassifier` is fitted on recordings and class labels
- **THEN** the transcriber learns no vocabulary from the labels and the classifier learns from them

#### Scenario: Supervised final step
- **WHEN** a pipeline of `AudioToTextGenerator` and `TextToTextGenerator(instructions='Summarize the text.')` is fitted on recordings and reference summaries
- **THEN** the text generator learns from the references and the transcriber does not

#### Scenario: Override
- **WHEN** a transcriber with `use_y=True` is an intermediate pipeline step fitted with reference transcripts
- **THEN** it learns its vocabulary from them

#### Scenario: Mismatched targets
- **WHEN** a supervised generator is fitted with a `y` whose length differs from `X`
- **THEN** `ValueError` is raised naming both lengths
