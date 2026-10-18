# classification Specification

## Purpose
Components that assign labels to samples of any input type, zero-shot or few-shot, importable from `skai.classification`.

## Requirements

### Requirement: TextClassifier
The library SHALL provide `TextClassifier`, a classifier that assigns texts to labels using zero-shot or few-shot prompting, with optional `instructions` that describe the labeling criteria.

#### Scenario: Few-shot binary classification
- **WHEN** `TextClassifier(k_shot=4)` is fitted on labeled reviews and predicts on new reviews
- **THEN** it returns one label from `classes_` per review

#### Scenario: Multilabel targets
- **WHEN** it is fitted with a binary indicator matrix `y` of shape `(n, k)`
- **THEN** `predict` returns an indicator matrix of shape `(m, k)` for `m` inputs

### Requirement: ImageClassifier
The library SHALL provide `ImageClassifier`, a classifier assigning images to labels with zero-shot (via `labels`) or few-shot (via fitted examples and `k_shot`) prediction, supporting binary, multiclass and multilabel targets.

#### Scenario: Zero-shot labels
- **WHEN** `ImageClassifier(labels=["cat", "dog"], backend="dummy").fit(None).predict(images)` is called
- **THEN** every prediction is `"cat"` or `"dog"` and `predict_proba` has two columns

#### Scenario: Labels conflict with targets
- **WHEN** it is fitted with both `labels` and a `y` containing values not in `labels`
- **THEN** `ValueError` is raised listing the unknown values

### Requirement: Classifiers can be fine-tuned
`TextClassifier` and `ImageClassifier` SHALL support `adaptation='finetune'`, training a model that answers with one label per sample from the training samples and their targets.

#### Scenario: Fine-tuned text classifier
- **WHEN** `TextClassifier(adaptation='finetune', backend='dummy')` is fitted on 20 labeled texts
- **THEN** `model_` is set and `predict` returns labels from `classes_`

### Requirement: AudioClassifier
The library SHALL provide `AudioClassifier` in `skai.classification`, a classifier that assigns labels to audio samples, such as sound events, music genres or speaker emotion, with zero-shot prediction from `labels` or few-shot prediction from training audio and `k_shot`.

#### Scenario: Zero-shot sound events
- **WHEN** `AudioClassifier(labels=['dog bark', 'siren'], backend='dummy').fit()` predicts on two audio samples
- **THEN** every prediction is `'dog bark'` or `'siren'` and `predict_proba` has shape `(2, 2)`

#### Scenario: Few-shot audio
- **WHEN** it is fitted on six labeled audio samples with `k_shot=2`
- **THEN** `examples_` holds two of the training samples with their labels

### Requirement: VideoClassifier
The library SHALL provide `VideoClassifier` in `skai.classification`, a classifier that assigns labels to videos with zero-shot or few-shot prediction, analyzing `n_frames` frames sampled from each video.

#### Scenario: Zero-shot video labels
- **WHEN** `VideoClassifier(labels=['cooking', 'sports'], n_frames=4, backend='dummy').fit()` predicts on three videos
- **THEN** it returns three labels from `labels`

#### Scenario: Multilabel videos
- **WHEN** it is fitted with a binary indicator matrix of shape `(n, 3)`
- **THEN** `predict` returns an indicator matrix with three columns
