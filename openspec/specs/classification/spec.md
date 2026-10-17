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
