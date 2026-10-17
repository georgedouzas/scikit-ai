# image-components Specification

## Purpose
Task components whose primary input is an image: natural-language description and zero/few-shot
classification, following the shared media estimator contract.

## Requirements

### Requirement: ImageDescriptor
The library SHALL provide `ImageDescriptor`, a generator returning one natural-language description per image, with `detail` in `{"brief", "detailed"}` and optional `instructions` steering what to describe.

#### Scenario: Describe images
- **WHEN** `ImageDescriptor(backend="dummy").fit().predict([img1, img2])` is called
- **THEN** it returns an object array of two description strings

#### Scenario: Descriptions feed a text model
- **WHEN** a pipeline of `ImageDescriptor` followed by `TextClassifier` is fitted on images and labels
- **THEN** it predicts one label per image

### Requirement: ImageClassifier
The library SHALL provide `ImageClassifier`, a classifier assigning images to labels with zero-shot (via `labels`) or few-shot (via fitted examples and `k_shot`) prediction, supporting binary, multiclass and multilabel targets.

#### Scenario: Zero-shot labels
- **WHEN** `ImageClassifier(labels=["cat", "dog"], backend="dummy").fit(None).predict(images)` is called
- **THEN** every prediction is `"cat"` or `"dog"` and `predict_proba` has two columns

#### Scenario: Labels conflict with targets
- **WHEN** it is fitted with both `labels` and a `y` containing values not in `labels`
- **THEN** `ValueError` is raised listing the unknown values
