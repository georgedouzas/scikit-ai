# description Specification

## Purpose
Components that describe media samples in natural language, importable from `skai.description`.

## Requirements

### Requirement: ImageDescriptor
The library SHALL provide `ImageDescriptor`, a generator returning one natural-language description per image, with `detail` in `{"brief", "detailed"}` and optional `instructions` steering what to describe.

#### Scenario: Describe images
- **WHEN** `ImageDescriptor(backend="dummy").fit().predict([img1, img2])` is called
- **THEN** it returns an object array of two description strings

#### Scenario: Descriptions feed a text model
- **WHEN** a pipeline of `ImageDescriptor` followed by `TextClassifier` is fitted on images and labels
- **THEN** it predicts one label per image
