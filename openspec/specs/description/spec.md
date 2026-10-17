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

### Requirement: Descriptor learns from reference descriptions
`ImageDescriptor.fit(X, y)`, with reference descriptions as `y`, SHALL select `k_shot` image and description pairs as `examples_`, sent with every request so the descriptions follow the style, length and vocabulary of the references.

#### Scenario: Style examples
- **WHEN** it is fitted with three images, three descriptions and `k_shot=2`
- **THEN** every description request carries two example pairs

### Requirement: Descriptor can be fine-tuned
`ImageDescriptor` SHALL support `adaptation='finetune'` from images and their descriptions.

#### Scenario: Fine-tuned descriptor
- **WHEN** `ImageDescriptor(adaptation='finetune', backend='dummy')` is fitted with 20 images and their descriptions
- **THEN** `model_` is set
