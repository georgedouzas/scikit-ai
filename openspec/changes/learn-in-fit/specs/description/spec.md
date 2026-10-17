# Spec Delta

## ADDED Requirements

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
