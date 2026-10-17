# Spec Delta

## ADDED Requirements

### Requirement: Classifiers can be fine-tuned
`TextClassifier` and `ImageClassifier` SHALL support `adaptation='finetune'`, training a model that answers with one label per sample from the training samples and their targets.

#### Scenario: Fine-tuned text classifier
- **WHEN** `TextClassifier(adaptation='finetune', backend='dummy')` is fitted on 20 labeled texts
- **THEN** `model_` is set and `predict` returns labels from `classes_`
