# Spec Delta

## ADDED Requirements

### Requirement: Classifier
The library SHALL provide `Classifier` in `skai.classification`, one classifier for every input type selected by its `input` parameter (`'text'` by default, `'image'`, `'audio'`, `'video'` or `'record'`). It SHALL keep the labels, instructions, few-shot and fine-tuning behavior of the 0.2 classifiers, and analyze videos from `n_frames` sampled frames.

#### Scenario: Text by default
- **WHEN** `Classifier(labels=['positive', 'negative'], backend='dummy').fit().predict(['I love it.'])` is called
- **THEN** it returns one label from `labels`

#### Scenario: Other input types
- **WHEN** `Classifier(input='image', labels=['cat', 'dog'])`, `Classifier(input='audio', labels=['speech', 'music'])` and `Classifier(input='video', labels=['cooking', 'sports'], n_frames=4)` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns one label from its `labels` per sample, and `predict_proba` has one column per label

#### Scenario: Wrong input type
- **WHEN** `Classifier(input='image')` receives a sample that cannot be read as an image
- **THEN** `MediaReadError` is raised naming the index

## MODIFIED Requirements

### Requirement: Classifiers can be fine-tuned
`Classifier` SHALL support `adaptation='finetune'` for every input type its backend can fine-tune on, training a model that answers with one label per sample from the training samples and their targets.

#### Scenario: Fine-tuned text classifier
- **WHEN** `Classifier(adaptation='finetune', backend='dummy')` is fitted on 20 labeled texts
- **THEN** `model_` is set and `predict` returns labels from `classes_`

## REMOVED Requirements

### Requirement: TextClassifier
**Reason**: Classifiers are one class whose input type is a parameter.
**Migration**: Use `Classifier()`, which takes text by default.

### Requirement: ImageClassifier
**Reason**: Classifiers are one class whose input type is a parameter.
**Migration**: Use `Classifier(input='image')`.

### Requirement: AudioClassifier
**Reason**: Classifiers are one class whose input type is a parameter.
**Migration**: Use `Classifier(input='audio')`.

### Requirement: VideoClassifier
**Reason**: Classifiers are one class whose input type is a parameter.
**Migration**: Use `Classifier(input='video', n_frames=...)`.
