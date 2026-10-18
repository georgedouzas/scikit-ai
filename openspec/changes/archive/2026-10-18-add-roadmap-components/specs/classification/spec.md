# Spec Delta

## ADDED Requirements

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
