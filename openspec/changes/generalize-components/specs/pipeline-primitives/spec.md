# Spec Delta

## Purpose

Provides the scikit-learn meta-estimators that let pipelines pass several inputs per sample, work on every part of a
split sample, and keep the best parts, while every pipeline row stays one sample.

## ADDED Requirements

### Requirement: MediaUnion
The library SHALL provide `MediaUnion` in `skai.pipeline`, which runs named branches on the same samples and returns one multimodal sample, a dict keyed by branch name, per input sample. A branch SHALL be an estimator, a pipeline or `'passthrough'`, fitted through `fit_transform` like `FeatureUnion`, with parameters routed as `<branch>__<parameter>`.

#### Scenario: Video and translated speech side by side
- **WHEN** `MediaUnion([('video', 'passthrough'), ('audio', make_pipeline(VideoToTextGenerator(operation='transcribe'), TextToTextGenerator(instructions='Translate into Greek.'), TextToAudioGenerator()))])` transforms two videos on `dummy`
- **THEN** it returns two dicts, each with the original `video` and the new `audio`

### Requirement: Field
The library SHALL provide `Field` in `skai.pipeline`, a stateless transformer that returns one field of every multimodal sample or region, or a dict of several fields when given a list.

#### Scenario: One field
- **WHEN** `Field('visual')` transforms dicts with `visual` and `narration`
- **THEN** it returns the `visual` values in order

#### Scenario: Missing field
- **WHEN** a sample lacks the field
- **THEN** `KeyError` names the field and the sample index

### Requirement: ForEach
The library SHALL provide `ForEach` in `skai.pipeline`, which applies an estimator to every item of list-valued samples and returns lists of the same lengths. Items SHALL be processed in one batch across all samples, and `fit` SHALL fit the estimator on all items with `y=None`.

#### Scenario: Every scene
- **WHEN** `ForEach(make_pipeline(Field('visual'), TextToImageGenerator(size=(64, 64))))` transforms two storyboards of three and two scenes
- **THEN** it returns lists of three and two images, and the backend receives one batch of five requests

#### Scenario: Empty lists
- **WHEN** a sample is an empty list
- **THEN** its output is an empty list

### Requirement: TopK
The library SHALL provide `TopK` in `skai.pipeline`, which keeps the `k` items of list-valued samples with the highest scores, or those above `threshold`, in their original order. The score SHALL come from a callable on one item, or from a classifier's `predict_proba` column for `label`.

#### Scenario: Best moments
- **WHEN** `TopK(VideoClassifier(labels=['exciting', 'calm']), label='exciting', k=2)` transforms a list of five clips
- **THEN** it returns the two clips with the highest `exciting` probability, in their original order
