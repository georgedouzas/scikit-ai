# Spec Delta

## Purpose

Provides the scikit-learn meta-estimators that let pipelines pass several inputs per sample, work on every part of a
split sample, and keep the best parts, while every pipeline row stays one sample.

## ADDED Requirements

### Requirement: MediaUnion
The library SHALL provide `MediaUnion` in `skai.pipeline`, which runs named branches on the same samples and returns one record per sample keyed by branch name. A branch SHALL be an estimator, a pipeline or `'passthrough'`, and branches SHALL be fitted through `fit_transform` like `FeatureUnion`.

#### Scenario: Video and translated speech side by side
- **WHEN** `MediaUnion([('video', 'passthrough'), ('audio', make_pipeline(Transcriber(input='video'), TextGenerator(instructions='Translate into Greek.'), AudioGenerator()))])` transforms two videos on `dummy`
- **THEN** it returns two records, each with the original `video` and the new `audio`

### Requirement: Pick
The library SHALL provide `Pick` in `skai.pipeline`, a stateless transformer that returns one field of every record, or a record of several fields when given a list.

#### Scenario: Pick a field
- **WHEN** `Pick('visual')` transforms records with `visual` and `narration`
- **THEN** it returns the `visual` values in order

#### Scenario: Missing field
- **WHEN** a record lacks the picked field
- **THEN** `KeyError` names the field and the sample index

### Requirement: ForEach
The library SHALL provide `ForEach` in `skai.pipeline`, which applies an estimator to every item of list-valued samples and returns lists of the same lengths. Items SHALL be processed in one batch across all samples, and `fit` SHALL fit the estimator on all items with `y=None`.

#### Scenario: Every scene
- **WHEN** `ForEach(make_pipeline(Pick('visual'), ImageGenerator(size=(64, 64))))` transforms two storyboards of three and two scenes
- **THEN** it returns lists of three and two images, and the backend receives one batch of five requests

#### Scenario: Empty lists
- **WHEN** a sample is an empty list
- **THEN** its output is an empty list

### Requirement: Select
The library SHALL provide `Select` in `skai.pipeline`, which keeps the items of list-valued samples with the highest scores. The score SHALL come from a `scorer` that is a callable on one item, or a classifier whose `predict_proba` column for `label` is used, and `top_k` or `threshold` SHALL set how many items stay, in their original order.

#### Scenario: Best moments
- **WHEN** `Select(Classifier(input='video', labels=['exciting', 'calm']), label='exciting', top_k=2)` transforms a list of five clips
- **THEN** it returns the two clips with the highest `exciting` probability, in their original order
