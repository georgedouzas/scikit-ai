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
- **THEN** it returns lists of three and two images, and the inner pipeline transforms all five scenes in one call, which the generator splits only by its `batch_size`

#### Scenario: Empty lists
- **WHEN** a sample is an empty list
- **THEN** its output is an empty list

### Requirement: TopK
The library SHALL provide `TopK` in `skai.pipeline`, which keeps the `k` items of list-valued samples with the highest scores, or those above `threshold`, in their original order. The score SHALL come from a callable on one item, or from a classifier's `predict_proba` column for `label`.

#### Scenario: Best moments
- **WHEN** `TopK(VideoClassifier(labels=['exciting', 'calm']), label='exciting', k=2)` transforms a list of five clips
- **THEN** it returns the two clips with the highest `exciting` probability, in their original order

### Requirement: Retriever
The library SHALL provide `Retriever` in `skai.pipeline`, which indexes the items of samples with an item pipeline and answers queries with a query pipeline. `fit` SHALL run the item pipeline, which returns one vector or one list of records with vectors per sample, and `kneighbors(queries, n_neighbors)` SHALL return, for every query, the records of the nearest items with their `sample` index.

#### Scenario: Shots found by text
- **WHEN** a `Retriever` whose item pipeline embeds the shots of videos and whose query pipeline is a `TextEmbedder` is fitted on two videos
- **THEN** `kneighbors(['a goal'], n_neighbors=2)` returns two records per query, each with `sample`, `start` and `end`
