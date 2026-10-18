# Spec Delta

## Purpose

Extracts structured records from texts, images, audio, videos and multimodal samples following a declared set of
fields, including nested records and lists, so storyboards, invoices, attributes and action items are extractors
configured by their fields.

## ADDED Requirements

### Requirement: Extractors for every input
The library SHALL provide `TextExtractor`, `ImageExtractor`, `AudioExtractor`, `VideoExtractor` (with `n_frames`) and `MultimodalExtractor` in `skai.extraction`. Each SHALL extract the required `fields` from every sample, with optional `instructions`, and `predict` SHALL return one dict per sample whose keys are exactly the field names, with `None` for absent fields.

#### Scenario: Fields from text and images
- **WHEN** `TextExtractor(fields={'name': 'person name', 'city': 'city'})` and `ImageExtractor(fields={'color': 'main color'})` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns one dict per sample with exactly its field names

#### Scenario: Missing fields
- **WHEN** an extractor is fitted without `fields`
- **THEN** `InvalidParameterError` is raised

### Requirement: Nested and list fields
A field SHALL be declared as a description string, as a dict of subfields for a nested record, or as a one-element list holding a description or a dict for a list of values or records. Extracted lists SHALL be Python lists.

#### Scenario: Storyboard
- **WHEN** `TextExtractor(fields={'scenes': [{'narration': 'spoken text', 'visual': 'what is shown', 'seconds': 'length'}]})` predicts on a brief
- **THEN** `scenes` is a list of dicts, each with exactly the keys `narration`, `visual` and `seconds`

### Requirement: Extracted fields as columns
`transform` SHALL return a two-dimensional object array with one column per top-level field, in the order of `fields`, and `get_feature_names_out` SHALL return the top-level field names.

#### Scenario: DataFrame output
- **WHEN** a `TextExtractor` with `set_output(transform='pandas')` transforms two texts
- **THEN** it returns a DataFrame with one column per top-level field

### Requirement: Extractors learn from reference records
`fit(X, y)` with reference records as `y` SHALL pass `k_shot` sample and record pairs to the backend as examples.

#### Scenario: Example records
- **WHEN** a `TextExtractor` is fitted with one text and its record
- **THEN** `examples_` holds that pair
