# Spec Delta

## Purpose

Extracts structured records from any input type following a declared set of fields, including nested records and
lists, so storyboards, invoices, attributes and action items are one component configured by its fields.

## ADDED Requirements

### Requirement: Extractor
The library SHALL provide `Extractor` in `skai.extraction`, which extracts the required `fields` from every sample of the input type selected by `input` (`'text'` by default, `'image'`, `'audio'`, `'video'` or `'record'`). `predict` SHALL return one dict per sample whose keys are exactly the field names, with `None` for absent fields.

#### Scenario: Fields from text and images
- **WHEN** `Extractor(fields={'name': 'person name', 'city': 'city'})` and `Extractor(input='image', fields={'color': 'main color'})` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns one dict per sample with exactly its field names

### Requirement: Nested and list fields
A field SHALL be declared as a description string, as a dict of subfields for a nested record, or as a one-element list holding a description or a dict for a list of values or records. Extracted lists SHALL be returned as Python lists.

#### Scenario: Storyboard
- **WHEN** `Extractor(fields={'scenes': [{'narration': 'spoken text', 'visual': 'what is shown', 'seconds': 'length'}]})` predicts on a brief
- **THEN** `scenes` is a list of dicts, each with exactly the keys `narration`, `visual` and `seconds`

### Requirement: Extracted fields as columns
`Extractor.transform` SHALL return a two-dimensional object array with one column per top-level field, in the order of `fields`, and `get_feature_names_out` SHALL return the top-level field names.

#### Scenario: DataFrame output
- **WHEN** an `Extractor` with `set_output(transform='pandas')` transforms two texts
- **THEN** it returns a DataFrame with one column per top-level field

### Requirement: Extractor learns from reference records
`Extractor.fit(X, y)` with reference records as `y` SHALL pass `k_shot` sample and record pairs to the backend as examples.

#### Scenario: Example records
- **WHEN** it is fitted with one text and its record
- **THEN** `examples_` holds that pair
