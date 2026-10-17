# Spec Delta

## ADDED Requirements

### Requirement: TextExtractor
The library SHALL provide `TextExtractor` in `skai.text`, a generator that extracts structured fields from every text. Its required `fields` parameter maps each field name to a description, and `predict` returns one dict per text whose keys are exactly the field names, with `None` for a field the text does not contain.

#### Scenario: Extract fields
- **WHEN** `TextExtractor(fields={'name': 'person name', 'city': 'city of residence'}, backend='dummy').fit().predict(['Ada lives in London.'])` is called
- **THEN** it returns one dict with exactly the keys `name` and `city`

#### Scenario: Missing fields
- **WHEN** it is fitted without `fields`
- **THEN** `InvalidParameterError` is raised

### Requirement: TextExtractor returns feature columns
`TextExtractor.transform` SHALL return a two-dimensional object array of shape `(n_texts, n_fields)` in the order of `fields`, and `get_feature_names_out` SHALL return the field names, so extracted fields can feed a `ColumnTransformer` or a DataFrame.

#### Scenario: Fields as columns
- **WHEN** `TextExtractor(fields={'name': '...', 'city': '...'}).set_output(transform='pandas').fit_transform(texts)` is called
- **THEN** it returns a DataFrame with the columns `name` and `city`
