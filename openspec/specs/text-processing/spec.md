# text-processing Specification

## Purpose
Components that turn texts into other texts, such as summaries and translations, importable from `skai.text`.

## Requirements

### Requirement: TextSummarizer
The library SHALL provide `TextSummarizer`, a generator returning one summary per text, controlled by `max_length` (approximate maximum words) and an optional free-text `style`.

#### Scenario: Length bound
- **WHEN** `TextSummarizer(max_length=50, backend="dummy")` summarizes a long text
- **THEN** the summary has at most 50 words

### Requirement: TextTranslator
The library SHALL provide `TextTranslator`, a generator translating each text into `target_language` (required), with optional `source_language` that defaults to auto-detection.

#### Scenario: Missing target language
- **WHEN** `TextTranslator()` is fitted without `target_language`
- **THEN** `InvalidParameterError` is raised

### Requirement: Summarizer learns from reference summaries
`TextSummarizer.fit(X, y)`, with reference summaries as `y`, SHALL select `k_shot` text and summary pairs as `examples_`, and, when `max_length` is `None`, SHALL set `max_length_` to the 90th percentile of the word counts of the references.

#### Scenario: Learned length
- **WHEN** `TextSummarizer(max_length=None, backend='dummy')` is fitted with references of 8 to 12 words
- **THEN** `max_length_` is between 8 and 12 and predictions have at most `max_length_` words

#### Scenario: Pairs are sent as examples
- **WHEN** it is fitted with reference summaries and `k_shot=2`
- **THEN** every summary request carries two example pairs

### Requirement: Translator learns from reference translations
`TextTranslator.fit(X, y)`, with reference translations as `y`, SHALL select `k_shot` text and translation pairs as `examples_`, and SHALL learn a `glossary_` of terms that appear in at least two references, which is sent with every request so terms are translated consistently.

#### Scenario: Glossary from references
- **WHEN** it is fitted on pairs where a product name keeps the same translation in two references
- **THEN** `glossary_` maps that name to its translation

### Requirement: Text components can be fine-tuned
`TextSummarizer` and `TextTranslator` SHALL support `adaptation='finetune'` from texts and their references.

#### Scenario: Fine-tuned summarizer
- **WHEN** `TextSummarizer(adaptation='finetune', backend='dummy')` is fitted with 20 texts and their summaries
- **THEN** `model_` is set and predictions are produced by that model

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
