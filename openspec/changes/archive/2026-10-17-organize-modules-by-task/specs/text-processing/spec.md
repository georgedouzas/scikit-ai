# Spec Delta

## Purpose

Components that turn texts into other texts, such as summaries and translations, importable from `skai.text`.

## ADDED Requirements

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
