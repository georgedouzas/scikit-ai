# Spec Delta

## Purpose

Components that turn speech in media samples into text, importable from `skai.transcription`.

## ADDED Requirements

### Requirement: SpeechTranscriber
The library SHALL provide `SpeechTranscriber`, a generator returning one transcript per audio sample, with optional `language` (auto-detected when `None`).

#### Scenario: Plain transcripts
- **WHEN** `SpeechTranscriber(backend="dummy").fit().predict([a1, a2])` is called
- **THEN** it returns an object array of two transcript strings

### Requirement: Timestamped transcripts
`SpeechTranscriber` SHALL accept `timestamps` in `{None, "segment", "word"}`. When set, each output SHALL be a list of records with `start`, `end` (seconds) and `text`, and `transform` SHALL return the joined plain text so pipelines keep receiving strings.

#### Scenario: Segment timestamps
- **WHEN** `timestamps="segment"` and `predict` is called on one audio
- **THEN** the output holds a list of records each with `start <= end` and non-empty `text`

#### Scenario: Unsupported granularity
- **WHEN** `timestamps="word"` is requested on a backend without word timing
- **THEN** `ValueError` is raised at `fit`
