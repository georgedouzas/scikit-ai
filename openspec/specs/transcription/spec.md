# transcription Specification

## Purpose
Components that turn speech in media samples into text, importable from `skai.transcription`.

## Requirements

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

### Requirement: Transcriber learns a vocabulary
`SpeechTranscriber.fit(X, y)`, with reference transcripts as `y`, SHALL learn `vocabulary_`, the domain terms of the references: tokens of two or more characters that contain a digit, have an upper-case letter after their first character, or start with an upper-case letter mid-sentence, ordered by frequency and capped at `max_vocabulary` terms. The vocabulary SHALL be sent to backends as context for spelling.

#### Scenario: Domain terms
- **WHEN** it is fitted with transcripts that mention `Kubernetes` and `gRPC`
- **THEN** both are in `vocabulary_` and every transcription request carries them

#### Scenario: Backend without context
- **WHEN** the backend cannot take context
- **THEN** a warning at `fit` states that the vocabulary is not used
