# Spec Delta

## ADDED Requirements

### Requirement: Transcriber learns a vocabulary
`SpeechTranscriber.fit(X, y)`, with reference transcripts as `y`, SHALL learn `vocabulary_`, the domain terms of the references: tokens of two or more characters that contain a digit, have an upper-case letter after their first character, or start with an upper-case letter mid-sentence, ordered by frequency and capped at `max_vocabulary` terms. The vocabulary SHALL be sent to backends as context for spelling.

#### Scenario: Domain terms
- **WHEN** it is fitted with transcripts that mention `Kubernetes` and `gRPC`
- **THEN** both are in `vocabulary_` and every transcription request carries them

#### Scenario: Backend without context
- **WHEN** the backend cannot take context
- **THEN** a warning at `fit` states that the vocabulary is not used
