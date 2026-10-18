# Spec Delta

## ADDED Requirements

### Requirement: Transcriber
The library SHALL provide `Transcriber` in `skai.transcription`, which turns the speech of audio or the audio track of video (`input='audio'` by default, or `'video'`), or the written text of images (`input='image'`), into text. A video without an audio track SHALL give an empty transcript without calling the backend.

#### Scenario: Speech, video and images
- **WHEN** `Transcriber()`, `Transcriber(input='video')` and `Transcriber(input='image')` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns one string per sample

#### Scenario: Silent video
- **WHEN** `Transcriber(input='video')` receives a video without an audio track
- **THEN** its transcript is an empty string

### Requirement: Speaker labels
With `speakers=True`, audio and video transcripts SHALL be timed records that also carry a `speaker` label, consistent across the sample, on backends that support speaker labels.

#### Scenario: Two speakers
- **WHEN** `Transcriber(speakers=True, backend='dummy')` predicts on a recording
- **THEN** every record has `start`, `end`, `text` and `speaker`

#### Scenario: Unsupported speaker labels
- **WHEN** `speakers=True` is set on a backend without speaker labels
- **THEN** `UnsupportedBackendError` is raised at `fit`

### Requirement: Document layout
With `input='image'` and `layout=True`, the transcript SHALL keep the reading order and structure of the page as Markdown headings, lists and tables.

#### Scenario: Scanned page
- **WHEN** `Transcriber(input='image', layout=True, backend='dummy')` predicts on a page image
- **THEN** it returns one Markdown string per page

## MODIFIED Requirements

### Requirement: Timestamped transcripts
`Transcriber` SHALL accept `timestamps` in `{None, "segment", "word"}` for audio and video. When set, each output SHALL be a list of records with `start`, `end` (seconds) and `text`, and `transform` SHALL return the joined plain text so pipelines keep receiving strings.

#### Scenario: Segment timestamps
- **WHEN** `timestamps="segment"` and `predict` is called on one audio
- **THEN** the output holds a list of records each with `start <= end` and non-empty `text`

#### Scenario: Unsupported granularity
- **WHEN** `timestamps="word"` is requested on a backend without word timing
- **THEN** `ValueError` is raised at `fit`

### Requirement: Transcriber learns a vocabulary
`Transcriber.fit(X, y)`, with reference transcripts as `y`, SHALL learn `vocabulary_`, the domain terms of the references: tokens of two or more characters that contain a digit, have an upper-case letter after their first character, or start with an upper-case letter mid-sentence, ordered by frequency and capped at `max_vocabulary` terms. The vocabulary SHALL be sent to backends as context for spelling.

#### Scenario: Domain terms
- **WHEN** it is fitted with transcripts that mention `Kubernetes` and `gRPC`
- **THEN** both are in `vocabulary_` and every transcription request carries them

#### Scenario: Backend without context
- **WHEN** the backend cannot take context
- **THEN** a warning at `fit` states that the vocabulary is not used

## REMOVED Requirements

### Requirement: SpeechTranscriber
**Reason**: Transcription is one class whose input type is a parameter.
**Migration**: Use `Transcriber()`.

### Requirement: VideoTranscriber
**Reason**: Transcription is one class whose input type is a parameter.
**Migration**: Use `Transcriber(input='video')`.
