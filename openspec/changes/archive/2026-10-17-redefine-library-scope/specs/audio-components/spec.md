# Spec Delta

## Purpose

Task components whose primary input is audio: speech transcription and talking-avatar video generation,
following the shared media estimator contract.

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

### Requirement: TalkingAvatarGenerator
The library SHALL provide `TalkingAvatarGenerator`, a generator that animates a fixed portrait `avatar` (any accepted image input, required) to speak each audio sample, returning one lip-synced video per sample with the audio track included.

#### Scenario: One video per audio
- **WHEN** `TalkingAvatarGenerator(avatar="me.png", backend="dummy").fit().predict(["hello.wav", "bye.wav"])` is called
- **THEN** it returns two videos whose durations match their audio inputs

#### Scenario: Missing avatar
- **WHEN** it is fitted without `avatar`
- **THEN** `InvalidParameterError` is raised

#### Scenario: Speech synthesized then animated
- **WHEN** a pipeline of `SpeechSynthesizer` followed by `TalkingAvatarGenerator` predicts on texts
- **THEN** it returns one video per text
