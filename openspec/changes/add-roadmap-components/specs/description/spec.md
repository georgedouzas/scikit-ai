# Spec Delta

## ADDED Requirements

### Requirement: AudioDescriptor
The library SHALL provide `AudioDescriptor` in `skai.description`, a generator returning one natural-language description per audio sample, covering sounds, music and speech, with `detail` in `{"brief", "detailed"}` and optional `instructions`.

#### Scenario: Describe sounds
- **WHEN** `AudioDescriptor(backend='dummy').fit().predict([a1, a2])` is called
- **THEN** it returns an object array of two description strings

#### Scenario: Descriptions feed a text component
- **WHEN** a pipeline of `AudioDescriptor` followed by `TextClassifier` is fitted on audio and labels
- **THEN** it predicts one label per audio sample

### Requirement: VideoDescriptor
The library SHALL provide `VideoDescriptor` in `skai.description`, a generator returning one natural-language description per video, based on `n_frames` sampled frames and, on backends that accept audio, the audio track, with `detail` and optional `instructions`.

#### Scenario: Describe videos
- **WHEN** `VideoDescriptor(n_frames=8, backend='dummy').fit().predict([v1])` is called
- **THEN** it returns one description string

#### Scenario: Video without audio
- **WHEN** a video has no audio track
- **THEN** it is described from its frames alone
