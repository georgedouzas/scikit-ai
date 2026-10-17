# Spec Delta

## ADDED Requirements

### Requirement: VideoTranscriber
The library SHALL provide `VideoTranscriber` in `skai.transcription`, a generator that transcribes the audio track of every video, with the same `language` and `timestamps` parameters and outputs as `SpeechTranscriber`.

#### Scenario: Transcribe a video
- **WHEN** `VideoTranscriber(backend='dummy').fit().predict([video])` is called on a video with speech
- **THEN** it returns one transcript string

#### Scenario: Timestamps follow the video
- **WHEN** `timestamps='segment'` is set
- **THEN** segment times are measured from the start of the video

#### Scenario: Video without audio
- **WHEN** a video has no audio track
- **THEN** its transcript is an empty string
