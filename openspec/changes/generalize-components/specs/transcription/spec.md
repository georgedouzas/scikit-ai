# Spec Delta

## REMOVED Requirements

### Requirement: SpeechTranscriber
**Reason**: Transcription is an operation of the text generators of audio, video and images.
**Migration**: Use `AudioToTextGenerator()`, whose default operation is `'transcribe'`.

### Requirement: Timestamped transcripts
**Reason**: Timed transcripts are lists of time regions.
**Migration**: Use `AudioSegmenter(by='speech')`, whose regions carry `start`, `end` and `text`.

### Requirement: Transcriber learns a vocabulary
**Reason**: The vocabulary is part of the transcribe operation.
**Migration**: See the `text-generation` capability.

### Requirement: VideoTranscriber
**Reason**: Transcription is an operation of the text generators.
**Migration**: Use `VideoToTextGenerator(operation='transcribe')`.
