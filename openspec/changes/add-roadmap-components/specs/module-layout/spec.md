# Spec Delta

## MODIFIED Requirements

### Requirement: Components grouped by task family
Every public component SHALL be importable from exactly one module named after its task family: `skai.classification`, `skai.description`, `skai.transcription`, `skai.text`, `skai.embedding` or `skai.generation`.

#### Scenario: Import by task family
- **WHEN** a user imports `TextClassifier`, `ImageClassifier`, `AudioClassifier` and `VideoClassifier` from `skai.classification`, `ImageDescriptor`, `AudioDescriptor` and `VideoDescriptor` from `skai.description`, `SpeechTranscriber` and `VideoTranscriber` from `skai.transcription`, `TextSummarizer`, `TextTranslator` and `TextExtractor` from `skai.text`, `TextEmbedder`, `ImageEmbedder` and `AudioEmbedder` from `skai.embedding`, and `SpeechSynthesizer`, `ImageGenerator`, `TalkingAvatarGenerator`, `VideoGenerator`, `ImageEditor` and `ImageAnimator` from `skai.generation`
- **THEN** every import succeeds

#### Scenario: Single location per component
- **WHEN** the public names of all task-family modules are listed
- **THEN** no component name appears in more than one module
