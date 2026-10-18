# module-layout Specification

## Purpose
Defines where public components are imported from, so users find a component by the task it performs and read its
input type from its class name.

## Requirements

### Requirement: Components grouped by task family
Every public component SHALL be importable from exactly one module named after its task family: `skai.classification`, `skai.description`, `skai.transcription`, `skai.text`, `skai.embedding` or `skai.generation`.

#### Scenario: Import by task family
- **WHEN** a user imports `TextClassifier`, `ImageClassifier`, `AudioClassifier` and `VideoClassifier` from `skai.classification`, `ImageDescriptor`, `AudioDescriptor` and `VideoDescriptor` from `skai.description`, `SpeechTranscriber` and `VideoTranscriber` from `skai.transcription`, `TextSummarizer`, `TextTranslator` and `TextExtractor` from `skai.text`, `TextEmbedder`, `ImageEmbedder` and `AudioEmbedder` from `skai.embedding`, and `SpeechSynthesizer`, `ImageGenerator`, `TalkingAvatarGenerator`, `VideoGenerator`, `ImageEditor` and `ImageAnimator` from `skai.generation`
- **THEN** every import succeeds

#### Scenario: Single location per component
- **WHEN** the public names of all task-family modules are listed
- **THEN** no component name appears in more than one module

### Requirement: Class names carry the input type
Every component name SHALL start with its primary input type (`Text`, `Image`, `Speech`, `Audio` or `Video`), except generators, which SHALL be named for their output and take their input as `X`.

#### Scenario: Input type in the name
- **WHEN** a user reads the names of the classification, description, transcription, text and embedding components
- **THEN** each starts with the type of input it accepts

### Requirement: Input-type modules removed
The modules `skai.image` and `skai.audio` SHALL NOT exist, and `skai.text` SHALL export only text-to-text components.

#### Scenario: Old import path
- **WHEN** a user runs `from skai.image import ImageDescriptor` or `from skai.text import TextClassifier`
- **THEN** an `ImportError` is raised
