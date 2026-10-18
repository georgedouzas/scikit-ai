# Spec Delta

## MODIFIED Requirements

### Requirement: Components grouped by task family
Every public component SHALL be importable from exactly one module named after its role: classifiers from `skai.classification`, extractors from `skai.extraction`, embedders from `skai.embedding`, segmenters from `skai.segmentation`, generators from `skai.generation`, and `MediaUnion`, `Field`, `ForEach` and `TopK` from `skai.pipeline`.

#### Scenario: Import by task family
- **WHEN** a user imports `TextClassifier` from `skai.classification`, `VideoExtractor` from `skai.extraction`, `AudioEmbedder` from `skai.embedding`, `AudioSegmenter` from `skai.segmentation`, `VideoToTextGenerator` from `skai.generation` and `ForEach` from `skai.pipeline`
- **THEN** every import succeeds

#### Scenario: Single location per component
- **WHEN** the public names of all role modules are listed
- **THEN** no component name appears in more than one module

### Requirement: Class names carry the input type
Analyzers SHALL be named `{Input}{Role}`, with `Input` one of `Text`, `Image`, `Audio`, `Video` or `Multimodal` and `Role` one of `Classifier`, `Extractor`, `Embedder` or `Segmenter`. Generators SHALL be named `{Input}To{Output}Generator`, with `Output` one of `Text`, `Image`, `Audio` or `Video`. The name SHALL fix the type of every sample read and of every output returned.

#### Scenario: Input type in the name
- **WHEN** the public classes of the role modules are listed
- **THEN** every name matches `{Input}{Role}` or `{Input}To{Output}Generator`

### Requirement: Input-type modules removed
The modules `skai.image`, `skai.audio`, `skai.text`, `skai.description` and `skai.transcription` SHALL NOT exist, and the 0.2 classes `TextSummarizer`, `TextTranslator`, `ImageDescriptor`, `AudioDescriptor`, `VideoDescriptor`, `SpeechTranscriber`, `VideoTranscriber`, `SpeechSynthesizer`, `ImageGenerator`, `ImageEditor`, `ImageAnimator`, `VideoGenerator` and `TalkingAvatarGenerator` SHALL NOT be importable.

#### Scenario: Old import path
- **WHEN** a user runs `from skai.text import TextSummarizer` or `from skai.generation import ImageGenerator`
- **THEN** an `ImportError` is raised
