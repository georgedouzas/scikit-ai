# Spec Delta

## ADDED Requirements

### Requirement: Input type is a parameter
Every component that accepts more than one input type SHALL select it with an `input` parameter that is validated at `fit`, and SHALL raise `MediaReadError` or `InvalidSampleError`, naming the index, for samples of another type.

#### Scenario: Text by default
- **WHEN** `Classifier()`, `TextGenerator(instructions='...')`, `Extractor(fields=...)` and `Embedder()` are constructed without `input`
- **THEN** they take text

#### Scenario: Invalid input type
- **WHEN** `TextGenerator(input='pdf', instructions='...')` is fitted
- **THEN** `InvalidParameterError` names the `input` parameter

## MODIFIED Requirements

### Requirement: Components grouped by task family
Every public component SHALL be importable from exactly one module named after its task family, apart from the deprecated 0.2 names: `Classifier` from `skai.classification`, `TextGenerator`, `ImageGenerator`, `AudioGenerator` and `VideoGenerator` from `skai.generation`, `Extractor` from `skai.extraction`, `Embedder` from `skai.embedding`, `Transcriber` from `skai.transcription`, `Segmenter` from `skai.segmentation`, `Composer` from `skai.composition`, and `MediaUnion`, `Pick`, `ForEach` and `Select` from `skai.pipeline`.

#### Scenario: Import by task family
- **WHEN** a user imports every class listed above from its module
- **THEN** every import succeeds

#### Scenario: Single location per component
- **WHEN** the public names of all task-family modules are listed, leaving out deprecated names
- **THEN** no component name appears in more than one module

## REMOVED Requirements

### Requirement: Class names carry the input type
**Reason**: Components take their input type as a parameter, so one class serves every input type.
**Migration**: Replace the input type in the class name with `input='image'`, `'audio'`, `'video'` or `'record'`.
