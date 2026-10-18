# Spec Delta

## Purpose

Generates text from texts, images, audio, videos and multimodal samples, by following instructions or by transcribing
exactly what is said or written, so summaries, translations, descriptions, answers, scripts and transcripts are text
generators configured by their instructions and operation.

## ADDED Requirements

### Requirement: Text generators for every input
The library SHALL provide `TextToTextGenerator`, `ImageToTextGenerator`, `AudioToTextGenerator`, `VideoToTextGenerator` (with `n_frames`) and `MultimodalToTextGenerator` in `skai.generation`. Each SHALL return one text per sample, in an object array, from both `predict` and `transform`.

#### Scenario: Summarize, translate and describe
- **WHEN** `TextToTextGenerator(instructions='Summarize in one sentence.')`, `TextToTextGenerator(instructions='Translate into Greek.')` and `ImageToTextGenerator(instructions='Describe the image.')` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns an object array with one string per sample

### Requirement: Instruct operation
With `operation='instruct'`, a text generator SHALL follow its required `instructions` with an instructable model. It is the default of every text generator except `AudioToTextGenerator`.

#### Scenario: Missing instructions
- **WHEN** a text generator with `operation='instruct'` is fitted without `instructions`
- **THEN** `InvalidParameterError` is raised

#### Scenario: Multimodal context
- **WHEN** `MultimodalToTextGenerator(instructions='Write a caption.')` predicts on `{'photo': image, 'notes': 'taken in Crete'}`
- **THEN** the request carries the image and the text, each named by its key

### Requirement: Transcribe operation
With `operation='transcribe'`, `AudioToTextGenerator` (its default) and `VideoToTextGenerator` SHALL return exactly what is said, in an optional `language`, using speech recognition models, and `ImageToTextGenerator` SHALL return exactly what is written, as plain text or, with `format='markdown'`, keeping the page structure. A video without an audio track SHALL give an empty transcript without calling the backend.

#### Scenario: Speech, video and page
- **WHEN** `AudioToTextGenerator()`, `VideoToTextGenerator(operation='transcribe')` and `ImageToTextGenerator(operation='transcribe', format='markdown')` predict on `dummy`
- **THEN** each returns one string per sample

#### Scenario: Silent video
- **WHEN** `VideoToTextGenerator(operation='transcribe')` receives a video without an audio track
- **THEN** its transcript is an empty string

### Requirement: Transcription learns a vocabulary
With `operation='transcribe'`, `fit(X, y)` with reference transcripts as `y` SHALL learn `vocabulary_`, the domain terms of the references: tokens of two or more characters that contain a digit, have an upper-case letter after their first character, or start with an upper-case letter mid-sentence, ordered by frequency and capped at `max_vocabulary`. It SHALL be sent to backends as context for spelling, with a warning at `fit` when the backend cannot take context.

#### Scenario: Domain terms
- **WHEN** `AudioToTextGenerator()` is fitted with transcripts that mention `Kubernetes` and `gRPC`
- **THEN** both are in `vocabulary_` and every request carries them

### Requirement: Instructed generators learn from references
With `operation='instruct'`, `fit(X, y)` with reference texts as `y` SHALL pass `k_shot` sample and reference pairs to the backend as examples. With `max_length='auto'` it SHALL learn `max_length_` from the 90th percentile of the reference word counts, and an integer `max_length` SHALL cut outputs to that many words.

#### Scenario: Learned length and examples
- **WHEN** `TextToTextGenerator(instructions='Summarize.', max_length='auto', k_shot=2, backend='dummy')` is fitted on texts and references of 8 to 12 words
- **THEN** `max_length_` is between 8 and 12, outputs have at most `max_length_` words, and every request carries two example pairs

### Requirement: Text generator learns a glossary
`TextToTextGenerator(learn_terms=True).fit(X, y)` SHALL learn `glossary_`, the terms that appear in at least two source texts mapped to the term their references share, and send it with every request.

#### Scenario: Consistent product name
- **WHEN** it is fitted on sources and translations that both keep `SkaiBox`
- **THEN** `glossary_` maps `SkaiBox` to `SkaiBox`

### Requirement: Text generators can be fine-tuned
Text generators with `operation='instruct'` SHALL support `adaptation='finetune'` from samples and their references.

#### Scenario: Fine-tuned generator
- **WHEN** `TextToTextGenerator(instructions='Summarize.', adaptation='finetune', backend='dummy')` is fitted with 20 texts and references
- **THEN** `model_` is set and requests go to that model without examples
