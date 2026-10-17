# text-components Specification

## Purpose
Task components whose primary input is text: classification, summarization, translation, embedding,
speech synthesis and image generation, each following the shared media estimator contract.

## Requirements

### Requirement: TextClassifier
The library SHALL provide `TextClassifier`, a classifier that assigns texts to labels using zero-shot or few-shot prompting, with optional `instructions` that describe the labeling criteria.

#### Scenario: Few-shot binary classification
- **WHEN** `TextClassifier(k_shot=4)` is fitted on labeled reviews and predicts on new reviews
- **THEN** it returns one label from `classes_` per review

#### Scenario: Multilabel targets
- **WHEN** it is fitted with a binary indicator matrix `y` of shape `(n, k)`
- **THEN** `predict` returns an indicator matrix of shape `(m, k)` for `m` inputs

### Requirement: TextSummarizer
The library SHALL provide `TextSummarizer`, a generator returning one summary per text, controlled by `max_length` (approximate maximum words) and an optional free-text `style`.

#### Scenario: Length bound
- **WHEN** `TextSummarizer(max_length=50, backend="dummy")` summarizes a long text
- **THEN** the summary has at most 50 words

### Requirement: TextTranslator
The library SHALL provide `TextTranslator`, a generator translating each text into `target_language` (required), with optional `source_language` that defaults to auto-detection.

#### Scenario: Missing target language
- **WHEN** `TextTranslator()` is fitted without `target_language`
- **THEN** `InvalidParameterError` is raised

### Requirement: TextEmbedder
The library SHALL provide `TextEmbedder`, a transformer mapping each text to a fixed-size float vector, with optional `dimensions` for backends that support truncation and `normalize` to return unit-length vectors.

#### Scenario: Fixed dimensionality
- **WHEN** `TextEmbedder(backend="dummy", dimensions=64).fit_transform(texts)` runs on 10 texts
- **THEN** it returns a float array of shape `(10, 64)`

### Requirement: SpeechSynthesizer
The library SHALL provide `SpeechSynthesizer`, a generator converting each text into spoken audio, with `voice` (a provider voice name or a reference audio sample for cloning on supporting backends) and optional `language`.

#### Scenario: One audio per text
- **WHEN** `SpeechSynthesizer(backend="dummy").predict(["Hello", "Bye"])` is called
- **THEN** it returns two audio outputs in the selected `output_format`

#### Scenario: Voice cloning unsupported
- **WHEN** `voice` is a reference audio sample and the backend does not support cloning
- **THEN** `ValueError` is raised at `fit` listing the backends that do

### Requirement: ImageGenerator
The library SHALL provide `ImageGenerator`, a generator creating one image per text prompt, with `size` as `(width, height)`, optional `style` and optional `negative_prompt`.

#### Scenario: Requested size
- **WHEN** `ImageGenerator(size=(512, 512), backend="dummy").predict(["a cat"])` is called
- **THEN** it returns one 512×512 image
