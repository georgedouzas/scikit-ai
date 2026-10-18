# Spec Delta

## Purpose

Generates text from text, images, audio, video or records by following instructions, so summaries, translations,
descriptions, answers, scripts, prompts and quizzes are one component configured by its instructions.

## ADDED Requirements

### Requirement: TextGenerator
The library SHALL provide `TextGenerator` in `skai.generation`, a generator that returns one text per sample by following its required `instructions`, for any input type selected by `input` (`'text'` by default, `'image'`, `'audio'`, `'video'` or `'record'`).

#### Scenario: Summarize, translate and describe
- **WHEN** `TextGenerator(instructions='Summarize in one sentence.')`, `TextGenerator(instructions='Translate into Greek.')` and `TextGenerator(input='image', instructions='Describe the image.')` are fitted on `dummy` and predict on samples of their type
- **THEN** each returns an object array with one string per sample

#### Scenario: Missing instructions
- **WHEN** `TextGenerator()` is fitted without `instructions`
- **THEN** `InvalidParameterError` is raised

### Requirement: Generation from records
With `input='record'`, every sample SHALL be a dict whose values are texts, images, audio or videos, and `TextGenerator` SHALL send all of them to the backend in one request, named by their keys.

#### Scenario: Caption with context
- **WHEN** a sample `{'photo': image, 'notes': 'taken in Crete'}` is passed to `TextGenerator(input='record', instructions='Write a caption.')`
- **THEN** the request carries the image and the text, each labeled with its key

### Requirement: Generator learns from reference texts
`TextGenerator.fit(X, y)` with reference outputs as `y` SHALL pass `k_shot` sample and reference pairs to the backend as examples. With `max_length='auto'` it SHALL learn `max_length_` from the 90th percentile of the reference word counts, and an integer `max_length` SHALL cut outputs to that many words.

#### Scenario: Learned length and examples
- **WHEN** `TextGenerator(instructions='Summarize.', max_length='auto', k_shot=2, backend='dummy')` is fitted on texts and references of 8 to 12 words
- **THEN** `max_length_` is between 8 and 12, outputs have at most `max_length_` words, and every request carries two example pairs

### Requirement: Generator learns a glossary
With `learn_terms=True` and text input, `fit(X, y)` SHALL learn `glossary_`, the terms that appear in at least two source texts mapped to the term their references share, and send it with every request.

#### Scenario: Consistent product name
- **WHEN** it is fitted on sources and translations that both keep `SkaiBox`
- **THEN** `glossary_` maps `SkaiBox` to `SkaiBox`

### Requirement: Text generator can be fine-tuned
`TextGenerator` SHALL support `adaptation='finetune'` from samples and their references.

#### Scenario: Fine-tuned generator
- **WHEN** `TextGenerator(instructions='Summarize.', adaptation='finetune', backend='dummy')` is fitted with 20 texts and references
- **THEN** `model_` is set and requests go to that model without examples
