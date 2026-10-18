# Spec Delta

## Purpose

Gives common tasks readable names on top of the components: presets return configured components, and recipes return
configured pipelines, both as plain scikit-learn estimators.

## ADDED Requirements

### Requirement: Component presets
The library SHALL provide `skai.presets`, functions that return a component configured for a common task, and every preset SHALL pass extra keyword arguments to the component.

#### Scenario: Available presets
- **WHEN** a user lists `skai.presets`
- **THEN** it holds at least `summarizer`, `translator`, `rewriter`, `describer`, `question_answerer`, `script_writer`, `prompt_writer`, `quiz_writer`, `storyboarder`, `moderator`, `ocr`, `document_parser`, `diarizer`, `object_detector`, `shot_detector`, `background_remover`, `upscaler`, `music_generator`, `voice_converter`, `separator`, `lip_syncer`, `talking_head` and `animator`

#### Scenario: Summarizer preset
- **WHEN** `skai.presets.summarizer(max_length=50, backend='dummy')` is called
- **THEN** it returns a `TextToTextGenerator` with summarizing instructions, `max_length=50` and the `dummy` backend

#### Scenario: Presets are ordinary estimators
- **WHEN** a preset's result is cloned, put in a pipeline or tuned with `GridSearchCV`
- **THEN** it behaves exactly like the component it returns

### Requirement: Pipeline recipes
The library SHALL provide `skai.recipes`, functions that return a configured `Pipeline` for every recipe of the `pipelines` capability. Every recipe SHALL take a `backend` passed to every step that uses one, and keyword arguments for its main choices.

#### Scenario: Recipe returns a pipeline
- **WHEN** `skai.recipes.explainer_video(backend='dummy')` is called
- **THEN** it returns a scikit-learn `Pipeline` whose steps can be inspected and changed with `set_params`
