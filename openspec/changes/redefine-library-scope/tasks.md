# Tasks

## 1. Remove old scope and scaffold

- [ ] 1.1 Delete `skai.llm`, `skai.base` and their tests, and verify `pdm tests` collects no references to them
- [ ] 1.2 Update the project description in `pyproject.toml`, `.copier-answers.yml` and README intro, and verify no "Reinforcement Learning"/"Large Language Models" framing remains (`grep -ri`)
- [ ] 1.3 Create the `skai/{io,backends,text,image,audio}` packages and declare required dependencies plus the `openai`, `hf`, `litellm`, `replicate`, `video` and `all` extras, then verify `pdm install` and `pdm checks dependencies` (deptry) pass

## 2. Media I/O

- [ ] 2.1 Implement sample-container coercion and text validation, with tests for the container and text-input scenarios in `media-io`
- [ ] 2.2 Implement image, audio and video input coercion (path, URL, bytes, objects, arrays), with tests for each accepted form and for unreadable inputs
- [ ] 2.3 Implement the `Audio`/`Video` containers and `output_format` writers (`object`/`bytes`/`path`, `output_dir`, no overwrite), with tests for every output-format scenario
- [ ] 2.4 Document input/output forms on a docs "Media inputs and outputs" page and verify `pdm docs build` passes

## 3. Backend layer

- [ ] 3.1 Implement the `Backend` base (params, secret masking, lazy client), registry, string parsing and entry-point discovery, with tests for the string, object, nested-param, registration and credential scenarios in `backends`
- [ ] 3.2 Implement the `dummy` provider for all R1 tasks, and verify deterministic, correctly typed outputs per task
- [ ] 3.3 Implement the missing-extra `ImportError` and the unsupported-provider `ValueError`, with tests for both
- [ ] 3.4 Document backend selection, credentials and custom registration, and verify the doc examples run against `dummy`

## 4. Estimator base

- [ ] 4.1 Implement `BaseMediaEstimator` (param validation in fit, backend resolution, NotFitted, `batch_size`, ordered concurrent execution, `on_error`, `random_state`), with tests for every `media-estimator-api` scenario using `dummy`
- [ ] 4.2 Implement `GeneratorMixin` (predict == transform), `MediaClassifierMixin` (classes_, binary/multiclass/multilabel, `labels` zero-shot, `k_shot`, predict_proba) and `EmbedderMixin` (transform, feature names), with clone/GridSearchCV/Pipeline tests

## 5. Text components

- [ ] 5.1 Implement `TextClassifier` on `dummy`, `openai`, `litellm` and `hf`, porting the existing classifier tests, and verify the `text-components` classifier scenarios pass
- [ ] 5.2 Implement `TextSummarizer` and `TextTranslator` on all four providers, and verify their scenarios pass
- [ ] 5.3 Implement `TextEmbedder` on `dummy`, `openai`, `litellm` and `hf`, and verify the shape and pipeline scenarios pass
- [ ] 5.4 Implement `SpeechSynthesizer` on `dummy`, `openai`, `hf` and `replicate`, including the voice-cloning capability check, and verify its scenarios pass
- [ ] 5.5 Implement `ImageGenerator` on `dummy`, `openai`, `hf` (diffusers) and `replicate`, and verify the size and seed scenarios pass
- [ ] 5.6 Add API reference and a usage example per text component, and verify `pdm docs build` passes

## 6. Image components

- [ ] 6.1 Implement `ImageDescriptor` on `dummy`, `openai`, `litellm` and `hf`, and verify its scenarios, including the descriptor → classifier pipeline
- [ ] 6.2 Implement `ImageClassifier` on `dummy`, `openai`, `litellm` and `hf` (CLIP zero-shot), and verify the zero-shot and label-conflict scenarios
- [ ] 6.3 Add API reference and usage examples for image components, and verify `pdm docs build` passes

## 7. Audio components

- [ ] 7.1 Implement `SpeechTranscriber` on `dummy`, `openai` and `hf` (Whisper) with `timestamps`, and verify its scenarios, including the transcriber → classifier pipeline
- [ ] 7.2 Implement `TalkingAvatarGenerator` on `dummy` and `replicate` (pinned open-source model, the default), and verify the duration, missing-avatar and synthesizer → avatar pipeline scenarios
- [ ] 7.3 Add API reference, usage examples and the responsible-use note for audio components, and verify `pdm docs build` passes

## 8. Integration and release

- [ ] 8.1 Add opt-in real-backend tests (skipped unless `SKAI_BACKEND_TESTS` and credentials are set), and run them once locally per provider
- [ ] 8.2 Rewrite the README around tasks with the backend matrix and the default-model table, and verify every README snippet runs against `dummy`
- [ ] 8.3 Fill in the Project Profile in `openspec/constitution.md` for the new scope, and verify `openspec validate --all --strict` passes
- [ ] 8.4 Add the 0.2.0 CHANGELOG migration note (`OpenAIClassifier` → `TextClassifier(backend="openai")`), and verify CI passes on all supported Python versions

## Workflow follow-up

- Archive `redefine-library-scope` after review, creating the main specs under `openspec/specs/`.
