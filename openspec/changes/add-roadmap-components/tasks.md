# Tasks

## 1. Core

- [x] 1.1 Add the request types, `VideoFrames`, the `extract_fields`, `classify_audio`, `classify_video`, `describe_audio`, `describe_video`, `embed_image`, `embed_audio`, `generate_video`, `edit_image` and `animate_image` tasks to `Backend`, and `accepts_audio`, with tests that `supported_tasks` reports them
- [x] 1.2 Add `sample_frames` and `Video.audio_track`, with tests for the frame sampling and audio extraction scenarios in `media-io`
- [x] 1.3 Add audio to `Prompt` and implement the audio, video and extraction tasks on `PromptedBackend`, with tests on a fake prompted backend
- [x] 1.4 Implement every new task on the `dummy` backend, and verify deterministic, correctly typed outputs per task

## 2. Provider verification

- [ ] 2.1 Make one real call per provider and new task with the planned default models, record the payloads under `tests/payloads/`, and update the default-model table in the design and the README with the models that work

## 3. Classification and description

- [ ] 3.1 Implement `AudioClassifier` and `VideoClassifier` on `dummy`, `hf`, `openai` and `litellm`, and verify the `classification` scenarios and the recorded-payload tests
- [ ] 3.2 Implement `AudioDescriptor` and `VideoDescriptor` on `dummy`, `hf`, `openai` and `litellm`, including the audio capability check, and verify the `description` scenarios
- [ ] 3.3 Document the four components in the classification and description guide pages, and verify `pdm docs build` passes

## 4. Transcription and text

- [ ] 4.1 Implement `VideoTranscriber` on every transcription backend, and verify the `transcription` scenarios
- [ ] 4.2 Implement `TextExtractor` with JSON parsing, feature columns and `get_feature_names_out` on `dummy`, `hf`, `openai` and `litellm`, and verify the `text-processing` scenarios
- [ ] 4.3 Document both components, and verify `pdm docs build` passes

## 5. Embedding

- [ ] 5.1 Implement `ImageEmbedder` and `AudioEmbedder` on `dummy`, `hf` and `replicate`, and verify the `embedding` scenarios, including the shared text and image space
- [ ] 5.2 Document both components and text-to-image retrieval, and verify `pdm docs build` passes

## 6. Generation

- [ ] 6.1 Implement `VideoGenerator` on `dummy`, `hf`, `openai` and `replicate`, and verify its scenario
- [ ] 6.2 Implement `ImageEditor`, with the mask capability check, on `dummy`, `hf`, `openai` and `replicate`, and verify its scenarios
- [ ] 6.3 Implement `ImageAnimator` on `dummy`, `hf` and `replicate`, and verify its scenario
- [ ] 6.4 Document the three components in the generation guide page, and verify `pdm docs build` passes

## 7. Pipelines

- [ ] 7.1 Add one test per requirement of the `pipelines` spec on `dummy`, including the backend call counts, the `joblib` cache and the cross-step grid search
- [ ] 7.2 Write a "Pipelines" guide page with every recipe as an executed example and the caching advice, and verify `pdm docs build` passes

## 8. Integration

- [ ] 8.1 Update the README component table, the module-layout tests and the constitution's Project Profile, then run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order and verify they all pass
