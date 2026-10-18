# Tasks

## 1. Core

- [x] 1.1 Add the request types, `VideoFrames`, the `extract_fields`, `classify_audio`, `classify_video`, `describe_audio`, `describe_video`, `embed_image`, `embed_audio`, `generate_video`, `edit_image` and `animate_image` tasks to `Backend`, and `accepts_audio`, with tests that `supported_tasks` reports them
- [x] 1.2 Add `sample_frames` and `Video.audio_track`, with tests for the frame sampling and audio extraction scenarios in `media-io`
- [x] 1.3 Add audio to `Prompt` and implement the audio, video and extraction tasks on `PromptedBackend`, with tests on a fake prompted backend
- [x] 1.4 Implement every new task on the `dummy` backend, and verify deterministic, correctly typed outputs per task

## 2. Provider verification

- [x] 2.1 Run the LiteLLM backend end to end through OpenAI routes (`openai/...` models) with the existing key: prompted text, image, audio and video tasks, JSON extraction and text embeddings, record the payloads, and verify they pass
- [x] 2.7 Run the Hugging Face backend end to end with the small local models (zero-shot text and image classification, embeddings with MiniLM and CLIP, Whisper-small transcription, MMS-TTS speech, SD-Turbo images, SmolVLM description, Qwen2.5-1.5B summary, translation and extraction, CLAP audio classification and embeddings), behind `SKAI_E2E_HF=1`, fix what fails, and verify they pass
- [ ] 2.8 Verify the remaining defaults when credentials and hardware are available: Gemini, Anthropic and Cohere through LiteLLM, and Qwen2-Audio-7B, Wan, InstructPix2Pix and Stable Video Diffusion on `hf` (blocked)
- [x] 2.5 Add an OpenAI end-to-end scenario for every new component OpenAI supports, record the payloads, and verify they pass (audio classification and description, video classification, description and transcription, extraction, image editing)
- [ ] 2.6 Run the Sora scenario for `VideoGenerator` and `ImageAnimator` behind `SKAI_E2E_VIDEO=1` and verify it passes (blocked: the OpenAI account gets 404 from `/v1/videos`, so it has no Sora API access)
- [x] 2.2 Add the `pdm e2e` harness: an `e2e/` suite outside `pdm tests` that loads credentials from `.env`, skips without them, records sanitized responses under `tests/payloads/<provider>/`, gates video generation behind `SKAI_E2E_VIDEO=1`, and a test that fails if a recording holds anything shaped like a credential
- [x] 2.3 Run `pdm e2e` on the OpenAI backend for the built components, fix every adapter or default model that fails, and verify all OpenAI end-to-end tests pass
- [x] 2.4 Add OpenAI end-to-end pipelines for the built components (descriptions → classifier, transcribe → summarize → translate, files between steps, descriptions → speech, descriptions → embeddings → `LogisticRegression`) on images generated once per session, and verify they pass

## 3. Classification and description

- [x] 3.1 Implement `AudioClassifier` and `VideoClassifier` on `dummy`, `hf`, `openai` and `litellm`, and verify the `classification` scenarios
- [x] 3.2 Implement `AudioDescriptor` and `VideoDescriptor` on `dummy`, `hf`, `openai` and `litellm`, including the audio capability check, and verify the `description` scenarios
- [x] 3.3 Document the four components in the classification and description guide pages, and verify `pdm docs build` passes

## 4. Transcription and text

- [x] 4.1 Implement `VideoTranscriber` on every transcription backend, and verify the `transcription` scenarios
- [x] 4.2 Implement `TextExtractor` with JSON parsing, feature columns and `get_feature_names_out` on `dummy`, `hf`, `openai` and `litellm`, and verify the `text-processing` scenarios
- [x] 4.3 Document both components, and verify `pdm docs build` passes

## 5. Embedding

- [x] 5.1 Implement `ImageEmbedder` on `dummy`, `hf` and `litellm`, and `AudioEmbedder` on `dummy` and `hf`, and verify the `embedding` scenarios, including the shared text and image space
- [x] 5.2 Document both components and text-to-image retrieval, and verify `pdm docs build` passes

## 6. Generation

- [x] 6.1 Implement `VideoGenerator` on `dummy`, `hf` and `openai`, and verify its scenario
- [x] 6.2 Implement `ImageEditor`, with the mask capability check, on `dummy`, `hf` and `openai`, and verify its scenarios
- [x] 6.3 Implement `ImageAnimator` on `dummy`, `hf` and `openai`, and verify its scenario
- [x] 6.4 Document the three components in the generation guide page, and verify `pdm docs build` passes

## 7. Pipelines

- [x] 7.1 Add one test per requirement of the `pipelines` spec on `dummy`, including the backend call counts, the `joblib` cache and the cross-step grid search
- [x] 7.2 Write a "Pipelines" guide page with every recipe as an executed example and the caching advice, and verify `pdm docs build` passes

## 8. Integration

- [x] 8.1 Update the README component table, the module-layout tests and the constitution's Project Profile, then run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order and verify they all pass
