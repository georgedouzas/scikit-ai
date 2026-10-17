# Proposal

## Why

`scikit-ai` currently tries to unify classical ML, RL and LLMs. That scope is too broad to finish and
doesn't tell users what the library is for. A narrower, more useful niche: **task-oriented, high-level
components for unstructured media (text, images, audio, video) that follow the scikit-learn API**. Each
component does one job ("describe this image", "summarize this text", "make this portrait speak this audio"),
works the same whether it runs on an open-source model or a cloud API, and drops into `Pipeline`,
`GridSearchCV`, `cross_val_score` and so on.

## What Changes

- **BREAKING** Remove `skai.llm`, `skai.base.llm`, `skai.base.rl` and the RL/LLM framing in README, docs
  and project description. `OpenAIClassifier` is replaced by a backend-agnostic `TextClassifier`.
- New project description: *"Task-oriented components for text, images, audio and video with a scikit-learn API, powered by open-source models or cloud APIs."*
- Introduce a common estimator contract for media components (input types, output types, `fit` semantics,
  sklearn tags, parameter validation).
- Introduce a backend layer so every component can run on open-source models (local, e.g. Hugging Face) or
  cloud APIs (e.g. OpenAI, Gemini, Anthropic, ElevenLabs, Replicate/fal) chosen by a constructor parameter.
- Introduce media I/O: accept file paths, URLs, bytes and in-memory objects. Return a consistent output type
  per modality.
- Introduce task components grouped by input modality (candidate catalog below).
- Optional dependency extras per backend (`scikit-ai[openai]`, `scikit-ai[hf]`, …) so the base install stays light.

### Design principle: X is what varies, the constructor holds what is fixed

Each sample in `X` is one item of the component's *primary input modality*. Fixed conditioning (a reference
portrait, a target language, a voice, an instruction, a label set) is a constructor parameter. This keeps
every component a plain `fit(X, y=None)` / `predict(X)` estimator. Example:
`TalkingAvatarGenerator(avatar="me.png").predict(["hello.wav", "bye.wav"])` returns two videos.

### Component catalog

Roles: **C** = classifier (`predict`, `predict_proba`; `fit` learns labels / few-shot examples),
**T** = transformer (`transform` returns features, usable mid-pipeline), **G** = generator/converter
(`predict` returns generated text or media, and `transform` returns the same so it can sit mid-pipeline).
**R1** marks the first release (this change). The other rows are a roadmap, each specified in a later change.

| Input | Component | Output | Role | Fixed params (examples) | Release |
|---|---|---|---|---|---|
| text | `TextClassifier` (zero/few-shot, multi-label) | labels | C | `labels`, `instructions` | R1 |
| text | `TextSummarizer` | text | G | `max_length`, `style` | R1 |
| text | `TextTranslator` | text | G | `target_language` | R1 |
| text | `TextExtractor` (entities / structured fields) | records | G | `schema` | later |
| text | `TextEmbedder` | vectors | T | `dimensions` | R1 |
| text | `SpeechSynthesizer` (TTS) | audio | G | `voice` | R1 |
| text | `ImageGenerator` | image | G | `size`, `style` | R1 |
| text | `VideoGenerator` | video | G | `duration` | later |
| image | `ImageDescriptor` (captioning) | text | G | `detail`, `instructions` | R1 |
| image | `ImageClassifier` (zero/few-shot) | labels | C | `labels` | R1 |
| image | `ImageTextExtractor` (OCR / doc parsing) | text | G | `output_format` | later |
| image | `ObjectDetector` | boxes + labels | G | `labels` | deferred |
| image | `ImageEmbedder` | vectors | T | | later |
| image | `ImageEditor` (instruction-based edit) | image | G | `instruction` | later |
| image | `ImageAnimator` (image → video) | video | G | `motion_prompt` | later |
| audio | `SpeechTranscriber` (ASR) | text | G | `language`, `timestamps` | R1 |
| audio | `AudioClassifier` (sound events, emotion) | labels | C | `labels` | later |
| audio | `AudioDescriptor` | text | G | | later |
| audio | `AudioEmbedder` | vectors | T | | later |
| audio | `VoiceConverter` (voice cloning / conversion) | audio | G | `target_voice` | deferred |
| audio | `TalkingAvatarGenerator` | video | G | `avatar` (image) | R1 |
| video | `VideoDescriptor` | text | G | `instructions` | later |
| video | `VideoClassifier` | labels | C | `labels` | later |
| video | `VideoSummarizer` | text | G | `max_length` | later |
| video | `VideoTranscriber` | text | G | `language` | later |
| video | `LipSyncer` | video | G | `audio` | later |
| video | `VideoDubber` | video | G | `target_language`, `voice` | deferred |

"deferred" components (`ObjectDetector`, `VoiceConverter`, `VideoDubber`) are kept off the roadmap until
there is a concrete need. Their outputs or legal surface need their own design.

### Out of scope

Reinforcement learning, agents/tool use, chat/conversation APIs, raw "LLM" or "model" wrappers with no task
attached, training infrastructure, and tabular data. Fine-tuning is out of scope for the first release.

## Capabilities

### New Capabilities
- `media-estimator-api`: the shared contract every component follows (fit/predict/transform semantics, input validation, sklearn tags, clone/get_params compatibility, batching).
- `backends`: choosing and configuring open-source vs cloud execution per component, credentials, and reporting unsupported backend/task pairs.
- `media-io`: accepted input forms per modality and the output representation of generated text, images, audio and video.
- `text-components`: first-release components whose primary input is text (`TextClassifier`, `TextSummarizer`, `TextTranslator`, `TextEmbedder`, `SpeechSynthesizer`, `ImageGenerator`).
- `image-components`: first-release components whose primary input is an image (`ImageDescriptor`, `ImageClassifier`).
- `audio-components`: first-release components whose primary input is audio (`SpeechTranscriber`, `TalkingAvatarGenerator`).

`video-components` is introduced by the first change that ships a video-input component.

### Modified Capabilities

None. No specs exist yet.

## Impact

- Public API: everything under `skai.llm` / `skai.base` is removed (pre-1.0, released only as 0.1.0).
  New modules `skai.text`, `skai.image`, `skai.audio` (and `skai.video` later), organized by primary input type.
- Dependencies: `openai` moves from a required dependency to an extra. New optional `transformers`/`diffusers`/`torch`,
  provider SDKs, and media libraries (Pillow, soundfile, and a video reader).
- Tests: the existing classifier tests are rewritten against `TextClassifier` with mocked backends. Real-backend
  tests are opt-in.
- Docs: README, usage and API reference rewritten around tasks.
- Depends on `update-copier-template` landing first.
