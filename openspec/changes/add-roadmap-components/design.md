# Design

## Context

Components declare a `task`, build a request per sample, and call `Backend.run`. Backends implement a task by
overriding the method of the same name, and `PromptedBackend` derives text and image tasks from one `complete` method.
Video inputs are `Video` objects that decode lazily with PyAV. See proposal.md for the motivation and specs/ for the
requirements.

## Goals / Non-Goals

**Goals:**
- Add the eleven components with the same contract as the first ten, mostly through new request types and backend
  methods, not new base classes.
- Make the pipeline recipes executable tests, not only documentation.

**Non-Goals:**
- Steps with two inputs that vary per sample (`LipSyncer`, `VideoDubber`).
- Streaming or chunked processing of long videos.

## Decisions

### New tasks and requests
| Component | Task | Request |
|---|---|---|
| `AudioClassifier` | `classify_audio` | `ClassificationRequest` with an `Audio` sample |
| `VideoClassifier` | `classify_video` | `ClassificationRequest` with a `VideoFrames` sample |
| `AudioDescriptor` | `describe_audio` | `DescriptionRequest` with an `Audio` sample |
| `VideoDescriptor` | `describe_video` | `DescriptionRequest` with a `VideoFrames` sample |
| `VideoTranscriber` | `transcribe` (reused) | `TranscriptionRequest` with the extracted audio track |
| `TextExtractor` | `extract_fields` | `ExtractionRequest(text, fields)` |
| `ImageEmbedder` | `embed_image` | `EmbeddingRequest` with an image sample |
| `AudioEmbedder` | `embed_audio` | `EmbeddingRequest` with an `Audio` sample |
| `VideoGenerator` | `generate_video` | `VideoGenerationRequest(prompt, duration, size, fps, style, negative_prompt, seed)` |
| `ImageEditor` | `edit_image` | `ImageEditRequest(image, instruction, mask, seed)` |
| `ImageAnimator` | `animate_image` | `AnimationRequest(image, motion_prompt, duration, fps, seed)` |

`ClassificationRequest.sample` and `DescriptionRequest` widen to the new sample types. `VideoFrames(frames, timestamps,
audio)` is a new frozen dataclass that components build with the sampling helper, so backends never decode video
themselves. `VideoTranscriber` reuses `transcribe`, so every transcription backend supports it for free.

### Prompts carry audio
`Prompt` gains `audio: tuple[Audio, ...]`, referred to as `<audio 1>`, like images. `PromptedBackend` implements
`classify_audio`, `describe_audio`, `classify_video`, `describe_video` and `extract_fields` on top of `complete`.
Video tasks send the sampled frames as images, plus the audio track when the backend accepts audio. Backends declare
`accepts_audio: ClassVar[bool]`. A component whose task needs audio raises `UnsupportedBackendError` at `fit` when the
backend does not accept it, as voice cloning and timestamps do today.

### Structured extraction
`extract_fields` asks for a JSON object with exactly the field names and `null` for absent fields, and uses the
provider's JSON mode where it exists (`response_format={'type': 'json_object'}` on OpenAI). The parser keeps known
keys, fills missing keys with `None`, drops unknown keys, and raises `InvalidResponseError` on invalid JSON, which
`on_error` then handles. `transform` returns the values in `fields` order, and `get_feature_names_out` returns the
field names. That is the first generator with feature names, so `set_output(transform='pandas')` works on it.

### Embeddings and the shared space
`ImageEmbedder` and `AudioEmbedder` reuse `BaseEmbedder`. On `hf`, image and text embeddings both go through
sentence-transformers, which encodes texts and images with CLIP models such as `clip-ViT-B-32`. The same model in
both embedders gives the shared space that the spec requires. Audio uses CLAP through `transformers`. The default
`TextEmbedder` model stays a text-only model, and the docs show how to pick a joint model for retrieval.

### Frame sampling and audio extraction
`core._io` gains `sample_frames(video, n_frames)`, which returns `VideoFrames` with frames at `linspace(0, n_total,
n_frames, endpoint=False)` rounded down. That puts the first frame first and spaces the rest evenly. `Video` gains an
`audio_track()` method that returns mono audio, or silence of the video's length when there is no track. The first
version decodes the whole video and then indexes it. Seeking with PyAV is a later optimization.

### Pipelines are plain scikit-learn pipelines
No pipeline API is added. Every recipe in the `pipelines` spec is a `make_pipeline` of existing components and is
tested end to end on `dummy`. Caching needs components to be hashable by `joblib`. They already are when the backend
is a string, because runtime clients live only in fitted attributes that a cached clone never has. A backend object
with an injected client is documented as not cacheable.

### Default models
Backends implement a task only where their provider supports it, and components raise `UnsupportedBackendError` at
`fit` elsewhere. OpenAI defaults are verified by the end-to-end suite. The `hf` and `litellm` defaults follow the
providers' documented APIs and are verified when credentials or hardware are available.

| Component | `hf` | `openai` | `litellm` |
|---|---|---|---|
| `AudioClassifier` | `laion/clap-htsat-unfused` | `gpt-audio-mini` | `gemini/gemini-2.5-flash` |
| `VideoClassifier` | `HuggingFaceTB/SmolVLM-256M-Instruct` (frames) | `gpt-5-mini` (frames) | `gemini/gemini-2.5-flash` |
| `AudioDescriptor` | `Qwen/Qwen2-Audio-7B-Instruct` | `gpt-audio-mini` | `gemini/gemini-2.5-flash` |
| `VideoDescriptor` | `HuggingFaceTB/SmolVLM-256M-Instruct` (frames) | `gpt-5-mini` (frames) | `gemini/gemini-2.5-flash` |
| `TextExtractor` | `Qwen/Qwen2.5-1.5B-Instruct` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` |
| `ImageEmbedder` | `clip-ViT-B-32` | | `cohere/embed-v4.0` |
| `AudioEmbedder` | `laion/clap-htsat-unfused` | | |
| `VideoGenerator` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `sora-2` | |
| `ImageEditor` | `timbrooks/instruct-pix2pix` | `gpt-image-1` | |
| `ImageAnimator` | `stabilityai/stable-video-diffusion-img2vid-xt` | `sora-2` with the image as first frame | |

Replicate is left out of the new tasks, because the inputs of the candidate models could not be confirmed and OpenAI
covers the cloud side. No cloud provider offers audio embeddings, so `AudioEmbedder` is the one exception to cloud
coverage. Sora takes durations of 4, 8 or 12 seconds and a few fixed sizes, so `VideoGenerator` and `ImageAnimator`
round to the closest and resize the frames to the requested size.

## Risks / Trade-offs

- [Model ids and provider APIs drift, and the first release showed that fakes alone hide this] → Implement each component only after one real call per provider, record that payload, and test against the recording.
- [Video models are large and slow locally] → Default `ImageAnimator` and `VideoGenerator` to the smallest checkpoints, and say in the docs that local video generation needs a GPU.
- [Decoding whole videos uses a lot of memory] → Document the limit for the first version and move to seeking when a user needs long videos.
- [Generator steps in a pipeline run on every training sample during `fit`] → The `pipelines` spec states it, and the docs recommend `Pipeline(memory=...)`.
