# Design

## Context

`skai` today ships `skai.llm.OpenAIClassifier` (sklearn-compatible, OpenAI-only, async batching, `k_shot` example
selection) and an empty `skai.base.rl`. The requirements are in `specs/`, and the motivation and catalog are
in `proposal.md`. This change assumes `update-copier-template` has landed (PDM, properdocs, deptry, pydoclint,
committed lock).

## Goals / Non-Goals

**Goals:**
- A small set of base mixins so each new component is mostly a task definition plus provider adapters.
- A provider layer where adding a provider or a task doesn't touch existing components.
- Tests that run fully offline on the `dummy` provider.

**Non-Goals:**
- Fine-tuning, response caching, streaming outputs, chat/agent APIs.
- Video-input components (a later change introduces `video-components`).
- Matching every provider feature. Adapters expose what the task needs, plus a pass-through `options` dict.

## Decisions

### Package layout by primary input type
```
skai/
  base.py            # BaseMediaEstimator, GeneratorMixin, MediaClassifierMixin, EmbedderMixin
  io/                # input coercion per modality, Audio/Video containers, output writers
  backends/          # Backend base, registry, string parsing, providers/{dummy,openai,hf,litellm,replicate}.py
  text/              # TextClassifier, TextSummarizer, TextTranslator, TextEmbedder, SpeechSynthesizer, ImageGenerator
  image/             # ImageDescriptor, ImageClassifier
  audio/             # SpeechTranscriber, TalkingAvatarGenerator
```
All public components are also re-exported from `skai`. Organizing by task was the alternative. It was rejected
because the user picks a module by what data they hold.

### Components declare a task, backends implement tasks
Each component has a class-level `_task` (e.g. `"summarize"`), and a typed request/response contract lives with
it. `fit` resolves `backend` → backend instance (`backend_`), checks `_task in backend_.supported_tasks`, and
raises `ValueError` listing capable providers otherwise. Components build the task request (prompt template,
labels, examples, options). Backends only execute it. The alternative was one class per (component, provider)
pair, like today's `OpenAIClassifier`. That grows as components × providers and duplicates sklearn plumbing.

### Backend objects are estimators-lite; strings are sugar
`Backend` subclasses `sklearn.base.BaseEstimator` for `get_params`/`set_params`/`clone`, so
`backend__model` works in `GridSearchCV`. `"openai:gpt-5-mini"` is parsed by the registry into
`OpenAIBackend(model="gpt-5-mini")`, and `"openai"` gets the provider's per-task default model. Secrets are
held in a private attribute that `get_params` omits and `__repr__` masks. Clients are created lazily in `fit`
and not deep-copied by `clone`.

### Registry
`skai.backends.register(name, backend_cls)`. The class declares `supported_tasks` and
`default_models: dict[task, str]`. Built-ins register on import. Third parties can register explicitly or through a
`skai.backends` entry point.

### First-release providers
| Provider | Kind | Tasks |
|---|---|---|
| `dummy` | offline | all |
| `hf` | local open-source (transformers, sentence-transformers, diffusers) | classify text/image (prompted LLM or zero-shot/CLIP), summarize, translate, embed, describe image, transcribe (Whisper), TTS, generate image |
| `openai` | cloud | classify text/image, summarize, translate, embed, describe image, transcribe, TTS, generate image |
| `litellm` | cloud router (Anthropic, Gemini, Mistral, Bedrock, Azure, Ollama, …) | text and vision prompted tasks, embed |
| `replicate` | hosted open-source | talking avatar (e.g. SadTalker/LivePortrait-class models), generate image, TTS |

`openai` keeps a native adapter because it covers the most tasks (TTS, images, ASR) and the existing code already
uses it. Other prompted text/vision providers go through LiteLLM instead of separate native adapters until there is
demand. `TalkingAvatarGenerator` has no well-packaged local implementation, so its default is `replicate` with an
open-source model. That is the case the backends spec allows for.

### Prompted tasks share one adapter path
Classification, summarization, translation and description on LLM-style providers share a prompt-building
helper per task and one "chat/vision completion" call per backend. Classification asks for structured output
restricted to `classes_`. `predict_proba` uses token logprobs where the provider exposes them, and one-hot on
the parsed label otherwise (documented per backend). This keeps today's `k_shot` semantics.

### Batching and concurrency
`batch_size` (default per backend) groups samples. Cloud backends issue requests concurrently with a bounded
semaphore (`max_concurrency` on the backend object), and local backends run batches through the model. Results
are reassembled by index, which keeps input order. Calls run on an internal event loop, so `predict` stays
synchronous and also works inside Jupyter's running loop (via a worker thread).

### Media I/O
- Inputs are coerced per modality by `skai.io` into a provider-neutral form (PIL image; `Audio(samples, sample_rate)`;
  `Video` lazy file handle). URLs are fetched with a timeout.
- Outputs: `output_format="object"` → PIL / `Audio` / `Video`. `"bytes"` → PNG / WAV / MP4. `"path"` → files named
  `<component>-<uuid>.<ext>` in `output_dir` or a `tempfile.mkdtemp()` directory.
- Base dependencies add `numpy`, `Pillow`, `soundfile`. Video decoding/encoding (`av`) goes in the `video` extra.

### Dependencies and extras
Required dependencies: `scikit-learn`, `numpy`, `Pillow`, `soundfile`, `httpx`. Extras: `openai`, `hf` (torch,
transformers, sentence-transformers, diffusers, accelerate), `litellm`, `replicate`, `video`, `all`. A missing
extra raises `ImportError` naming `pip install scikit-ai[<extra>]`. `python-dotenv` is dropped, and loading `.env`
is left to the user.

## Risks / Trade-offs

- [The `hf` extra is heavy (torch), and local defaults download weights on first fit] → Document the sizes, use small default models, and never download at import time.
- [`predict_proba` quality differs widely across providers] → Document the method per backend, and in scenarios only test shape/normalization, not calibration.
- [Talking-avatar and image models on hosted providers change or disappear] → Pin model versions in defaults and keep defaults in one table that is easy to update.
- [Prompted outputs drift from the allowed labels] → Use structured output where available, validate against `classes_`, and fall back to `on_error` handling.
- [The breaking removal of `skai.llm` affects 0.1.0 users] → Release as 0.2.0 with a CHANGELOG migration note (`OpenAIClassifier` → `TextClassifier(backend="openai")`).
- [Generated media may be misused (voice cloning, avatars of real people)] → The docs state the user's responsibility and the providers' policies, and the library adds no watermark removal or safety bypass.

## Migration Plan

1. Land `update-copier-template`.
2. Implement on a branch in the task order. `skai.llm` is removed in the same release that adds `TextClassifier`.
3. Release 0.2.0. Rollback is pinning `scikit-ai==0.1.0`.

## Open Questions

- Exact default model per (provider, task). This is chosen during implementation by size, licence and quality, and recorded in the docs table.
