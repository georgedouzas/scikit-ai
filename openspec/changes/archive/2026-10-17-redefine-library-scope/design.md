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
  __init__.py        # front page docstring, re-exports nothing but subpackages' public names via subpackages
  core/              # shared leaves: _errors, _types, _io (coercion, Audio/Video, writers), _estimator (bases), _backend (Backend base, registry)
  backends/          # _dummy, _openai, _hf, _litellm, _replicate; re-exports backend classes and `register`
  text/              # _classifier, _summarizer, _translator, _embedder, _synthesizer, _image_generator
  image/             # _descriptor, _classifier
  audio/             # _transcriber, _avatar
```
Imports run downward: `core` → `backends` → `text`/`image`/`audio`. Each package `__init__` re-exports its own
public names with `__all__`, and `skai/__init__.py` keeps the docstring only, so users import from `skai.text`,
`skai.image`, `skai.audio`, `skai.backends` and `skai.core`. Named exceptions (`UnsupportedBackendError(ValueError)`,
`MissingExtraError(ImportError)`, `MissingCredentialError`, `MediaReadError(ValueError)`) live in `core._errors`
and are re-exported from `skai.core`. Organizing by task was the alternative. It was rejected because the user picks
a module by what data they hold.

### Components declare a task, backends implement tasks
Each component has a class-level `_task` (e.g. `"summarize"`), and a typed request/response contract lives with
it. `fit` resolves `backend` → backend instance (`backend_`), checks `_task in backend_.supported_tasks`, and
raises `ValueError` listing capable providers otherwise. Components build the task request (prompt template,
labels, examples, options). Backends only execute it. The alternative was one class per (component, provider)
pair, like today's `OpenAIClassifier`. That grows as components × providers and duplicates sklearn plumbing.

### Backend objects are estimators-lite; strings are sugar
`Backend` subclasses `sklearn.base.BaseEstimator` for `get_params`/`set_params`/`clone`, so
`backend__model` works in `GridSearchCV`. `"openai:gpt-5-mini"` is parsed by the registry into
`OpenAIBackend(model="gpt-5-mini")`, and `"openai"` gets the provider's per-task default model. Following the
constitution's credential rule, backends take `api_key_env` (the variable name) and read the secret only when the
client is created in `fit`, so no secret is ever a parameter. A user-supplied `client` object is used as is.

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
Classification, summarization, translation and description on LLM-style providers share one prompt builder per task
in `core._prompts` and one `complete(prompt, model)` method per backend (`PromptedBackend`). Classification asks for
labels only, and the answer is matched against `classes_`. Prompted backends score one for every named label and zero
for the others, so `predict_proba` is one-hot for them. Token log-probabilities were left out because providers expose
them inconsistently. The `hf` backend uses zero-shot NLI and CLIP models instead, which return real scores. This keeps
today's `k_shot` semantics.

### Default models are a documented exception
The constitution leaves the choice of model with the caller. scikit-learn-style defaults are kept on purpose: every
component and every provider-only string has a documented default model. The Project Profile records this
exception, and the caller can always override it with `backend`.

### Tests never reach the network
Provider adapters are tested against fake clients and recorded payloads. A `conftest.py` socket guard fails any
test that opens a connection. Real-provider runs are left to the user and are not part of the suite.

### Batching and concurrency
`batch_size` on the component groups samples. Each batch is passed to `Backend.run`, which runs the requests on a
thread pool of `max_concurrency` workers (one for the local `hf` backend) and returns a result or an exception per
request. Results are reassembled by index, which keeps input order, and `predict` stays synchronous, also inside
Jupyter. Threads were chosen over an event loop because provider SDKs and local pipelines are mostly synchronous.

### Media I/O
- `core._io` coerces inputs per modality into a provider-neutral form (PIL image, `Audio(samples, sample_rate)`,
  `Video`). URLs are fetched with `httpx` and a timeout.
- `Video` holds decoded frames or encoded MP4 bytes and converts lazily between them with PyAV, so the `video` extra is
  needed only when a conversion happens.
- Outputs: `output_format="object"` → PIL / `Audio` / `Video`. `"bytes"` → PNG / WAV / MP4. `"path"` → files named
  `<component>-<uuid>.<ext>`, opened in exclusive mode, in `output_dir` or a `tempfile.mkdtemp()` directory.
- Base dependencies add `numpy`, `Pillow`, `soundfile` and `httpx`.

### Dependencies and extras
Required dependencies: `scikit-learn`, `numpy`, `Pillow`, `soundfile`, `httpx`. Built-in backends are registered as
`skai.backends` entry points, so `core` never imports `backends`. Extras: `openai`, `hf` (torch,
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
