# Design

## Context

Components are scikit-learn estimators built on `BaseClassifier`, `BaseGenerator`, `BaseSupervisedGenerator`,
`BaseMediaGenerator` and `BaseEmbedder`. Each class fixes its `task` and `input_kind` as class variables, and backends
implement tasks as methods. The archived `add-roadmap-components` change verified most components on OpenAI, LiteLLM
and small Hugging Face models. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- Ten components whose behavior depends on parameters, not on class names, over the existing bases and backends.
- Primitives that keep every pipeline row one sample, so scikit-learn tools keep working.
- No loss of the learning, verification and backend coverage that 0.2 has.

**Non-Goals:**
- A pipeline framework or graph runner beyond scikit-learn `Pipeline`.
- Removing the 0.2 names before 0.4.

## Decisions

### `input` and `operation` resolve the backend task at `fit`
`task` and `input_kind` stop being class variables. Every component computes them from its parameters at `fit`:
`Classifier(input='image')` uses `classify_image`, and `ImageGenerator(operation='upscale')` uses `upscale_image`. Then
it resolves the backend and checks support exactly as today, so `UnsupportedBackendError` keeps listing capable
backends. `input` defaults to the most common type of each component, so text for the text-producing ones and audio for
`Transcriber`. There is no automatic detection, because a string can be text, a path or a URL.

| Component | `input` / `operation` → backend task |
|---|---|
| `Classifier` | `classify_text`, `classify_image`, `classify_audio`, `classify_video`, `classify_record` |
| `TextGenerator` | `generate_text` (one task, the request carries the input type) |
| `Extractor` | `extract_fields` (the request carries the input type) |
| `Embedder` | `embed_text`, `embed_image`, `embed_audio`, and video as the mean of `embed_image` over frames |
| `Transcriber` | `transcribe` (audio, video), `transcribe_image`, plus `supports_speakers` |
| `Segmenter` | `detect_objects`, `segment_image`, `diarize` (speakers), `find_moments` (prompt), and local shots, silence and text splitting |
| `ImageGenerator` | `generate_image`, `edit_image`, `upscale_image`, `remove_background` |
| `AudioGenerator` | `synthesize_speech`, `generate_music`, `generate_sound`, `convert_voice`, `enhance_audio`, `separate_audio` |
| `VideoGenerator` | `generate_video`, `animate_image`, `animate_avatar`, `lip_sync`, and `reframe` built locally on `detect_objects` |
| `Composer` | none, local with PyAV, NumPy and Pillow |

`summarize`, `translate` and the three `describe_*` tasks merge into `generate_text`. `PromptedBackend` builds the
prompt from the instructions, the examples, the glossary and the media of the request, so every prompted backend
supports every input type it accepts (images on vision models, audio where `accepts_audio`). Third-party backends of
0.2 that implement the old tasks keep working in 0.3: the base `generate_text` falls back to them when the request
matches. That fallback is removed in 0.4.

### One base per scikit-learn role, input types as readers
The bases stay: classifier, supervised generator, media generator and embedder. Reading is chosen by `input` through
the existing `READERS` table, which gains `'record'` (a dict whose values are read by their own types). Video frame
sampling moves into the reader of video for every component that analyzes frames, so `n_frames` is a parameter of
`Classifier`, `TextGenerator`, `Extractor` and `Embedder`.

### Records and lists are plain Python values
A record is a `dict`, and a list-valued sample is a `list`, both stored in the usual one-dimensional object array, so
`Pipeline`, `GridSearchCV`, `cross_val_score` and `Pipeline(memory=...)` work unchanged. `MediaUnion` is a
`TransformerMixin` meta-estimator modeled on `FeatureUnion`, with named branches and `get_params` routing
(`union__audio__voice`). It returns records instead of concatenated columns. `ForEach` flattens the items of all
samples, runs the inner estimator once on the flat list (one batch to the backend), and splits the outputs back by
the original lengths. `Select` scores items with a callable or with a fitted classifier's `predict_proba` column.

### `Composer` and local segmentation run without a backend
Concatenation, mixing, muxing and burned subtitles are deterministic media operations on `Video` and `Audio`. They use
PyAV for encoding and Pillow for drawing subtitles, with the `video` extra. Mixing ducks tracks with a smoothed RMS
envelope of the voice track. Shot cuts use the histogram distance between consecutive frames with a threshold,
silence uses an RMS threshold, and text splitting uses paragraph, sentence or fixed-size chunks. These run locally
because no model adds value there.

### Presets and recipes are functions, not classes
`skai.presets.summarizer(max_length=50, **kwargs)` returns `TextGenerator(instructions=..., max_length=50, **kwargs)`,
so presets add readable names without adding classes. Their instruction strings live in one module-level table, which
the docs render as the recipes of instructions. `skai.recipes` functions return `Pipeline` objects built only from
public components and primitives, and every recipe has an executed docs example and a `dummy` test.

### Deprecation shims
The 0.2 classes become thin subclasses in their 0.2 modules, for example `class ImageClassifier(Classifier)` whose
`__init__` sets `input='image'` and emits one `FutureWarning`. `skai.text` and `skai.description` keep only shims.
`get_params` of a shim returns its 0.2 parameters, so cloned and tuned shims keep working.

### Default models of new operations
The defaults are verified one operation at a time with the end-to-end suite, as in `add-roadmap-components`. OpenAI
covers `generate_text`, `extract_fields`, `transcribe_image` (vision prompt), `detect_objects` (vision prompt with
boxes), `edit_image`, speech and Sora. Hugging Face covers `segment_image` (SAM-class models), `detect_objects`
(OWL-ViT), `diarize` (pyannote), `separate_audio` (Demucs), `enhance_audio`, `upscale_image` (Swin2SR),
`remove_background` (RMBG), `generate_music` (MusicGen) and `lip_sync`. Replicate is added back only for operations
whose model inputs are confirmed by a real run.

## Risks / Trade-offs

- [A second breaking release right after 0.2] → Shims keep 0.2 code working through 0.3, and the migration page maps every name.
- [Generic classes are less discoverable than task names] → Presets give task names, and the docs open with a table of tasks and the configuration of each.
- [Records and lists in object arrays are opaque to `set_output('pandas')`] → `Extractor.transform` keeps returning columns, and the docs show `Pick` to get a field.
- [Local media assembly needs PyAV, and long videos use much memory] → `Composer` requires the `video` extra at `fit` with a clear error, and the docs state the memory limit of the first version.
- [Many new operations depend on models that are hard to run here] → Each operation is implemented only with a provider whose API is known, verified end to end where possible, and listed as open in tasks otherwise.
