# Design

## Context

Components are scikit-learn estimators built on `BaseClassifier`, `BaseGenerator`, `BaseSupervisedGenerator`,
`BaseMediaGenerator` and `BaseEmbedder`. Each class fixes its `task` and `input_kind` as class variables, and backends
implement tasks as methods. The archived `add-roadmap-components` change verified most components on OpenAI, LiteLLM
and small Hugging Face models. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- A fixed grid of classes named by input and output, thin over the existing bases and backends.
- Primitives that keep every pipeline row one sample, so scikit-learn tools keep working.
- No loss of the learning, verification and backend coverage that 0.2 has.

**Non-Goals:**
- A pipeline framework or graph runner beyond scikit-learn `Pipeline`.

## Decisions

### The class name fixes input and output, `operation` the procedure, `instructions` the content
A class declares its input kind, its output kind and its valid operations as class variables, and its own parameters
in its signature. So `get_params`, grid searches and documentation show only what applies, and `predict` always returns
one type. There is no `input` parameter and no automatic detection. A string sample is text in text classes and a path
or URL in media classes. `operation` is validated at `fit` and selects the backend task. It never depends on the
wording of `instructions`.

| Class | `operation` → backend task |
|---|---|
| classifiers | `classify_text`, `classify_image`, `classify_audio`, `classify_video`, `classify_multimodal` |
| extractors | `extract_fields` (the request carries the input kind) |
| embedders | `embed_text`, `embed_image`, `embed_audio`, and video as the mean of `embed_image` over frames |
| `TextSegmenter` | local |
| `ImageSegmenter` | `detect_objects` (`by='object'`), `segment_image` (`by='mask'`) |
| `AudioSegmenter` | `transcribe` with segments (`speech`), `diarize` (`speaker`), `find_moments` (`moment`), local (`silence`) |
| `VideoSegmenter` | local (`shot`), `find_moments` (`moment`) |
| `*ToTextGenerator` | `generate_text` (`instruct`), `transcribe` (audio and video), `transcribe_image` (images) |
| `TextToImageGenerator` | `generate_image` |
| `ImageToImageGenerator` | `edit_image`, `upscale_image`, `remove_background`, local `collage` |
| `VideoToImageGenerator` | `pick_thumbnail`, local `frame` |
| `MultimodalToImageGenerator` | `edit_image` with per-sample `instructions` and `mask` |
| `TextToAudioGenerator` | `synthesize_speech`, `generate_music`, `generate_sound` |
| `AudioToAudioGenerator` | `separate_audio`, `enhance_audio`, `convert_voice`, local `concatenate` |
| `VideoToAudioGenerator` | local `extract` |
| `MultimodalToAudioGenerator` | local `mix`, `synthesize_speech` with a per-sample `voice` |
| `TextToVideoGenerator` | `generate_video` |
| `ImageToVideoGenerator` | `animate_image` |
| `AudioToVideoGenerator` | `animate_avatar` |
| `VideoToVideoGenerator` | `reframe` built on `detect_objects`, local `concatenate` |
| `MultimodalToVideoGenerator` | `lip_sync`, local `mux` and `subtitle` |

`summarize`, `translate` and the three `describe_*` tasks merge into `generate_text`. `PromptedBackend` builds the
prompt from the instructions, the examples, the glossary and the media of the request, so every prompted backend
supports every input kind it accepts (images on vision models, audio where `accepts_audio`).

### Thin classes over the existing bases
The bases stay, one per scikit-learn role: classifier, extractor, embedder, segmenter, supervised text generator and
media generator. Each public class is a few lines that set the input kind, output kind, operations and parameters,
which keeps the grid regular without duplicating logic. The readers gain `'multimodal'`, a dict or DataFrame row whose
values are read by their own kinds, and video frame sampling moves into the video reader, so every video class has
`n_frames`.

### Multimodal and list-valued samples are plain Python values
A multimodal sample is a `dict`, and a list-valued sample is a `list`. Both are stored in the usual one-dimensional
object array, so `Pipeline`, `GridSearchCV`, `cross_val_score` and `Pipeline(memory=...)` work unchanged. A DataFrame
passed to a multimodal class is read row by row. Every operation declares its required keys and checks them per sample.

### Primitives modeled on scikit-learn meta-estimators
`MediaUnion` is modeled on `FeatureUnion`, with named branches and `<branch>__<parameter>` routing, and it returns
dicts instead of concatenated columns. `Field` is a stateless transformer. `ForEach` flattens the items of all samples,
runs the inner estimator once on the flat list (one batch to the backend), and splits the outputs back by the original
lengths. `TopK` scores items with a callable or with a fitted classifier's `predict_proba` column.

### Per-sample parameters are multimodal keys
A parameter that changes per sample, such as the voice of every speaker turn or the edit text of every image, is a
key of a multimodal sample, never a constructor parameter. `MultimodalToAudioGenerator(operation='speech')` speaks
`text` in `voice`, and `MultimodalToImageGenerator(operation='edit')` edits `image` with `instructions` inside an
optional `mask`, so recipes build those samples with `MediaUnion` and `Field`.

### Retrieval
`Retriever` is the one primitive with a query method. `fit` runs the item pipeline on the samples, flattens list
outputs with their sample index, and indexes their vectors. `kneighbors` embeds queries with the query pipeline and
returns the records of the nearest items, so a text query finds the shots of a video library.

### Local operations
Concatenation, mixing, muxing, subtitles, collages, frames and audio extraction are deterministic media operations on
`Video`, `Audio` and PIL images, with PyAV and Pillow from the `video` extra and no backend. Mixing ducks tracks with a
smoothed RMS envelope of the voice track, and a track can be a list of timed records placed at their `start`. Shot
cuts use the histogram distance between consecutive frames, silence an RMS threshold, and text splitting paragraphs,
sentences or fixed-size chunks.

### Presets and recipes are functions
`skai.presets.summarizer(max_length=50, **kwargs)` returns
`TextToTextGenerator(instructions=..., max_length=50, **kwargs)`. The instruction strings live in one module-level
table, which the docs render. `skai.recipes` functions return `Pipeline` objects built only from public classes and
primitives, and every recipe has an executed docs example and a `dummy` test.

### No deprecation shims
0.3 removes the 0.2 task names outright. The release is pre-1.0, the grid keeps the classifier and embedder names that
most 0.2 code uses, and the migration page maps every removed name to its replacement.

### Default models of new operations
Defaults are verified one operation at a time with the end-to-end suite, as in `add-roadmap-components`. OpenAI covers
`generate_text`, `extract_fields`, `transcribe_image` and `detect_objects` (vision prompts), `edit_image`, speech and
Sora. Hugging Face covers `segment_image` (SAM-class models), `detect_objects` (OWL-ViT), `diarize` (pyannote),
`separate_audio` (Demucs), `enhance_audio`, `upscale_image` (Swin2SR), `remove_background` (RMBG) and
`generate_music` (MusicGen). Replicate is added back only for operations whose model inputs are confirmed by a real run.

### Providers of the operations without a transformers pipeline
- Background removal uses `briaai/RMBG-1.4`, which loads only with `trust_remote_code=True`. The backend pins the
  model revision to a commit hash, so the code that runs is fixed, and a different revision must be set explicitly.
- Lip sync and voice conversion run open-source models hosted on Replicate. A component declares a default backend
  per operation, so `MultimodalToVideoGenerator(operation='lip_sync')` defaults to Replicate while `mux` and
  `subtitle` stay local.
- Diarization uses `pyannote/speaker-diarization-3.1` from a new `diarize` extra, with the Hugging Face token read
  from the environment variable named by `token_env`, and speaker turns get their text from Whisper.
- Audio enhancement on the Hugging Face backend is spectral-gating noise reduction with NumPy, with no model.

## Risks / Trade-offs

- [A second breaking release right after 0.2, without shims] → The grid keeps the most used 0.2 names, and the migration page maps every removed name with its replacement code.
- [Forty classes look like many] → They form one grid, so names can be guessed, and the docs open with the grid and a table of tasks with their configuration.
- [Records and lists in object arrays are opaque to `set_output('pandas')`] → `Extractor.transform` keeps returning columns, and the docs show `Pick` to get a field.
- [Local media assembly needs PyAV, and long videos use much memory] → The local operations require the `video` extra at `fit` with a clear error, and the docs state the memory limit of the first version.
- [Pinned remote code of RMBG still runs downloaded code] → The revision is pinned and documented, and no other model loads remote code.
- [Many new operations depend on models that are hard to run here] → Each operation is implemented only with a provider whose API is known, verified end to end where possible, and listed as open in tasks otherwise.
