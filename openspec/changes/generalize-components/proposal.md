# Proposal

## Why

0.2 has one class per task and input type: `TextSummarizer`, `TextTranslator`, `ImageDescriptor`, `AudioDescriptor`,
`SpeechSynthesizer`, `ImageEditor` and so on. Many are the same operation with a different prompt, and the roadmap
would add dozens more (`QuizWriter`, `ScriptWriter`, `Storyboarder`, ...). That makes the catalog large and hard to
compose. The most useful results, such as dubbed videos and narrated explainers, also need shapes a linear pipeline
cannot express: several inputs per sample, one sample split into many, and many joined into one.

A component should be distinct only when it returns a different kind of output, plays a different scikit-learn role,
learns differently in `fit`, or needs a different model family. Prompt-only variations become parameters.

## What Changes

- **BREAKING** Replace the 21 classes of 0.2 with 10 general components, whose input type is a parameter:

| Component | Module | Input | Output |
|---|---|---|---|
| `Classifier` | `skai.classification` | text, image, audio, video, record | labels and probabilities |
| `TextGenerator` | `skai.generation` | text, image, audio, video, record | text |
| `Extractor` | `skai.extraction` | text, image, audio, video, record | records following `fields`, with nested lists |
| `Embedder` | `skai.embedding` | text, image, audio, video | vectors |
| `Transcriber` | `skai.transcription` | audio, video, image | text or timed records, optionally with speakers |
| `Segmenter` | `skai.segmentation` | image, video, audio, text | lists of regions in space or time |
| `ImageGenerator` | `skai.generation` | text, image | images (generate, edit, upscale, remove background) |
| `AudioGenerator` | `skai.generation` | text, audio | audio (speech, music, sound, convert voice, enhance, separate) |
| `VideoGenerator` | `skai.generation` | text, image, audio, record, video | videos (generate, animate, talking head, lip sync, reframe) |
| `Composer` | `skai.composition` | lists and records of media | one image, audio or video (concatenate, mix, mux, subtitle) |

- Add four pipeline primitives in `skai.pipeline`: `MediaUnion` (records of several inputs), `Pick` (one field),
  `ForEach` (every item of a list) and `Select` (the best items of a list).
- Add `skai.presets`, functions that return configured components for common tasks, and `skai.recipes`, functions that
  return 16 configured pipelines, from a brief to an explainer video to dubbing in the original voice.
- Keep every 0.2 name importable for 0.3 as a deprecated subclass, removed in 0.4.
- Replace the prompted backend tasks `summarize`, `translate` and `describe_*` with one `generate_text` task, and add
  the backend tasks of the new operations.
- Carry over the verification left open by `add-roadmap-components`: Sora, and the providers and models that need
  credentials or larger hardware.

## Capabilities

### New Capabilities
- `text-generation`: `TextGenerator` for every input type, with examples, a learned length and a glossary.
- `extraction`: `Extractor` for every input type, with nested and list fields.
- `segmentation`: `Segmenter` for regions in space and time.
- `composition`: `Composer` for local assembly of media.
- `pipeline-primitives`: `MediaUnion`, `Pick`, `ForEach` and `Select`.
- `presets`: `skai.presets` and `skai.recipes`.
- `deprecations`: the 0.2 names during 0.3.

### Modified Capabilities
- `classification`, `embedding`, `transcription`, `generation`: one general class each, with `input` and `operation`.
- `pipelines`: 16 recipes replace the chain-specific requirements.
- `module-layout`, `media-estimator-api`, `media-io`, `backends`: new names, records and list-valued samples, and
  backend coverage per operation.
- `description`, `text-processing`: retired, their requirements now in `text-generation` and `extraction`.

## Impact

- Public API: new classes and modules. The old names warn and keep working until 0.4. Released as 0.3.0.
- Backends: new tasks for the new operations, implemented where providers offer them, with `UnsupportedBackendError`
  elsewhere.
- Docs: the guide is reorganized around the 10 components, with a presets page and a recipes page.
