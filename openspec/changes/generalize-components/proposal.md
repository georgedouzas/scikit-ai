# Proposal

## Why

0.2 has one class per task: `TextSummarizer`, `TextTranslator`, `ImageDescriptor`, `SpeechTranscriber`,
`ImageEditor` and so on. Many are the same operation with a different prompt, and the roadmap would add dozens more
(`QuizWriter`, `ScriptWriter`, `Storyboarder`, ...). The class names mix three axes (task, input and output), so they
are hard to guess and hard to compose. The most useful results, such as dubbed videos and narrated explainers, also
need shapes a linear pipeline cannot express: several inputs per sample, one sample split into many, and many joined
into one.

## What Changes

- **BREAKING** Replace the task-named classes of 0.2 with a fixed grid named by input and output, with no aliases:
  - Analyzers are `{Input}{Role}`, with `Input` one of `Text`, `Image`, `Audio`, `Video` or `Multimodal` and
    `Role` one of `Classifier`, `Extractor`, `Embedder` or `Segmenter`.
  - Generators are `{Input}To{Output}Generator`, with `Output` one of `Text`, `Image`, `Audio` or `Video`.
  - Tasks that are prompts become `instructions`, and model families or procedures become `operation`.
- The grid:

| Module | Classes |
|---|---|
| `skai.classification` | `TextClassifier`, `ImageClassifier`, `AudioClassifier`, `VideoClassifier`, `MultimodalClassifier` |
| `skai.extraction` | `TextExtractor`, `ImageExtractor`, `AudioExtractor`, `VideoExtractor`, `MultimodalExtractor` |
| `skai.embedding` | `TextEmbedder`, `ImageEmbedder`, `AudioEmbedder`, `VideoEmbedder` |
| `skai.segmentation` | `TextSegmenter`, `ImageSegmenter`, `AudioSegmenter`, `VideoSegmenter` |
| `skai.generation` | `TextToTextGenerator`, `ImageToTextGenerator`, `AudioToTextGenerator`, `VideoToTextGenerator`, `MultimodalToTextGenerator`, `TextToImageGenerator`, `ImageToImageGenerator`, `VideoToImageGenerator`, `MultimodalToImageGenerator`, `TextToAudioGenerator`, `AudioToAudioGenerator`, `VideoToAudioGenerator`, `MultimodalToAudioGenerator`, `TextToVideoGenerator`, `ImageToVideoGenerator`, `AudioToVideoGenerator`, `VideoToVideoGenerator`, `MultimodalToVideoGenerator` |
| `skai.pipeline` | `MediaUnion`, `Field`, `ForEach`, `TopK`, `Retriever` |

- Add multimodal samples (dicts or DataFrame rows of named media) and list-valued samples, built by `MediaUnion` and
  the segmenters and consumed by `ForEach`, `TopK` and the joining operations.
- Add `skai.presets` (configured components for common tasks) and `skai.recipes` (16 configured pipelines, from a brief
  to an explainer video to dubbing in the original voice).
- Remove `skai.text`, `skai.description` and `skai.transcription`, and the 0.2 names `TextSummarizer`, `TextTranslator`,
  `ImageDescriptor`, `AudioDescriptor`, `VideoDescriptor`, `SpeechTranscriber`, `VideoTranscriber`, `SpeechSynthesizer`,
  `ImageGenerator`, `ImageEditor`, `ImageAnimator`, `VideoGenerator` and `TalkingAvatarGenerator`. The classifiers,
  `TextExtractor` and the text, image and audio embedders keep their names.
- Replace the prompted backend tasks `summarize`, `translate` and `describe_*` with one `generate_text` task, and add
  the backend tasks of the new operations.
- Carry over the verification left open by `add-roadmap-components`: Sora, and the providers and models that need
  credentials or larger hardware.

## Capabilities

### New Capabilities
- `text-generation`: the five text generators, with the instruct and transcribe operations.
- `extraction`: the five extractors, with nested and list fields.
- `segmentation`: the four segmenters, with regions in space and time.
- `pipeline-primitives`: `MediaUnion`, `Field`, `ForEach`, `TopK` and `Retriever`.
- `presets`: `skai.presets` and `skai.recipes`.

### Modified Capabilities
- `classification`: adds `MultimodalClassifier`.
- `embedding`: adds `VideoEmbedder`.
- `generation`: the thirteen media generators and their operations.
- `pipelines`: 16 recipes replace the chain-specific requirements.
- `module-layout`, `media-estimator-api`, `media-io`, `backends`: the naming grid, multimodal and list-valued samples,
  and backend coverage per operation.
- `description`, `text-processing`, `transcription`: retired, their requirements now in `text-generation`,
  `extraction` and `segmentation`.

## Impact

- Public API: a breaking release, 0.3.0, with the migration page mapping every removed name.
- Backends: new tasks for the new operations, implemented where providers offer them, with `UnsupportedBackendError`
  elsewhere. Custom backends that implement the 0.2 prompted tasks must implement `generate_text`.
- Docs: the guide is reorganized around the grid, with a presets page and a recipes page.
