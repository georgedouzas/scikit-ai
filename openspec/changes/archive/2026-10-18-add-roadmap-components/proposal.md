# Proposal

## Why

The first release covers ten components. The catalog still has gaps in each task family: no audio or video
classification, no audio or video description, no transcription of videos, no structured extraction, no image or audio
embeddings, and no video generation or image editing. The components are also more useful chained than alone, but
nothing specifies which chains work or how pipelines behave when steps generate media.

## What Changes

- Add eleven components, each in its task family:
  - `skai.classification`: `AudioClassifier`, `VideoClassifier`
  - `skai.description`: `AudioDescriptor`, `VideoDescriptor`
  - `skai.transcription`: `VideoTranscriber`
  - `skai.text`: `TextExtractor`
  - `skai.embedding`: `ImageEmbedder`, `AudioEmbedder`
  - `skai.generation`: `VideoGenerator`, `ImageEditor`, `ImageAnimator`
- Add video frame sampling (`n_frames`) and audio track extraction to media I/O.
- Require open-source and cloud coverage for every component, not only the first release.
- Add a `pipelines` capability: type compatibility between steps, eight supported chains, caching, tuning and the cost
  of fitting.
- Some catalog entries become pipelines instead of classes: a video summary is `VideoTranscriber` → `TextSummarizer`,
  and speech translation is `SpeechTranscriber` → `TextTranslator` → `SpeechSynthesizer`.

### Out of scope

- `LipSyncer` and `VideoDubber`. Both need two inputs that vary per sample (a video and new speech), which a linear
  scikit-learn pipeline cannot pass. They need their own design.
- `ObjectDetector`, `VoiceConverter`, which stay deferred.

## Capabilities

### New Capabilities
- `pipelines`: how components compose in scikit-learn pipelines and which chains are supported.

### Modified Capabilities
- `classification`: adds `AudioClassifier` and `VideoClassifier`.
- `description`: adds `AudioDescriptor` and `VideoDescriptor`.
- `transcription`: adds `VideoTranscriber`.
- `text-processing`: adds `TextExtractor` and its feature-column output.
- `embedding`: adds `ImageEmbedder`, `AudioEmbedder` and a shared text and image space.
- `generation`: adds `VideoGenerator`, `ImageEditor` and `ImageAnimator`.
- `media-io`: adds video frame sampling and audio track extraction.
- `backends`: open-source and cloud coverage applies to every component.
- `module-layout`: lists the new components in their modules.

## Impact

- New backend tasks and request types in `skai.core`, implemented by the providers in `skai.backends`.
- New optional dependencies for video models in the `hf` extra. Video components need the `video` extra.
- Tests for every component and every pipeline recipe on the `dummy` backend, plus fake-client tests per provider.
