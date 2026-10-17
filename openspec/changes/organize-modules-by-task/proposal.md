# Proposal

## Why

Components are grouped by input type (`skai.text`, `skai.image`, `skai.audio`). That puts generators where users don't
look for them (`ImageGenerator` and `SpeechSynthesizer` in `skai.text`) and gives components with several inputs no
natural home (`TalkingAvatarGenerator` takes a portrait and audio and returns video, yet sits in `skai.audio`). The
module also repeats what the class name already says. scikit-learn groups estimators by task family (`sklearn.cluster`,
`sklearn.decomposition`), and 0.2 is unreleased, so this is the cheapest moment to follow it.

## What Changes

- **BREAKING** Move components into task-family modules:
  - `skai.classification`: `TextClassifier`, `ImageClassifier`
  - `skai.description`: `ImageDescriptor`
  - `skai.transcription`: `SpeechTranscriber`
  - `skai.text`: `TextSummarizer`, `TextTranslator`
  - `skai.embedding`: `TextEmbedder`
  - `skai.generation`: `SpeechSynthesizer`, `ImageGenerator`, `TalkingAvatarGenerator`
- **BREAKING** Remove `skai.image` and `skai.audio`, and narrow `skai.text` to text-to-text components.
- Re-home the component requirements in task-family specs, unchanged.
- Update tests, docs, README and the constitution's Project Profile.

## Capabilities

### New Capabilities
- `module-layout`: where components are imported from and how they are named.
- `classification`: components that assign labels (moved from text and image components).
- `description`: components that describe media (moved from image components).
- `transcription`: components that transcribe speech (moved from audio components).
- `text-processing`: text-to-text components (moved from text components).
- `embedding`: components that return vectors (moved from text components).
- `generation`: components that generate media (moved from text and audio components).

### Modified Capabilities
- `text-components`, `image-components`, `audio-components`: every requirement is removed and moved unchanged to a task-family capability. The emptied specs are deleted after archive.

## Impact

- Public import paths change. Component behavior, parameters and backends do not.
- Tests move from `tests/{text,image,audio}` to one folder per task family.
