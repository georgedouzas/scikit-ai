# Spec Delta

## Purpose

Components that generate images, audio or video from their inputs, importable from `skai.generation`.

## ADDED Requirements

### Requirement: SpeechSynthesizer
The library SHALL provide `SpeechSynthesizer`, a generator converting each text into spoken audio, with `voice` (a provider voice name or a reference audio sample for cloning on supporting backends) and optional `language`.

#### Scenario: One audio per text
- **WHEN** `SpeechSynthesizer(backend="dummy").predict(["Hello", "Bye"])` is called
- **THEN** it returns two audio outputs in the selected `output_format`

#### Scenario: Voice cloning unsupported
- **WHEN** `voice` is a reference audio sample and the backend does not support cloning
- **THEN** `ValueError` is raised at `fit` listing the backends that do

### Requirement: ImageGenerator
The library SHALL provide `ImageGenerator`, a generator creating one image per text prompt, with `size` as `(width, height)`, optional `style` and optional `negative_prompt`.

#### Scenario: Requested size
- **WHEN** `ImageGenerator(size=(512, 512), backend="dummy").predict(["a cat"])` is called
- **THEN** it returns one 512×512 image

### Requirement: TalkingAvatarGenerator
The library SHALL provide `TalkingAvatarGenerator`, a generator that animates a fixed portrait `avatar` (any accepted image input, required) to speak each audio sample, returning one lip-synced video per sample with the audio track included.

#### Scenario: One video per audio
- **WHEN** `TalkingAvatarGenerator(avatar="me.png", backend="dummy").fit().predict(["hello.wav", "bye.wav"])` is called
- **THEN** it returns two videos whose durations match their audio inputs

#### Scenario: Missing avatar
- **WHEN** it is fitted without `avatar`
- **THEN** `InvalidParameterError` is raised

#### Scenario: Speech synthesized then animated
- **WHEN** a pipeline of `SpeechSynthesizer` followed by `TalkingAvatarGenerator` predicts on texts
- **THEN** it returns one video per text
