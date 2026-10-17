# Spec Delta

## ADDED Requirements

### Requirement: Synthesizer learns a voice from samples
`SpeechSynthesizer` SHALL accept one recording or a list of recordings of the target voice in its `voice` parameter. `fit` SHALL learn `voice_` by joining them into one reference, and SHALL clone that voice at prediction. It SHALL raise `UnsupportedBackendError` at `fit` on backends that cannot clone voices.

#### Scenario: Voice learned from samples
- **WHEN** it is constructed with `voice=[recording_1, recording_2]` and fitted on a backend that clones voices
- **THEN** `voice_` is the joined reference audio and every speech request carries it

### Requirement: Image generator learns a style from references
`ImageGenerator` SHALL accept reference images in a `style_images` parameter, and `fit` SHALL learn `style_`, a short description of the visual style that the reference images share, written by the backend's image description task, and SHALL apply it to every prompt. An explicit `style` parameter SHALL take precedence over `style_`.

#### Scenario: Learned style
- **WHEN** it is constructed with `style_images` holding three watercolor images and fitted on a backend that describes images
- **THEN** `style_` is set and every generation request carries it

#### Scenario: Backend without description
- **WHEN** the backend cannot describe images
- **THEN** `UnsupportedBackendError` is raised at `fit`, listing the backends that can

### Requirement: Avatar generator prepares the portrait
`TalkingAvatarGenerator.fit` SHALL prepare `avatar_` from `avatar`: converted to RGB, center-cropped to a square around the image center, and resized to the backend's input size.

#### Scenario: Prepared portrait
- **WHEN** it is fitted with a 640×480 portrait on a backend whose input size is 512
- **THEN** `avatar_` is a 512×512 RGB image
