# generation Specification

## Purpose
Components that generate images, audio or video from their inputs, importable from `skai.generation`.

## Requirements

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

### Requirement: VideoGenerator
The library SHALL provide `VideoGenerator` in `skai.generation`, a generator creating one video per text prompt, with `duration` in seconds, `size` as `(width, height)`, optional `fps`, `style` and `negative_prompt`, and the `output_format` and `output_dir` parameters of media generators.

#### Scenario: Requested duration and size
- **WHEN** `VideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit().predict(['a boat at sea'])` is called
- **THEN** it returns one video of 2 seconds whose frames are 320×240

### Requirement: ImageEditor
The library SHALL provide `ImageEditor` in `skai.generation`, a generator that applies one fixed `instruction` (required) to every image, such as `'make it night'`, and returns one edited image per input image. An optional `mask` limits the edit to a region on backends that support masks.

#### Scenario: Edit images
- **WHEN** `ImageEditor(instruction='add snow', backend='dummy').fit().predict([img1, img2])` is called
- **THEN** it returns two images with the sizes of their inputs

#### Scenario: Missing instruction
- **WHEN** it is fitted without `instruction`
- **THEN** `InvalidParameterError` is raised

#### Scenario: Mask unsupported
- **WHEN** `mask` is given and the backend does not support masks
- **THEN** `ValueError` is raised at `fit` listing the backends that do

### Requirement: ImageAnimator
The library SHALL provide `ImageAnimator` in `skai.generation`, a generator that turns every image into a short video that starts from it, with an optional fixed `motion_prompt`, `duration` and `fps`.

#### Scenario: Animate images
- **WHEN** `ImageAnimator(motion_prompt='slow zoom', duration=2, backend='dummy').fit().predict([image])` is called
- **THEN** it returns one video of 2 seconds whose first frame matches the image size
