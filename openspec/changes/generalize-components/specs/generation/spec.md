# Spec Delta

## ADDED Requirements

### Requirement: Generation operations
Every media generator SHALL take an `operation` parameter that selects what it produces from its input, and SHALL raise `UnsupportedBackendError` at `fit`, listing the backends that support the operation, when its backend does not.

#### Scenario: Unsupported operation
- **WHEN** `AudioGenerator(operation='separate', backend='openai')` is fitted
- **THEN** `UnsupportedBackendError` is raised listing the backends that separate audio

### Requirement: AudioGenerator
The library SHALL provide `AudioGenerator` in `skai.generation`. Its `operation` SHALL be `'speech'` (default, text to speech in `voice`), `'music'` or `'sound'` (text to audio of `duration` seconds), `'convert_voice'` (audio to the same speech in `voice`), `'enhance'` (audio with noise removed) or `'separate'` (audio to a record with `vocals` and `accompaniment`).

#### Scenario: Speech and music
- **WHEN** `AudioGenerator().predict(['Hello'])` and `AudioGenerator(operation='music', duration=5).predict(['calm piano'])` run on `dummy`
- **THEN** they return `Audio` objects, the music 5 seconds long

#### Scenario: Separation returns stems
- **WHEN** `AudioGenerator(operation='separate', backend='dummy')` predicts on a song
- **THEN** it returns one record per song with the keys `vocals` and `accompaniment`, both `Audio`

### Requirement: Speech voice learned from samples
`AudioGenerator` SHALL accept one recording or a list of recordings of the target voice in `voice`, and `fit` SHALL learn `voice_` by joining them, for the `'speech'` and `'convert_voice'` operations. It SHALL raise `UnsupportedBackendError` at `fit` on backends that cannot clone voices.

#### Scenario: Voice learned from samples
- **WHEN** it is constructed with `voice=[recording_1, recording_2]` and fitted on a backend that clones voices
- **THEN** `voice_` is the joined reference audio and every speech request carries it

### Requirement: Image style learned from references
For `operation='generate'`, `ImageGenerator` SHALL learn `style_` in `fit` from `style_images`, written by the backend's image description, and apply it to every prompt, with an explicit `style` taking precedence. A backend that cannot describe images SHALL raise `UnsupportedBackendError` at `fit`.

#### Scenario: Learned style
- **WHEN** it is constructed with `style_images` holding three watercolor images and fitted on a backend that describes images
- **THEN** `style_` is set and every generation request carries it

### Requirement: Portrait prepared for talking heads
For `operation='talking_head'`, `VideoGenerator.fit` SHALL prepare `avatar_` from `avatar`: converted to RGB, center-cropped to a square, and resized to the backend's input size.

#### Scenario: Prepared portrait
- **WHEN** it is fitted with a 640×480 portrait on a backend whose input size is 512
- **THEN** `avatar_` is a 512×512 RGB image

## MODIFIED Requirements

### Requirement: ImageGenerator
The library SHALL provide `ImageGenerator` in `skai.generation`. Its `operation` SHALL be `'generate'` (default, text to image of `size` with optional `style` and `negative_prompt`), `'edit'` (image and `instructions`, optionally inside `mask`, to image), `'upscale'` (image to image `scale` times larger) or `'remove_background'` (image to image with a transparent background). `reference_images` SHALL keep one subject consistent across generated images on backends that support references.

#### Scenario: Requested size
- **WHEN** `ImageGenerator(size=(512, 512), backend="dummy").predict(["a cat"])` is called
- **THEN** it returns one 512×512 image

#### Scenario: Edit, upscale and background removal
- **WHEN** `ImageGenerator(operation='edit', instructions='add snow')`, `ImageGenerator(operation='upscale', scale=2)` and `ImageGenerator(operation='remove_background')` predict on a 64×48 image on `dummy`
- **THEN** they return a 64×48 image, a 128×96 image and an RGBA 64×48 image

### Requirement: VideoGenerator
The library SHALL provide `VideoGenerator` in `skai.generation`. Its `operation` SHALL be `'generate'` (default, text to video of `duration` seconds and `size`), `'animate'` (image to a video starting from it, with optional `instructions` for the motion), `'talking_head'` (audio to a video of `avatar` speaking it), `'lip_sync'` (a record with `video` and `audio` to the video speaking the new audio) or `'reframe'` (video to `aspect_ratio`, following the main subject).

#### Scenario: Requested duration and size
- **WHEN** `VideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit().predict(['a boat at sea'])` is called
- **THEN** it returns one video of 2 seconds whose frames are 320×240

#### Scenario: Lip sync from a record
- **WHEN** `VideoGenerator(operation='lip_sync', backend='dummy')` predicts on `{'video': clip, 'audio': speech}`
- **THEN** it returns one video as long as `speech`

#### Scenario: Reframe to vertical
- **WHEN** `VideoGenerator(operation='reframe', aspect_ratio=(9, 16), backend='dummy')` predicts on a 1280×720 video
- **THEN** it returns a video whose frames are 405×720

## REMOVED Requirements

### Requirement: SpeechSynthesizer
**Reason**: Speech is an operation of the audio generator.
**Migration**: Use `AudioGenerator(voice=...)`, whose default operation is `'speech'`.

### Requirement: TalkingAvatarGenerator
**Reason**: Talking heads are an operation of the video generator.
**Migration**: Use `VideoGenerator(operation='talking_head', avatar=...)`.

### Requirement: ImageEditor
**Reason**: Edits are an operation of the image generator.
**Migration**: Use `ImageGenerator(operation='edit', instructions=..., mask=...)`.

### Requirement: ImageAnimator
**Reason**: Animation is an operation of the video generator.
**Migration**: Use `VideoGenerator(operation='animate', instructions=...)`.

### Requirement: Synthesizer learns a voice from samples
**Reason**: The requirement now names `AudioGenerator`.
**Migration**: See "Speech voice learned from samples".

### Requirement: Image generator learns a style from references
**Reason**: The requirement is restated for the operations of `ImageGenerator`.
**Migration**: See "Image style learned from references".

### Requirement: Avatar generator prepares the portrait
**Reason**: The requirement now names `VideoGenerator`.
**Migration**: See "Portrait prepared for talking heads".
