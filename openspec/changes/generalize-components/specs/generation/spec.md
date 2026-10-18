# Spec Delta

## ADDED Requirements

### Requirement: Media generators for every input and output
The library SHALL provide in `skai.generation` the generators `TextToImageGenerator`, `ImageToImageGenerator`, `VideoToImageGenerator`, `TextToAudioGenerator`, `AudioToAudioGenerator`, `VideoToAudioGenerator`, `MultimodalToAudioGenerator`, `TextToVideoGenerator`, `ImageToVideoGenerator`, `AudioToVideoGenerator`, `VideoToVideoGenerator` and `MultimodalToVideoGenerator`. Each SHALL return one output of the type its name ends with, in the requested `output_format`.

#### Scenario: Output type follows the name
- **WHEN** each media generator predicts on one sample of its input type on `dummy`
- **THEN** generators ending in `ImageGenerator` return PIL images, in `AudioGenerator` return `Audio`, and in `VideoGenerator` return `Video`

### Requirement: Generator operations
Every generator SHALL take an `operation` from its own set, with a default, and SHALL validate it at `fit`.

#### Scenario: Operations per generator
- **WHEN** a user reads the `operation` values of every generator
- **THEN** they are, default first: `TextToImageGenerator` `generate`; `ImageToImageGenerator` `edit`, `upscale`, `remove_background`, `collage`; `VideoToImageGenerator` `thumbnail`, `frame`; `TextToAudioGenerator` `speech`, `music`, `sound`; `AudioToAudioGenerator` `separate`, `enhance`, `convert_voice`, `concatenate`; `VideoToAudioGenerator` `extract`; `MultimodalToAudioGenerator` `mix`; `TextToVideoGenerator` `generate`; `ImageToVideoGenerator` `animate`; `AudioToVideoGenerator` `talking_head`; `VideoToVideoGenerator` `reframe`, `concatenate`; `MultimodalToVideoGenerator` `lip_sync`, `mux`, `subtitle`

#### Scenario: Invalid operation
- **WHEN** `TextToAudioGenerator(operation='edit')` is fitted
- **THEN** `InvalidParameterError` lists the operations of `TextToAudioGenerator`

#### Scenario: Unsupported operation
- **WHEN** an operation is set on a backend that does not support it
- **THEN** `UnsupportedBackendError` is raised at `fit`, listing the backends that do

### Requirement: Local operations without a backend
The operations `collage`, `frame`, `concatenate`, `extract`, `mix`, `mux` and `subtitle` SHALL run locally with the `video` extra, without calling any backend or needing credentials.

#### Scenario: Offline assembly
- **WHEN** `VideoToVideoGenerator(operation='concatenate')` is fitted with `backend=None` and predicts on a list of three clips
- **THEN** it returns one video of the summed durations without importing any provider

### Requirement: Image operations
`TextToImageGenerator` SHALL create one image of `size` per prompt, with optional `style`, `negative_prompt` and `reference_images` that keep one subject consistent. `ImageToImageGenerator` SHALL edit with `instructions` inside an optional `mask`, upscale by `scale`, remove the background into a transparent image, and lay a list of images out as a collage. `VideoToImageGenerator` SHALL return a thumbnail chosen by the backend or the frame at `at` seconds.

#### Scenario: Image operations keep or set sizes
- **WHEN** `ImageToImageGenerator(operation='edit', instructions='add snow')`, `operation='upscale', scale=2` and `operation='remove_background'` predict on a 64×48 image on `dummy`
- **THEN** they return a 64×48 image, a 128×96 image and an RGBA 64×48 image

### Requirement: Audio from text
`TextToAudioGenerator` SHALL produce speech in `voice` and `language`, or music or sound of `duration` seconds.

#### Scenario: Music length
- **WHEN** `TextToAudioGenerator(operation='music', duration=5, backend='dummy')` predicts on `['calm piano']`
- **THEN** it returns one `Audio` of 5 seconds

### Requirement: Audio from audio and video
`AudioToAudioGenerator` SHALL return the `stem` (`vocals` or `accompaniment`) of a separation, a denoised copy, the same speech in `voice`, or a list of audio joined with `gap` seconds between. `VideoToAudioGenerator` SHALL return the mono audio track of a video. `MultimodalToAudioGenerator` SHALL mix named tracks with `volumes`, lowering the tracks in `duck_under` while the track `duck_for` is not silent.

#### Scenario: Separated vocals
- **WHEN** `AudioToAudioGenerator(operation='separate', stem='vocals', backend='dummy')` predicts on a song
- **THEN** it returns one `Audio` as long as the song

#### Scenario: Ducked music
- **WHEN** `MultimodalToAudioGenerator(duck_under=['music'], duck_for='voice')` mixes `{'voice': speech, 'music': song}`
- **THEN** the output is as long as the longer track and the music is quieter where the voice speaks

### Requirement: Video from text, images and audio
`TextToVideoGenerator` SHALL create a video of `duration` and `size`. `ImageToVideoGenerator` SHALL animate an image with optional `instructions` for the motion. `AudioToVideoGenerator` SHALL make `avatar` speak the audio.

#### Scenario: Requested duration and size
- **WHEN** `TextToVideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit().predict(['a boat at sea'])` is called
- **THEN** it returns one video of 2 seconds whose frames are 320×240

### Requirement: Video from video and multimodal samples
`VideoToVideoGenerator` SHALL reframe to `aspect_ratio` following the main subject, or join a list of clips. `MultimodalToVideoGenerator` SHALL lip-sync `video` to `audio`, replace the soundtrack of `video` with `audio`, or show `transcript` as subtitles, burned in or as an SRT file when `burn=False`.

#### Scenario: Lip sync
- **WHEN** `MultimodalToVideoGenerator(operation='lip_sync', backend='dummy')` predicts on `{'video': clip, 'audio': speech}`
- **THEN** it returns one video as long as `speech`

#### Scenario: Missing key
- **WHEN** `MultimodalToVideoGenerator(operation='mux')` receives a sample without `audio`
- **THEN** `InvalidSampleError` names the missing key and the sample index

### Requirement: Speech voice learned from samples
`TextToAudioGenerator` and `AudioToAudioGenerator(operation='convert_voice')` SHALL accept one recording or a list of recordings of the target voice in `voice`, and `fit` SHALL learn `voice_` by joining them. Backends that cannot clone voices SHALL raise `UnsupportedBackendError` at `fit`.

#### Scenario: Voice learned from samples
- **WHEN** `TextToAudioGenerator(voice=[recording_1, recording_2])` is fitted on a backend that clones voices
- **THEN** `voice_` is the joined reference audio and every speech request carries it

### Requirement: Image style learned from references
`TextToImageGenerator.fit` SHALL learn `style_` from `style_images`, written by the backend's image description, and apply it to every prompt, with an explicit `style` taking precedence. A backend that cannot describe images SHALL raise `UnsupportedBackendError` at `fit`.

#### Scenario: Learned style
- **WHEN** it is constructed with three watercolor `style_images` and fitted on a backend that describes images
- **THEN** `style_` is set and every generation request carries it

### Requirement: Portrait prepared for talking heads
`AudioToVideoGenerator.fit` SHALL prepare `avatar_` from `avatar`: converted to RGB, center-cropped to a square, and resized to the backend's input size.

#### Scenario: Prepared portrait
- **WHEN** it is fitted with a 640×480 portrait on a backend whose input size is 512
- **THEN** `avatar_` is a 512×512 RGB image

## REMOVED Requirements

### Requirement: SpeechSynthesizer
**Reason**: Generators are named by input and output.
**Migration**: Use `TextToAudioGenerator()`.

### Requirement: ImageGenerator
**Reason**: Generators are named by input and output.
**Migration**: Use `TextToImageGenerator()`.

### Requirement: TalkingAvatarGenerator
**Reason**: Generators are named by input and output.
**Migration**: Use `AudioToVideoGenerator(avatar=...)`.

### Requirement: Synthesizer learns a voice from samples
**Reason**: The requirement now names `TextToAudioGenerator`.
**Migration**: See "Speech voice learned from samples".

### Requirement: Image generator learns a style from references
**Reason**: The requirement now names `TextToImageGenerator`.
**Migration**: See "Image style learned from references".

### Requirement: Avatar generator prepares the portrait
**Reason**: The requirement now names `AudioToVideoGenerator`.
**Migration**: See "Portrait prepared for talking heads".

### Requirement: VideoGenerator
**Reason**: Generators are named by input and output.
**Migration**: Use `TextToVideoGenerator()`.

### Requirement: ImageEditor
**Reason**: Edits are an operation of the image-to-image generator.
**Migration**: Use `ImageToImageGenerator(operation='edit')`.

### Requirement: ImageAnimator
**Reason**: Generators are named by input and output.
**Migration**: Use `ImageToVideoGenerator()`.
