# Spec Delta

## ADDED Requirements

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
