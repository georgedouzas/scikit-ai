# Spec Delta

## Purpose

Splits every sample into labeled regions in space or time, such as objects, masks, shots, speaker turns or passages,
so later steps can work on each region and composers can reassemble them.

## ADDED Requirements

### Requirement: Segmenter
The library SHALL provide `Segmenter` in `skai.segmentation`, which returns one list of region records per sample of the input type selected by `input` (`'image'` by default, `'video'`, `'audio'` or `'text'`). Every region SHALL carry `label`, `score` and `media`, the cropped piece of the sample, so it can feed later steps.

#### Scenario: Regions feed later steps
- **WHEN** `Segmenter(labels=['person'], backend='dummy')` predicts on an image
- **THEN** it returns a list of regions whose `media` values are PIL images

### Requirement: Image regions
For images, `mode='box'` SHALL return regions with `box` as `(left, top, right, bottom)` pixels, and `mode='mask'` SHALL also return `mask`, a grayscale image of the input size whose white pixels belong to the region. The regions to find SHALL be named by `labels` or described by `prompt`.

#### Scenario: Mask for an edit
- **WHEN** `Segmenter(mode='mask', prompt='the sky', backend='dummy')` predicts on a photo
- **THEN** every region has a `mask` of the photo's size

### Requirement: Time regions
For video, audio and text, regions SHALL carry `start` and `end` (seconds for video and audio, character offsets for text). Video `mode` SHALL be `'shots'` (cuts) or `'prompt'` (moments matching `prompt`). Audio `mode` SHALL be `'silence'`, `'speakers'` or `'prompt'`. Text `mode` SHALL be `'paragraphs'`, `'sentences'` or `'chunks'` of `chunk_size` characters.

#### Scenario: Shots
- **WHEN** `Segmenter(input='video', mode='shots')` predicts on a video with two cuts
- **THEN** it returns three regions in time order whose spans cover the video

#### Scenario: Passages
- **WHEN** `Segmenter(input='text', mode='chunks', chunk_size=100)` predicts on a 250-character text
- **THEN** it returns three regions whose `media` values join back to the text

### Requirement: Local segmentation without a backend
Shot cuts, silence and text splitting SHALL run locally without a model, and other modes SHALL use the backend.

#### Scenario: Offline shots
- **WHEN** `Segmenter(input='video', mode='shots')` is fitted with `backend=None` and no extras besides `video`
- **THEN** it predicts without importing any provider
