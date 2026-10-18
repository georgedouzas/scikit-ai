# Spec Delta

## Purpose

Splits every sample into labeled regions in space or time, such as objects, masks, shots, speaker turns, timed speech
or passages, so later steps can work on each region and generators can reassemble them.

## ADDED Requirements

### Requirement: Segmenters for every input
The library SHALL provide `TextSegmenter`, `ImageSegmenter`, `AudioSegmenter` and `VideoSegmenter` in `skai.segmentation`. Each SHALL return one list of region records per sample, choosing what to find with `by`, and every region SHALL carry `label`, `score` and `media`, the cropped piece of the sample.

#### Scenario: Regions feed later steps
- **WHEN** `ImageSegmenter(labels=['person'], backend='dummy')` predicts on an image
- **THEN** it returns a list of regions whose `media` values are PIL images

#### Scenario: Invalid choice
- **WHEN** a segmenter is fitted with a `by` that its input does not offer
- **THEN** `InvalidParameterError` names the `by` parameter and the valid values

### Requirement: Image regions
`ImageSegmenter(by='object')` SHALL return regions with `box` as `(left, top, right, bottom)` pixels, and `by='mask'` SHALL also return `mask`, a grayscale image of the input size whose white pixels belong to the region. The regions SHALL be named by `labels` or described by `prompt`.

#### Scenario: Mask for an edit
- **WHEN** `ImageSegmenter(by='mask', prompt='the sky', backend='dummy')` predicts on a photo
- **THEN** every region has a `mask` of the photo's size

### Requirement: Audio regions
`AudioSegmenter` regions SHALL carry `start` and `end` in seconds. `by='speech'` (the default) SHALL return timed speech with `text`, `by='speaker'` SHALL return speaker turns with `speaker` and `text`, `by='silence'` SHALL return the parts between silences, and `by='moment'` SHALL return the parts matching `prompt`.

#### Scenario: Speaker turns
- **WHEN** `AudioSegmenter(by='speaker', backend='dummy')` predicts on a recording
- **THEN** every region has `start <= end`, a `speaker` label and `text`

#### Scenario: Timed speech with a vocabulary
- **WHEN** `AudioSegmenter(by='speech')` is fitted with reference transcripts as `y`
- **THEN** it learns `vocabulary_` as `AudioToTextGenerator` does and sends it with every request

### Requirement: Video and text regions
`VideoSegmenter` regions SHALL carry `start` and `end` in seconds, with `by='shot'` (cuts) or `by='moment'` (parts matching `prompt`). `TextSegmenter` regions SHALL carry `start` and `end` character offsets, with `by='paragraph'`, `'sentence'` or `'chunk'` of `chunk_size` characters.

#### Scenario: Shots
- **WHEN** `VideoSegmenter(by='shot')` predicts on a video with two cuts
- **THEN** it returns three regions in time order whose spans cover the video

#### Scenario: Passages
- **WHEN** `TextSegmenter(by='chunk', chunk_size=100)` predicts on a 250-character text
- **THEN** it returns three regions whose `media` values join back to the text

### Requirement: Local segmentation without a backend
Shot cuts, silence and text splitting SHALL run locally without a model, and other choices of `by` SHALL use the backend.

#### Scenario: Offline shots
- **WHEN** `VideoSegmenter(by='shot')` is fitted with `backend=None` and the `video` extra
- **THEN** it predicts without importing any provider
