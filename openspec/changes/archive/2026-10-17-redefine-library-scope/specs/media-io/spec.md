# Spec Delta

## Purpose

Defines which input forms components accept for text, images, audio and video, and how generated media
is returned (in memory, as files or as bytes), so data moves between components and user code consistently.

## ADDED Requirements

### Requirement: Sample containers
Components SHALL accept `X` as a list, tuple, 1-D NumPy array, pandas Series, or single-column 2-D array or DataFrame, where each element is one sample.

#### Scenario: DataFrame column input
- **WHEN** `TextSummarizer.predict` receives a one-column DataFrame of 5 strings
- **THEN** it returns 5 summaries

#### Scenario: Multi-column input rejected
- **WHEN** a component receives a 2-D input with more than one column
- **THEN** `ValueError` is raised explaining that one column of samples is expected

### Requirement: Text inputs
Text samples SHALL be Python strings. Non-string samples SHALL raise `TypeError` naming the first offending index.

#### Scenario: Non-string sample
- **WHEN** a text component receives `["ok", 3]`
- **THEN** `TypeError` is raised naming index 1

### Requirement: Image inputs
Image samples SHALL be accepted as a local path, an `http(s)` URL, raw encoded bytes, a PIL image, or a NumPy array of shape `(H, W)` or `(H, W, C)`.

#### Scenario: Mixed image forms
- **WHEN** `ImageDescriptor.predict` receives a path, a URL and a PIL image
- **THEN** it returns three descriptions

### Requirement: Audio inputs
Audio samples SHALL be accepted as a local path, an `http(s)` URL, raw encoded bytes, a library audio object, or a `(samples, sample_rate)` tuple of a NumPy array and an integer.

#### Scenario: Array audio input
- **WHEN** `SpeechTranscriber.predict` receives `[(waveform, 16000)]`
- **THEN** it returns one transcript

### Requirement: Video inputs
Video samples SHALL be accepted as a local path, an `http(s)` URL, raw encoded bytes, or a library video object.

#### Scenario: Path video input
- **WHEN** a video-consuming parameter or input is given an `.mp4` path
- **THEN** it is read without the user decoding frames

### Requirement: Unreadable inputs
A sample that cannot be read or decoded SHALL raise `ValueError` naming its index and the reason, subject to `on_error`.

#### Scenario: Missing file
- **WHEN** an image component receives a path that does not exist
- **THEN** `ValueError` names the index and the missing path

### Requirement: Text and label outputs
Text outputs SHALL be returned as a 1-D NumPy array of `dtype=object` holding strings. Label outputs SHALL follow scikit-learn conventions for the target type seen in `fit`.

#### Scenario: Text output type
- **WHEN** `TextTranslator.predict` is called on 4 strings
- **THEN** it returns a NumPy object array of shape `(4,)` containing strings

### Requirement: Media output format selection
Components producing images, audio or video SHALL accept `output_format` with values `"object"` (default), `"path"` and `"bytes"`, and return a 1-D object array of the selected form.

#### Scenario: In-memory image output
- **WHEN** `ImageGenerator(output_format="object").predict(["a cat"])` is called
- **THEN** it returns an array holding one PIL image

#### Scenario: Bytes output
- **WHEN** `SpeechSynthesizer(output_format="bytes").predict(["hi"])` is called
- **THEN** it returns an array holding one `bytes` value encoded in the documented default audio format

### Requirement: File output location
With `output_format="path"`, components SHALL write each output to `output_dir` (a temporary directory when `None`) under a unique file name and return the paths. Existing files SHALL NOT be overwritten.

#### Scenario: Paths into a directory
- **WHEN** `TalkingAvatarGenerator(output_format="path", output_dir="out")` predicts 2 videos
- **THEN** two new files exist under `out/` and their paths are returned in input order

### Requirement: In-memory audio and video objects
In-memory audio outputs SHALL expose the samples array, the sample rate and a save method. In-memory video outputs SHALL expose frames, frame rate, optional audio and a save method. Both SHALL be accepted back as inputs by any component.

#### Scenario: Round trip between components
- **WHEN** the in-memory audio from `SpeechSynthesizer` is passed to `SpeechTranscriber.predict`
- **THEN** it is accepted without conversion by the user
