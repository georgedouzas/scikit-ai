# Spec Delta

## Purpose

Assembles media locally, without a model: joins clips and audio in sequence, mixes tracks, puts audio under video and
writes or burns subtitles, so generated pieces become one finished result.

## ADDED Requirements

### Requirement: Composer
The library SHALL provide `Composer` in `skai.composition`, a transformer that turns every sample into one image, audio or video by its `operation`, without calling any backend. It SHALL accept the `output_format` and `output_dir` of media generators.

#### Scenario: No backend
- **WHEN** a `Composer` is fitted and used with the `video` extra installed
- **THEN** it reaches no provider and needs no credentials

### Requirement: Concatenation
With `operation='concatenate'`, a sample SHALL be a list of videos or a list of audio, or a list of records with `video` and `audio`, and the output SHALL play them in order, each video with its audio when the record gives one.

#### Scenario: Scenes into one video
- **WHEN** a sample holds three records of a 2-second clip and its narration
- **THEN** the output is one video of the summed durations of the longer of each clip and its narration

### Requirement: Mixing
With `operation='mix'`, a sample SHALL be a record of named audio tracks with optional per-track `volumes`, and tracks named in `duck_under` SHALL be lowered while a track named in `duck_for` is not silent.

#### Scenario: Music under narration
- **WHEN** a record `{'voice': speech, 'music': song}` is mixed with `duck_under=['music']` and `duck_for='voice'`
- **THEN** the output is as long as the longer track and the music is quieter where the voice speaks

### Requirement: Audio under video
With `operation='mux'`, a sample SHALL be a record with `video` and `audio`, and the output SHALL be the video with that audio as its track, cut or padded to the video's length.

#### Scenario: Replace the soundtrack
- **WHEN** a record with a 4-second video and 6 seconds of audio is muxed
- **THEN** the output is 4 seconds long and its audio is the first 4 seconds of the audio

### Requirement: Subtitles
With `operation='subtitle'`, a sample SHALL be a record with `video` and `transcript` (timed records), and the output SHALL be the video with the text burned in, or an SRT file when `burn=False`.

#### Scenario: Burned subtitles
- **WHEN** a record with a video and two timed segments is subtitled
- **THEN** the output video has the same length and frames that differ from the input where the segments are shown
