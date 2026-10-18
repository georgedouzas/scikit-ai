# Spec Delta

## ADDED Requirements

### Requirement: Video frame sampling
Components that analyze video frames SHALL accept `n_frames` and sample that many frames evenly spaced over the video, including the first frame. A video with fewer frames than `n_frames` SHALL use all of its frames.

#### Scenario: Evenly spaced frames
- **WHEN** a 10-second video at 25 fps is analyzed with `n_frames=5`
- **THEN** the frames at 0, 2, 4, 6 and 8 seconds are used

#### Scenario: Short video
- **WHEN** a video of 3 frames is analyzed with `n_frames=8`
- **THEN** its 3 frames are used

### Requirement: Video audio track extraction
Components that analyze the speech or sound of a video SHALL read its audio track as `Audio`, mixed to mono, and treat a video without an audio track as silence of the video's length.

#### Scenario: Silent video
- **WHEN** a video without an audio track is passed to a component that reads audio
- **THEN** the component receives silent audio as long as the video
