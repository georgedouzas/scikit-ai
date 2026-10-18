# Spec Delta

## ADDED Requirements

### Requirement: VideoEmbedder
The library SHALL provide `VideoEmbedder` in `skai.embedding`, a transformer that maps every video to the mean of the embeddings of `n_frames` frames sampled evenly over it, with the `dimensions`, `center` and `normalize` parameters of the other embedders.

#### Scenario: Video vectors
- **WHEN** `VideoEmbedder(n_frames=4, backend='dummy').fit_transform(videos)` runs on 3 videos
- **THEN** it returns a float array with three rows
