# Spec Delta

## ADDED Requirements

### Requirement: ImageEmbedder
The library SHALL provide `ImageEmbedder` in `skai.embedding`, a transformer mapping every image to a fixed-size float vector, with the same `dimensions` and `normalize` parameters as `TextEmbedder`.

#### Scenario: Image vectors
- **WHEN** `ImageEmbedder(dimensions=32, backend='dummy').fit_transform(images)` runs on 5 images
- **THEN** it returns a float array of shape `(5, 32)`

#### Scenario: Clustering images
- **WHEN** a pipeline of `ImageEmbedder` followed by `KMeans(n_clusters=2)` is fitted on images
- **THEN** it assigns every image to one of two clusters

### Requirement: Shared text and image space
When `TextEmbedder` and `ImageEmbedder` use the same joint text-image model, their vectors SHALL have the same dimensions and be directly comparable, so texts can retrieve images.

#### Scenario: Same model, same space
- **WHEN** both embedders use the same backend and joint model and `normalize=True`
- **THEN** both return vectors of equal length whose dot products are cosine similarities

### Requirement: AudioEmbedder
The library SHALL provide `AudioEmbedder` in `skai.embedding`, a transformer mapping every audio sample to a fixed-size float vector, with `dimensions` and `normalize`.

#### Scenario: Audio vectors feed a classifier
- **WHEN** a pipeline of `AudioEmbedder` followed by `LogisticRegression` is fitted on labeled audio
- **THEN** it predicts one label per audio sample
