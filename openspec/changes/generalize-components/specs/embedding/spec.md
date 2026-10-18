# Spec Delta

## ADDED Requirements

### Requirement: Embedder
The library SHALL provide `Embedder` in `skai.embedding`, one transformer that maps samples of the input type selected by `input` (`'text'` by default, `'image'`, `'audio'` or `'video'`) to fixed-size vectors, with `dimensions`, `center` and `normalize`. A video SHALL be embedded as the mean of the embeddings of `n_frames` sampled frames.

#### Scenario: Every input type
- **WHEN** `Embedder()`, `Embedder(input='image')`, `Embedder(input='audio')` and `Embedder(input='video', n_frames=4)` are fitted on `dummy` samples of their type
- **THEN** each returns a float array with one row per sample

## MODIFIED Requirements

### Requirement: Embedder learns centering and a projection
`Embedder.fit(X)` SHALL embed `X` and learn `mean_`, which is subtracted from every embedding when `center=True` (the default). When `dimensions` is set and the backend cannot shorten embeddings natively, `fit` SHALL learn a PCA projection `components_` to `dimensions`, used instead of truncation.

#### Scenario: Centered embeddings
- **WHEN** it is fitted on a corpus and transforms the same corpus with `normalize=False`
- **THEN** the mean of the output is zero

#### Scenario: PCA instead of truncation
- **WHEN** `dimensions=4` is set on a backend that cannot shorten embeddings
- **THEN** `components_` has shape `(4, n_native_dimensions)` and the output keeps the four directions of largest variance

#### Scenario: Native shortening
- **WHEN** `dimensions=4` is set on a backend that shortens embeddings natively
- **THEN** no projection is learned and the backend is asked for four dimensions

### Requirement: Shared text and image space
When an `Embedder` of text and an `Embedder` of images use the same joint text-image model, their vectors SHALL have the same dimensions and be directly comparable, so texts can retrieve images.

#### Scenario: Same model, same space
- **WHEN** both embedders use the same backend and joint model and `normalize=True`
- **THEN** both return vectors of equal length whose dot products are cosine similarities

## REMOVED Requirements

### Requirement: TextEmbedder
**Reason**: Embedders are one class whose input type is a parameter.
**Migration**: Use `Embedder()`, which takes text by default.

### Requirement: ImageEmbedder
**Reason**: Embedders are one class whose input type is a parameter.
**Migration**: Use `Embedder(input='image')`.

### Requirement: AudioEmbedder
**Reason**: Embedders are one class whose input type is a parameter.
**Migration**: Use `Embedder(input='audio')`.
