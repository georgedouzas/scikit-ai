# embedding Specification

## Purpose
Components that map samples to fixed-size vectors for scikit-learn models, importable from `skai.embedding`.

## Requirements

### Requirement: TextEmbedder
The library SHALL provide `TextEmbedder`, a transformer mapping each text to a fixed-size float vector, with optional `dimensions` for backends that support truncation and `normalize` to return unit-length vectors.

#### Scenario: Fixed dimensionality
- **WHEN** `TextEmbedder(backend="dummy", dimensions=64).fit_transform(texts)` runs on 10 texts
- **THEN** it returns a float array of shape `(10, 64)`

### Requirement: Embedder learns centering and a projection
`TextEmbedder.fit(X)` SHALL embed `X` and learn `mean_`, which is subtracted from every embedding when `center=True` (the default). When `dimensions` is set and the backend cannot shorten embeddings natively, `fit` SHALL learn a PCA projection `components_` to `dimensions`, used instead of truncation.

#### Scenario: Centered embeddings
- **WHEN** it is fitted on a corpus and transforms the same corpus with `normalize=False`
- **THEN** the mean of the output is zero

#### Scenario: PCA instead of truncation
- **WHEN** `dimensions=4` is set on a backend that cannot shorten embeddings
- **THEN** `components_` has shape `(4, n_native_dimensions)` and the output keeps the four directions of largest variance

#### Scenario: Native shortening
- **WHEN** `dimensions=4` is set on a backend that shortens embeddings natively
- **THEN** no projection is learned and the backend is asked for four dimensions
