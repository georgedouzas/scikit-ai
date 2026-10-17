# embedding Specification

## Purpose
Components that map samples to fixed-size vectors for scikit-learn models, importable from `skai.embedding`.

## Requirements

### Requirement: TextEmbedder
The library SHALL provide `TextEmbedder`, a transformer mapping each text to a fixed-size float vector, with optional `dimensions` for backends that support truncation and `normalize` to return unit-length vectors.

#### Scenario: Fixed dimensionality
- **WHEN** `TextEmbedder(backend="dummy", dimensions=64).fit_transform(texts)` runs on 10 texts
- **THEN** it returns a float array of shape `(10, 64)`
