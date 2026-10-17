# Embedding

`skai.embedding` holds the components that map samples to fixed-size vectors, so any scikit-learn model can use them
as features. `TextEmbedder` returns vectors of `dimensions` entries, optionally of unit length.

```python exec="true" source="above" result="text"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from skai.embedding import TextEmbedder

pipeline = make_pipeline(TextEmbedder(dimensions=16, normalize=True, backend='dummy'), LogisticRegression())
pipeline.fit(['cheap', 'expensive', 'bargain', 'luxury'], [0, 1, 0, 1])
print(pipeline.predict(['affordable']))
```

## What fit learns

`fit(X)` embeds the texts and learns their mean `mean_`, which is subtracted from every embedding when `center=True`.
When `dimensions` is set and the backend cannot shorten embeddings, it learns a PCA projection `components_` that
keeps the directions of largest variance.

```python exec="true" source="above" result="text"
from skai.embedding import TextEmbedder

corpus = [f'sentence {index}' for index in range(10)]
embeddings = TextEmbedder(backend='dummy').fit(corpus).transform(corpus)
print(abs(embeddings.mean(axis=0)).max() < 1e-12)
```

The API reference is in [`skai.embedding`](../api/embedding/index.md).
