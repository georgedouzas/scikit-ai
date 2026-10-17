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

The API reference is in [`skai.embedding`](../api/embedding/index.md).
