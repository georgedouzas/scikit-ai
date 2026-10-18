# Embedding

`skai.embedding` holds the components that map samples to fixed-size vectors: `TextEmbedder`, `ImageEmbedder`,
`AudioEmbedder` and `VideoEmbedder`. Any scikit-learn model can use the vectors as features. `TextEmbedder` returns
vectors of `dimensions` entries, optionally of unit length.

```python exec="true" source="above" result="text"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from skai.embedding import TextEmbedder

pipeline = make_pipeline(TextEmbedder(dimensions=16, normalize=True, backend='dummy'), LogisticRegression())
pipeline.fit(['cheap', 'expensive', 'bargain', 'luxury'], [0, 1, 0, 1])
print(pipeline.predict(['affordable']))
```

## Images and audio

`ImageEmbedder` and `AudioEmbedder` map images and audio to vectors with the same parameters as `TextEmbedder`. With
the same joint text and image model in `TextEmbedder` and `ImageEmbedder`, such as `'hf:clip-ViT-B-32'`, the
vectors share one space, so texts can retrieve images with `NearestNeighbors`.

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from skai.embedding import ImageEmbedder, TextEmbedder

images = [Image.new('RGB', (16, 16), color) for color in ('red', 'green', 'blue')]
gallery = ImageEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(images)
query = TextEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(['a green square'])
print(NearestNeighbors(n_neighbors=2).fit(gallery).kneighbors(query, return_distance=False))
```

## Videos

`VideoEmbedder` maps every video to the mean of the embeddings of `n_frames` frames sampled evenly over it, with the
image embedding model of the backend. With a joint text and image model, texts can retrieve videos, and with
`VideoSegmenter` and `Retriever` they can retrieve single shots, as the `video_search` [recipe](recipes.md) does.

```python exec="true" source="above" result="text"
import numpy as np
from skai.core import Video
from skai.embedding import VideoEmbedder

videos = [Video(np.full((8, 8, 8, 3), value, dtype=np.uint8), fps=4) for value in (0, 100, 200)]
print(VideoEmbedder(n_frames=4, dimensions=2, backend='dummy').fit_transform(videos).shape)
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
