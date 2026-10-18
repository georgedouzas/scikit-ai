# Pipelines

Components are scikit-learn estimators, so they chain with `make_pipeline`. The output of every generator is a valid
input of any component whose input type matches it, in every `output_format`. A mismatched input, such as an image
passed to a text component, raises `InvalidSampleError` naming the first sample. Pipelines fit intermediate steps
through `fit_transform`, which runs every generator once over the training samples and ignores `y`, so only the final
step learns from the target. Generating media during `fit` costs one backend call per sample and step, and
`Pipeline(memory=...)` caches those outputs so a refit on the same samples reuses them.

## Chains of components

Transcripts and descriptions feed any text component, for call classification, meeting summaries, subtitle
translation or form filling.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.classification import TextClassifier
from skai.core import Audio
from skai.generation import AudioToTextGenerator, TextToAudioGenerator, TextToTextGenerator

recordings = [Audio(np.zeros(size, dtype=np.float32), 16000) for size in (16000, 24000, 32000, 40000)]
calls = make_pipeline(AudioToTextGenerator(backend='dummy'), TextClassifier(backend='dummy'))
print(calls.fit(recordings, ['sales', 'support'] * 2).predict(recordings[:2]))
translation = make_pipeline(
    AudioToTextGenerator(backend='dummy'),
    TextToTextGenerator(instructions='Translate into Greek.', backend='dummy'),
    TextToAudioGenerator(backend='dummy'),
)
print(translation.fit(None).predict(recordings[:1]))
```

## Several inputs per sample with `MediaUnion`

`MediaUnion` runs named branches on the same samples and returns one multimodal sample per input sample, a dict keyed
by branch name. It is modeled on `FeatureUnion`: a branch is an estimator, a pipeline or `'passthrough'`, branches are
fitted through `fit_transform`, and their parameters are routed as `<branch>__<parameter>`. A branch that cannot
transform, such as one ending in a classifier, contributes its predictions. A later multimodal step then receives
several inputs per sample.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.core import Audio, Video
from skai.generation import MultimodalToVideoGenerator, TextToAudioGenerator, TextToTextGenerator, VideoToTextGenerator
from skai.pipeline import MediaUnion

speech = make_pipeline(
    VideoToTextGenerator(operation='transcribe', backend='dummy'),
    TextToTextGenerator(instructions='Translate into Greek.', backend='dummy'),
    TextToAudioGenerator(backend='dummy'),
)
revoice = make_pipeline(
    MediaUnion([('video', 'passthrough'), ('audio', speech)]),
    MultimodalToVideoGenerator(operation='mux'),
)
video = Video(np.zeros((16, 8, 8, 3), dtype=np.uint8), fps=8, audio=Audio(np.zeros(32000, dtype=np.float32), 16000))
print(revoice.fit(None).predict([video]), revoice.get_params()['mediaunion__audio__texttotextgenerator__instructions'])
```

## Picking fields with `Field`

`Field` is a stateless transformer that returns one field of every multimodal sample or region, or a dict of several
fields when given a list. A sample without the field raises `KeyError` naming the field and the sample index.

```python exec="true" source="above" result="text"
from skai.pipeline import Field

scenes = [{'visual': 'a boat', 'narration': 'At sea.', 'seconds': '4'}]
print(Field('visual').fit_transform(scenes), Field(['visual', 'narration']).fit_transform(scenes))
```

## Every item with `ForEach`

`ForEach` applies an estimator to every item of list-valued samples, such as the regions of a segmenter or the scenes
of a storyboard, and returns lists of the same lengths. The items of all samples go to the estimator in one call, so
the backend receives them as one batch. `fit` fits the estimator on all the items with `y=None`, and an empty list
gives an empty list.

```python exec="true" source="above" result="text"
from sklearn.pipeline import make_pipeline
from skai.extraction import TextExtractor
from skai.generation import TextToImageGenerator
from skai.pipeline import Field, ForEach

fields = {'scenes': [{'narration': 'spoken text', 'visual': 'what is shown'}]}
storyboard = TextExtractor(fields=fields, backend='dummy')
scene_images = ForEach(make_pipeline(Field('visual'), TextToImageGenerator(size=(64, 48), backend='dummy')))
images = make_pipeline(storyboard, scene_images).fit(None).predict(['A brief about solar panels.', 'Another brief.'])
print([len(sample) for sample in images], images[0][0].size)
```

## The best items with `TopK`

`TopK` keeps the `k` items of every list-valued sample with the highest scores, or all those above `threshold`, in
their original order. The score comes from a callable on one item, or from the `predict_proba` column of `label` of a
classifier, which `fit` fits zero-shot on the items.

```python exec="true" source="above" result="text"
import numpy as np
from skai.classification import VideoClassifier
from skai.core import Video
from skai.pipeline import TopK

clips = [Video(np.full((4, 8, 8, 3), value, dtype=np.uint8), fps=4) for value in (0, 60, 120, 180, 240)]
exciting = TopK(VideoClassifier(labels=['exciting', 'calm'], n_frames=2, backend='dummy'), label='exciting', k=2)
print(exciting.fit_transform([clips]), TopK(len, k=2).fit_transform([['a', 'abc', 'ab', 'abcd']]))
```

## Search with `Retriever`

`Retriever` indexes the items of samples with an item pipeline and answers queries with a query pipeline. `fit` runs
the item pipeline, which returns one vector, or one list of records with a `vector` key, per sample, and indexes every
item with the index of its sample. `kneighbors(queries, n_neighbors)` embeds the queries and returns, for every query,
the records of the nearest items with their `sample` index and `distance`. The
[`video_search` recipe](recipes.md#searching-a-video-library-by-text) indexes the shots of videos this way.

```python exec="true" source="above" result="text"
from skai.embedding import TextEmbedder
from skai.pipeline import Retriever

retriever = Retriever(TextEmbedder(center=False, backend='dummy'), TextEmbedder(center=False, backend='dummy'))
(nearest,) = retriever.fit(['a goal', 'a corner', 'a foul']).kneighbors(['a goal'], n_neighbors=2)
print([(record['sample'], round(record['distance'], 2)) for record in nearest])
```

## Embeddings with scikit-learn

Every embedder works before any scikit-learn estimator that takes a numeric matrix, and `FeatureUnion` combines
embedders of the same input type.

```python exec="true" source="above" result="text"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from skai.embedding import TextEmbedder

features = FeatureUnion([('a', TextEmbedder(backend='dummy:a')), ('b', TextEmbedder(dimensions=4, backend='dummy:b'))])
model = make_pipeline(features, LogisticRegression()).fit(['cheap', 'pricey', 'bargain', 'luxury'], [0, 1, 0, 1])
print(model.predict(['affordable']))
```

## Caching and tuning

`Pipeline(memory=...)` caches the outputs of generator steps whose backend is a string or `None`, and `GridSearchCV`
tunes any step parameter, including backend parameters, with `<step>__<parameter>` names.

```python exec="true" source="above" result="text"
import tempfile
from PIL import Image
from sklearn.pipeline import Pipeline
from skai.classification import TextClassifier
from skai.generation import ImageToTextGenerator

images = [Image.new('RGB', (8, 8), color) for color in ('red', 'blue')]
pipeline = Pipeline(
    [
        ('describe', ImageToTextGenerator(instructions='Describe the image.', backend='dummy')),
        ('classify', TextClassifier(backend='dummy')),
    ],
    memory=tempfile.mkdtemp(),
)
print(pipeline.fit(images, ['warm', 'cold']).predict(images))
```

[Recipes](recipes.md) combine these primitives into complete pipelines, such as an explainer video or a dubbed video.
