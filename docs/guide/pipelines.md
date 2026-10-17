# Pipelines

Components are scikit-learn estimators, so they chain with `make_pipeline`. The output of every generator is a valid
input of any component whose input type matches it, in every `output_format`. Pipelines fit intermediate steps through
`fit_transform`, which runs every generator once over the training samples and ignores `y`, so only the final step
learns from the target. Generating media during `fit` costs one backend call per sample and step, and
`Pipeline(memory=...)` caches those outputs so a refit on the same samples reuses them.

## Understanding speech and video

Transcripts feed any text component, for call classification, meeting summaries, subtitle translation or form
filling.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.classification import TextClassifier
from skai.core import Audio, Video
from skai.text import TextSummarizer
from skai.transcription import SpeechTranscriber, VideoTranscriber

recordings = [Audio(np.zeros(size, dtype=np.float32), 16000) for size in (16000, 24000, 32000, 40000)]
calls = make_pipeline(SpeechTranscriber(backend='dummy'), TextClassifier(backend='dummy'))
print(calls.fit(recordings, ['sales', 'support'] * 2).predict(recordings[:2]))
video = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10, audio=recordings[0])
print(make_pipeline(VideoTranscriber(backend='dummy'), TextSummarizer(backend='dummy')).fit(None).predict([video]))
```

## Understanding images and sounds through descriptions

Descriptions feed text components, so media can be classified, searched or turned into records.

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.description import ImageDescriptor
from skai.text import TextExtractor

photos = [Image.new('RGB', (32, 32), color) for color in ('brown', 'gray')]
attributes = make_pipeline(
    ImageDescriptor(backend='dummy'),
    TextExtractor(fields={'color': 'main color', 'material': 'main material'}, backend='dummy'),
)
print(attributes.fit(photos).predict(photos))
```

## Speech translation and narration

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.core import Audio
from skai.description import ImageDescriptor
from skai.generation import SpeechSynthesizer
from skai.text import TextTranslator
from skai.transcription import SpeechTranscriber

speech = [Audio(np.zeros(16000, dtype=np.float32), 16000)]
translation = make_pipeline(
    SpeechTranscriber(backend='dummy'),
    TextTranslator(target_language='Greek', backend='dummy'),
    SpeechSynthesizer(backend='dummy'),
)
alt_text = make_pipeline(ImageDescriptor(backend='dummy'), SpeechSynthesizer(backend='dummy'))
print(translation.fit(speech).predict(speech), alt_text.fit(None).predict([Image.new('RGB', (8, 8))]))
```

## Talking presenter

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.generation import SpeechSynthesizer, TalkingAvatarGenerator
from skai.text import TextTranslator

presenter = make_pipeline(
    TextTranslator(target_language='Spanish', backend='dummy'),
    SpeechSynthesizer(backend='dummy'),
    TalkingAvatarGenerator(avatar=Image.new('RGB', (32, 32), 'beige'), backend='dummy'),
)
print([round(video.duration, 2) for video in presenter.fit(None).predict(['Welcome!', 'Thanks for watching.'])])
```

## Prompt to animated scene

```python exec="true" source="above" result="text"
from sklearn.pipeline import make_pipeline
from skai.generation import ImageAnimator, ImageEditor, ImageGenerator

scene = make_pipeline(
    ImageGenerator(size=(64, 64), backend='dummy'),
    ImageEditor(instruction='make it night', backend='dummy'),
    ImageAnimator(duration=1, backend='dummy'),
)
print(scene.fit(None).predict(['a quiet harbor']))
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
from skai.description import ImageDescriptor

images = [Image.new('RGB', (8, 8), color) for color in ('red', 'blue')]
pipeline = Pipeline(
    [('describe', ImageDescriptor(backend='dummy')), ('classify', TextClassifier(backend='dummy'))],
    memory=tempfile.mkdtemp(),
)
print(pipeline.fit(images, ['warm', 'cold']).predict(images))
```
