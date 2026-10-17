# Text components

The components of `skai.text` take strings as input.

## Classify texts

`TextClassifier` assigns labels zero-shot from `labels`, or few-shot from training texts and targets. `labels` can be
a dict that maps every label to a description, and `instructions` adds guidance. Targets can be labels or a binary
indicator matrix for multilabel classification.

```python exec="true" source="above" result="text"
from skai.text import TextClassifier

zero_shot = TextClassifier(labels={'bug': 'a defect report', 'feature': 'a request for something new'}, backend='dummy')
print(zero_shot.fit().predict(['The app crashes on start.', 'Please add dark mode.']))

few_shot = TextClassifier(k_shot=2, random_state=0, backend='dummy')
few_shot.fit(['Loved it.', 'Hated it.', 'Superb.', 'Awful.'], ['positive', 'negative', 'positive', 'negative'])
print(few_shot.predict_proba(['Not bad at all.']).round(2))
```

## Summarize and translate

`TextSummarizer` returns summaries of at most `max_length` words in an optional `style`. `TextTranslator` translates
into `target_language`, from `source_language` or from the detected language.

```python exec="true" source="above" result="text"
from skai.text import TextSummarizer, TextTranslator

text = 'scikit-ai gives every media task the same scikit-learn interface, whatever model runs it.'
print(TextSummarizer(max_length=6, style='one line', backend='dummy').fit().predict([text]))
print(TextTranslator(target_language='Greek', backend='dummy').fit().predict(['Good morning']))
```

## Embed texts

`TextEmbedder` maps every text to a vector of `dimensions` entries, optionally of unit length, so any scikit-learn
model can use texts as features.

```python exec="true" source="above" result="text"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from skai.text import TextEmbedder

pipeline = make_pipeline(TextEmbedder(dimensions=16, normalize=True, backend='dummy'), LogisticRegression())
pipeline.fit(['cheap', 'expensive', 'bargain', 'luxury'], [0, 1, 0, 1])
print(pipeline.predict(['affordable']))
```

## Speak texts and generate images

`SpeechSynthesizer` speaks every text in a `voice`. A string names a voice of the backend, and reference audio clones
a voice on backends that support it. `ImageGenerator` creates one image of `size` per prompt, with an optional `style`
and `negative_prompt`. Both return media in the requested `output_format`.

```python exec="true" source="above" result="text"
from skai.text import ImageGenerator, SpeechSynthesizer

speech = SpeechSynthesizer(voice='alloy', backend='dummy').fit().predict(['Welcome to scikit-ai.'])
images = ImageGenerator(size=(256, 128), style='watercolor', random_state=0, backend='dummy').fit().predict(['a lake'])
print(speech[0], images[0].size)
```

The API reference of every component is in [`skai.text`](../api/text/index.md).
