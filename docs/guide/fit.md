# What fit learns

`fit(X, y)` keeps the scikit-learn signature. For classifiers, `y` holds labels. For the summarizer, the translator,
the image descriptor and the transcriber, `y` holds the reference output of every sample, so scorers,
`cross_val_score` and `GridSearchCV` compare predictions with references as usual. Conditioning that is not paired
with `X`, such as a voice or reference images, stays in constructor parameters and is prepared in `fit`.

--8<-- "README.md:fit"

## Pipelines and `use_y`

`Pipeline`, `FeatureUnion` and `ColumnTransformer` fit intermediate steps through `fit_transform(X, y)` and pass them
the target of the final step. Supervised generators learn from `y` in `fit` and ignore it in `fit_transform`, so only
the final step learns from it. `use_y=True` makes a generator learn from `y` in `fit_transform` too, and `use_y=False`
makes it never learn from `y`.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.text import TextSummarizer
from skai.transcription import SpeechTranscriber

audio = [(np.zeros(16000), 16000), (np.zeros(8000), 16000)]
pipeline = make_pipeline(SpeechTranscriber(backend='dummy'), TextSummarizer(backend='dummy'))
pipeline.fit(audio, ['A short call.', 'A shorter call.'])
print(pipeline[0].vocabulary_, len(pipeline[1].examples_))
```

## Fine-tuning

`adaptation='finetune'` trains a model in `fit` on all the training data, with the same prompts that are used at
prediction, stores its id in `model_`, and sends later requests to it without examples. It needs at least the
backend's minimum number of samples (10 on OpenAI). A backend that cannot fine-tune raises `UnsupportedBackendError`
at `fit`, and a provider that refuses the job raises `FineTuneError` with its reason. OpenAI stopped accepting new
fine-tuning jobs from self-serve organizations in 2026.

```python exec="true" source="above" result="text"
from skai.classification import TextClassifier

texts = [f'review {index}' for index in range(10)]
labels = ['positive', 'negative'] * 5
classifier = TextClassifier(adaptation='finetune', backend='dummy').fit(texts, labels)
print(classifier.model_.startswith('dummy-finetuned-'), classifier.examples_)
```
