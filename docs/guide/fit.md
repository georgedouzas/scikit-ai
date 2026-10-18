# What fit learns

`fit(X, y)` keeps the scikit-learn signature. For classifiers, `y` holds labels. For text generators and extractors,
`y` holds the reference output of every sample, so scorers, `cross_val_score` and `GridSearchCV` compare predictions
with references as usual. Conditioning that is not paired with `X`, such as a voice or reference images, stays in
constructor parameters and is prepared in `fit`.

--8<-- "README.md:fit"

## Learning from references

```python exec="true" source="above" result="text"
from skai.generation import TextToTextGenerator

texts = ['The meeting moved to Friday because the room was booked on Thursday.', 'Sales rose in the third quarter.']
references = ['Meeting moved to Friday.', 'Sales rose.']
summarizer = TextToTextGenerator(instructions='Summarize the text.', max_length='auto', k_shot=1, backend='dummy')
summarizer.fit(texts, references)
print(summarizer.max_length_, summarizer.examples_)
```

## Pipelines and `use_y`

`Pipeline`, `FeatureUnion`, `ColumnTransformer` and `MediaUnion` fit intermediate steps through `fit_transform(X, y)`
and pass them the target of the final step. Generators learn from `y` in `fit` and ignore it in `fit_transform`, so
only the final step learns from it. `use_y=True` makes a generator learn from `y` in `fit_transform` too, and
`use_y=False` makes it never learn from `y`.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.generation import AudioToTextGenerator, TextToTextGenerator

audio = [(np.zeros(16000), 16000), (np.zeros(8000), 16000)]
pipeline = make_pipeline(
    AudioToTextGenerator(backend='dummy'),
    TextToTextGenerator(instructions='Summarize the call.', backend='dummy'),
)
pipeline.fit(audio, ['A short call.', 'A shorter call.'])
print(pipeline[0].vocabulary_, len(pipeline[1].examples_))
```

`ForEach` fits its estimator on all the items with `y=None`, and `TopK` fits its classifier zero-shot on the items.

## Fine-tuning

`adaptation='finetune'` trains a model in `fit` on all the training data, with the same prompts that are used at
prediction, stores its id in `model_`, and sends later requests to it without examples. Classifiers and text generators
with `operation='instruct'` support it. It needs at least the backend's minimum number of samples (10 on OpenAI). A
backend that cannot fine-tune raises `UnsupportedBackendError` at `fit`, and a provider that refuses the job raises
`FineTuneError` with its reason. OpenAI stopped accepting new fine-tuning jobs from self-serve organizations in 2026.

```python exec="true" source="above" result="text"
from skai.classification import TextClassifier

texts = [f'review {index}' for index in range(10)]
labels = ['positive', 'negative'] * 5
classifier = TextClassifier(adaptation='finetune', backend='dummy').fit(texts, labels)
print(classifier.model_.startswith('dummy-finetuned-'), classifier.examples_)
```
