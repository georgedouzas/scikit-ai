# User guide

scikit-ai components are scikit-learn estimators that do one task on text, images, audio or video. This guide shows how
to use them, how to choose where they run, and what each component does.

## The estimator contract

Every component is configured with keyword parameters, is fitted with `fit`, and is then used with `predict`,
`predict_proba` or `transform`:

- Generators, such as `TextSummarizer` or `ImageGenerator`, return one output per sample from both `predict` and
  `transform`, so they can be the last step or an intermediate step of a `Pipeline`. They learn nothing from data, so
  `fit()` only validates the parameters and resolves the backend.
- Classifiers, `TextClassifier` and `ImageClassifier`, learn the classes from the targets, or take them from `labels`
  for zero-shot classification, and pass `k_shot` training samples to the backend as examples.
- Embedders, such as `TextEmbedder`, return a float array from `transform` that any scikit-learn model can use.

Calling `predict` or `transform` before `fit` raises `NotFittedError`, and invalid parameters raise
`InvalidParameterError` at `fit` time.

```python exec="true" source="above" result="text"
from sklearn.pipeline import make_pipeline
from skai.audio import SpeechTranscriber
from skai.core import Audio
from skai.text import SpeechSynthesizer, TextClassifier

speech = SpeechSynthesizer(backend='dummy').fit().predict(['great product', 'awful service', 'love it'])
pipeline = make_pipeline(SpeechTranscriber(backend='dummy'), TextClassifier(k_shot=2, random_state=0, backend='dummy'))
pipeline.fit(speech, ['positive', 'negative', 'positive'])
print(pipeline.predict(speech))
```

## Batches, failures and seeds

Every component takes the same execution parameters:

- `batch_size` is the number of samples submitted to the backend at once. Outputs always keep the input order.
- `on_error='raise'` raises `skai.core.SampleError` naming the first failed sample. `on_error='ignore'` returns `None`,
  or a row of NaN for numeric outputs, for every failed sample and emits one `SampleFailureWarning` that names them.
- `random_state` controls the selection of few-shot examples and the seeds sent to backends that support seeding.

## Next steps

- [Media inputs and outputs](media.md) lists the accepted samples and output formats.
- [Backends](backends.md) explains how to choose and configure where components run.
- [Text](text.md), [images](image.md) and [audio](audio.md) describe every component.
- [Migrating from 0.1](migration.md) maps the removed `OpenAIClassifier` to `TextClassifier`.
