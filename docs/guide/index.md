# User guide

scikit-ai components are scikit-learn estimators that do one task on text, images, audio or video. This guide shows how
to use them, how to choose where they run, and what each component does.

## The estimator contract

Every component is configured with keyword parameters, is fitted with `fit`, and is then used with `predict`,
`predict_proba` or `transform`:

- Generators, such as `TextSummarizer` or `ImageGenerator`, return one output per sample from both `predict` and
  `transform`, so they can be the last step or an intermediate step of a `Pipeline`. Supervised generators learn
  from reference outputs given as `y`, and the others prepare their conditioning, such as a voice or a portrait.
- Classifiers, `TextClassifier` and `ImageClassifier`, learn the classes from the targets, or take them from `labels`
  for zero-shot classification, and pass training samples to the backend as examples, one per class by default.
- Embedders, such as `TextEmbedder`, learn centering and an optional projection from `X`, and return a float array
  from `transform` that any scikit-learn model can use.

[What fit learns](fit.md) lists what every component learns, and how pipelines and fine-tuning work.

Calling `predict` or `transform` before `fit` raises `NotFittedError`, and invalid parameters raise
`InvalidParameterError` at `fit` time.

```python exec="true" source="above" result="text"
from sklearn.pipeline import make_pipeline
from skai.transcription import SpeechTranscriber
from skai.core import Audio
from skai.classification import TextClassifier
from skai.generation import SpeechSynthesizer

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
- [Classification](classification.md), [description](description.md), [transcription](transcription.md),
  [text](text.md), [embedding](embedding.md) and [generation](generation.md) describe every component.
- [What fit learns](fit.md) explains learning from data, `use_y` in pipelines, and fine-tuning.
- [Migrating from 0.1](migration.md) maps the removed `OpenAIClassifier` to `TextClassifier`.
