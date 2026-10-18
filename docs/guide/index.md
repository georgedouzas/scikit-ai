# User guide

scikit-ai components are scikit-learn estimators that work on text, images, audio, video and multimodal samples.
This guide shows how the components are named, how to use them, how to choose where they run, and how to combine
them into pipelines.

## The estimator contract

Every component is configured with keyword parameters, is fitted with `fit`, and is then used with `predict`,
`predict_proba` or `transform`:

- Classifiers learn the classes from the targets, or take them from `labels` for zero-shot classification, and pass
  training samples to the backend as examples, one per class by default.
- Extractors return one record of declared `fields` per sample from `predict`, and one column per field from
  `transform`.
- Embedders learn centering and an optional projection from `X`, and return a float array from `transform` that any
  scikit-learn model can use.
- Segmenters return one list of regions per sample, such as objects, shots, speaker turns or passages.
- Generators return one output per sample from both `predict` and `transform`, so they can be the last step or an
  intermediate step of a `Pipeline`. Text generators learn from reference outputs given as `y`, and media generators
  prepare their conditioning, such as a voice or a portrait.

[What fit learns](fit.md) lists what every component learns, and how pipelines and fine-tuning work.

Calling `predict` or `transform` before `fit` raises `NotFittedError`, and invalid parameters raise
`InvalidParameterError` at `fit` time.

```python exec="true" source="above" result="text"
from sklearn.pipeline import make_pipeline
from skai.classification import TextClassifier
from skai.generation import AudioToTextGenerator, TextToAudioGenerator

speech = TextToAudioGenerator(backend='dummy').fit().predict(['great product', 'awful service', 'love it'])
pipeline = make_pipeline(
    AudioToTextGenerator(backend='dummy'),
    TextClassifier(k_shot=2, random_state=0, backend='dummy'),
)
pipeline.fit(speech, ['positive', 'negative', 'positive'])
print(pipeline.predict(speech))
```

## The naming grid

A class name says what every sample is and what every output is. The input is `Text`, `Image`, `Audio`, `Video` or
`Multimodal`, a dict of named parts. Analyzers are named `{Input}{Role}`:

| Input | Classifier | Extractor | Embedder | Segmenter |
|---|---|---|---|---|
| Text | `TextClassifier` | `TextExtractor` | `TextEmbedder` | `TextSegmenter` |
| Image | `ImageClassifier` | `ImageExtractor` | `ImageEmbedder` | `ImageSegmenter` |
| Audio | `AudioClassifier` | `AudioExtractor` | `AudioEmbedder` | `AudioSegmenter` |
| Video | `VideoClassifier` | `VideoExtractor` | `VideoEmbedder` | `VideoSegmenter` |
| Multimodal | `MultimodalClassifier` | `MultimodalExtractor` | | |

Generators are named `{Input}To{Output}Generator`:

| Input | Text | Image | Audio | Video |
|---|---|---|---|---|
| Text | `TextToTextGenerator` | `TextToImageGenerator` | `TextToAudioGenerator` | `TextToVideoGenerator` |
| Image | `ImageToTextGenerator` | `ImageToImageGenerator` | | `ImageToVideoGenerator` |
| Audio | `AudioToTextGenerator` | | `AudioToAudioGenerator` | `AudioToVideoGenerator` |
| Video | `VideoToTextGenerator` | `VideoToImageGenerator` | `VideoToAudioGenerator` | `VideoToVideoGenerator` |
| Multimodal | `MultimodalToTextGenerator` | `MultimodalToImageGenerator` | `MultimodalToAudioGenerator` | `MultimodalToVideoGenerator` |

Classifiers live in `skai.classification`, extractors in `skai.extraction`, embedders in `skai.embedding`, segmenters
in `skai.segmentation` and generators in `skai.generation`. The pipeline primitives `MediaUnion`, `Field`, `ForEach`,
`TopK` and `Retriever` live in `skai.pipeline`.

## `operation` and `instructions`

Two parameters say what a component does:

- `operation` selects the procedure, and with it the backend task and the kind of model. It takes a fixed set of
  values per class, has a default, and is validated at `fit`. `AudioToTextGenerator(operation='transcribe')` runs a
  speech recognition model, and `ImageToImageGenerator(operation='upscale')` runs an upscaler.
- `instructions` is the content of a prompt, written in natural language. A summary, a translation, a description
  and an answer are all `TextToTextGenerator` or `ImageToTextGenerator` with different instructions.

The wording of `instructions` never changes the operation. [Presets](presets.md) give common instructions a name,
such as `presets.summarizer()` or `presets.translator('Greek')`.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.generation import ImageToTextGenerator

image = Image.new('RGB', (64, 48), 'white')
describer = ImageToTextGenerator(instructions='Describe the image.', backend='dummy').fit()
reader = ImageToTextGenerator(operation='transcribe', format='markdown', backend='dummy').fit()
print(describer.predict([image]), reader.predict([image]))
```

## Batches, failures and seeds

Every component that uses a backend takes the same execution parameters:

- `batch_size` is the number of samples submitted to the backend at once. Outputs always keep the input order.
- `on_error='raise'` raises `skai.core.SampleError` naming the first failed sample. `on_error='ignore'` returns `None`,
  or a row of NaN for numeric outputs, for every failed sample and emits one `SampleFailureWarning` that names them.
- `random_state` controls the selection of few-shot examples and the seeds sent to backends that support seeding.

## Next steps

- [Media inputs and outputs](media.md) lists the accepted samples, multimodal and list-valued samples, and output
  formats.
- [Backends](backends.md) explains how to choose where components run, and lists the providers of every operation.
- [Classification](classification.md), [extraction](extraction.md), [embedding](embedding.md),
  [segmentation](segmentation.md), [text generation](text-generation.md) and [media generation](generation.md)
  describe every component.
- [Pipelines](pipelines.md) shows the primitives and the scikit-learn chains of components.
- [Presets](presets.md) and [recipes](recipes.md) give configured components and pipelines for common tasks.
- [What fit learns](fit.md) explains learning from data, `use_y` in pipelines, and fine-tuning.
- [Migrating from 0.2](migration.md) maps every removed name to its replacement.
