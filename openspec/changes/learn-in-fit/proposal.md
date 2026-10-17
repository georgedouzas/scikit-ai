# Proposal

## Why

Only the classifiers learn anything in `fit`. Every generator and the embedder ignore their data, so `fit` is a
formality. That wastes the main promise of a scikit-learn API: give a component examples of what you want, and it
gets better at your task. Full training is out of reach for most of these models, but two cheaper kinds of fit are
available for every component:

- **Inference-time fit**, which learns state from the data and uses it at prediction time. Examples are few-shot
  demonstrations, a learned vocabulary, a learned style, a cloned voice, or a learned projection.
- **Fine-tuning**, where the provider supports it, which trains a new model from the data.

## What Changes

- Every component SHALL learn from the data given to `fit`. Fitting without data stays possible and means zero-shot.
- Generators take their references through named fit parameters, such as `summaries=` and `transcripts=`, and ignore
  `y`. In a pipeline, `y` is the final step's target and would otherwise be read as references. The parameters can
  be routed to one step with scikit-learn metadata routing.
- **BREAKING** `k_shot=None` on classifiers selects one example per class, and `k_shot=0` is zero-shot.
- What each component learns:

| Component | Fit data | Learned state | Used at prediction |
|---|---|---|---|
| `TextClassifier`, `ImageClassifier` | `X`, `y` | `classes_`, `examples_` | few-shot examples (unchanged, new default) |
| `TextSummarizer` | `X`, `summaries=` | `examples_`, `max_length_` | few-shot pairs and a learned length |
| `TextTranslator` | `X`, `translations=` | `examples_`, `glossary_` | few-shot pairs and term translations |
| `ImageDescriptor` | `X`, `descriptions=` | `examples_` | few-shot image and description pairs |
| `SpeechTranscriber` | `X`, `transcripts=` | `vocabulary_` | a context prompt of domain terms |
| `TextEmbedder` | `X` | `mean_`, `components_` | centering, and PCA when `dimensions` is set and the backend cannot shorten |
| `SpeechSynthesizer` | `voice_samples=` | `voice_` | voice cloning from the samples |
| `ImageGenerator` | `style_images=` | `style_` | a style description learned from reference images |
| `TalkingAvatarGenerator` | `avatar` | `avatar_` | the portrait cropped and sized for the backend |

- Add `adaptation='finetune'` to `TextClassifier`, `ImageClassifier`, `TextSummarizer`, `TextTranslator` and
  `ImageDescriptor` on backends that support fine-tuning (OpenAI first). Fitting trains a model and stores its id
  in `model_`.

### Out of scope

- Local fine-tuning (LoRA and similar) on the `hf` backend.
- Learning from data in `TalkingAvatarGenerator` beyond preparing the portrait. The talking-head models in reach
  take one image and no training data.

## Capabilities

### Modified Capabilities
- `media-estimator-api`: fit learns from data, generators take references through named parameters, and `k_shot`
  gets a new default.
- `classification`: fine-tuning.
- `text-processing`: summaries and translations learned from references, and fine-tuning.
- `transcription`: vocabulary learned from reference transcripts.
- `description`: style learned from reference descriptions, and fine-tuning.
- `embedding`: centering and PCA learned from the samples.
- `generation`: voice, style and portrait learned in `fit`.
- `backends`: a fine-tuning task.

## Impact

- `fit` now calls backends for `ImageGenerator` (to describe the style images), for `TextEmbedder` (to embed `X`),
  and for every component with `adaptation='finetune'`. The `pipelines` requirement on backend calls during `fit`
  in `add-roadmap-components` is updated to match.
- The roadmap components in `add-roadmap-components` follow the same rule through the shared requirement.
- Docs, the migration page and the end-to-end suite gain fit examples. Fine-tuning runs end to end only behind
  `SKAI_E2E_FINETUNE=1`, because it costs money and takes minutes.
