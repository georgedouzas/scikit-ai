# Design

## Context

`BaseGenerator.fit` validates parameters and resolves the backend. `BaseClassifier.fit` learns classes and few-shot
examples, and `BaseEmbedder.fit` only probes the dimensions. Requests are frozen dataclasses built per sample by
`_build_request`, and prompted backends turn them into prompts. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- One mechanism for few-shot pairs that every supervised generator reuses.
- Pipelines that keep working when `y` is the final target.

**Non-Goals:**
- Training local models. Fine-tuning covers providers with a fine-tuning API only.
- Learning that needs more than one pass of backend calls over the data.

## Decisions

### Named references with metadata routing
Generators declare their reference parameter in a class variable, such as `reference_param = 'summaries'`, and
`BaseGenerator.fit(X=None, y=None, **references)` validates that only that name is passed. scikit-learn builds
`set_fit_request` from the `fit` signature, so each generator overrides `fit` with an explicit keyword, for example
`fit(self, X=None, y=None, *, summaries=None)`, and delegates to the base. That makes routing work with
`sklearn.set_config(enable_metadata_routing=True)`. Reading `y` only outside pipelines was the alternative. It was
rejected because a component cannot tell whether it is inside a pipeline.

### Shared few-shot pairs
`BaseGenerator` gains `_select_pairs(samples, references)`, the same selection as the classifiers'
`_select_examples`, moved to one helper both bases use. `k_shot` on generators defaults to `None` (all references,
capped at `max_examples=8` to bound prompt size), `0` disables examples, and an integer or indices select them.
Requests that take examples (`SummaryRequest`, `TranslationRequest`, `DescriptionRequest`) gain
`examples: tuple[tuple[sample, str], ...]`, and the prompt builders render them as for classification.

### What each component learns
- **Summarizer length.** `max_length=None` sets `max_length_` from the 90th percentile of reference word counts.
  The default stays `100`, so existing behavior holds.
- **Translator glossary.** For terms that appear unchanged in at least two source texts (capitalized words and words
  with digits), the glossary keeps the target word that appears in every matching reference. It is a small
  heuristic, sent as `Glossary: term → translation` lines, and it is skipped when nothing matches.
- **Transcriber vocabulary.** Terms are extracted with a regex and filtered against a small bundled word list
  (`core/_words.txt`, the 10,000 most common English words). OpenAI receives them as the `prompt` of the
  transcription call, and Hugging Face Whisper as `prompt_ids`. The `dummy` backend echoes them in its transcript, so
  tests can see them. Backends declare `accepts_context: ClassVar[bool]`.
- **Embedder projection.** `fit` embeds `X` once. The PCA is fitted with NumPy SVD on the centered embeddings, so
  no scikit-learn decomposition objects are stored. Backends declare `shortens_embeddings: ClassVar[bool]`, which is
  `True` for OpenAI. The probe sample is no longer needed when `X` is given.
- **Synthesizer voice.** The samples are mixed to mono, resampled to the first sample's rate by linear
  interpolation, joined with 0.3 seconds of silence, and capped at 30 seconds.
- **Generator style.** The backend describes every reference image with `instructions='Describe only the visual
  style, not the content.'`, and the descriptions are joined. `style_` is kept short by asking for one sentence per
  image.
- **Avatar portrait.** Backends declare `avatar_size: ClassVar[int | None]`. `None` keeps the cropped size.

### Fine-tuning
`Backend.fine_tune(request: FineTuneRequest) -> str` is a new task. `FineTuneRequest` holds the task name and the
training prompts with their expected answers, built by the component from its own prompt builders, so a fine-tuned
model sees the same prompt format at prediction time. `OpenAIBackend` writes the chat examples as JSONL (images as
data URLs), uploads the file, creates the job, and polls it every 15 seconds up to `finetune_timeout` (default one
hour). Then it returns `fine_tuned_model`. The minimum is OpenAI's 10 examples. With `adaptation='finetune'`, the
component stores `model_` and sends requests to a clone of its backend whose `model` is `model_`, with no examples.

## Risks / Trade-offs

- [`fit` now costs money for `TextEmbedder`, `ImageGenerator` with style images, and fine-tuning] → The docs state the cost of every fit, and pipelines with `memory=` avoid refits.
- [Prompts grow with examples] → `max_examples` caps them, and `k_shot=0` turns examples off.
- [The glossary and vocabulary heuristics are crude] → They are small, documented, visible as fitted attributes, and easy to replace later.
- [Fine-tuning jobs can take long and fail] → A timeout names the job id, so the user can pass the finished model later as `model=`.
- [`k_shot=None` changes meaning for classifiers] → 0.2 is unreleased, and the migration page documents it.
