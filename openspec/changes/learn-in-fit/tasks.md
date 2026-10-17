# Tasks

## 1. Shared mechanics

- [ ] 1.1 Move example selection into one helper used by classifiers and generators, add `k_shot` and `max_examples` to the supervised generators, and add `examples` to the summary, translation and description requests and their prompt builders, with tests for selection, caps and rendering
- [ ] 1.2 Make supervised generators read `y` as their targets in `fit` and ignore it in `fit_transform`, add `use_y`, validate target lengths, and test the pipeline, override, cross-validation and mismatched-length scenarios
- [ ] 1.3 Change the classifier `k_shot=None` default to one example per class, and test the default and the `k_shot=0` scenarios

## 2. What each component learns

- [ ] 2.1 `TextSummarizer`: examples from reference summaries and `max_length_`, with tests for the learned length and the example pairs
- [ ] 2.2 `TextTranslator`: examples and `glossary_` from reference translations, with tests for the glossary scenario
- [ ] 2.3 `ImageDescriptor`: examples from reference descriptions, with tests on `dummy` and the OpenAI recordings
- [ ] 2.4 `SpeechTranscriber`: `vocabulary_` from reference transcripts, `accepts_context`, and context in the OpenAI, Hugging Face and `dummy` transcription calls, with tests for both scenarios
- [ ] 2.5 `TextEmbedder`: `mean_`, `center`, PCA `components_` and `shortens_embeddings`, with tests for centering, PCA and native shortening
- [ ] 2.6 `SpeechSynthesizer`: `voice_` from one or more recordings in `voice`, with tests for joining samples and the unsupported-backend error
- [ ] 2.7 `ImageGenerator`: the `style_images` parameter and `style_`, with tests for the learned style, the precedence of `style` and the unsupported-backend error
- [ ] 2.8 `TalkingAvatarGenerator`: prepared `avatar_` and `avatar_size`, with tests for cropping and resizing

## 3. Fine-tuning

- [ ] 3.1 Add the `fine_tune` task, `FineTuneRequest`, `adaptation` and `model_`, and the `dummy` implementation, with tests for the supported, unsupported and too-little-data scenarios
- [ ] 3.2 Implement OpenAI fine-tuning (JSONL upload, job creation, polling, timeout), with fake-client tests for success, failure and timeout

## 4. Verification and docs

- [ ] 4.1 Add end-to-end OpenAI tests for every learned state (few-shot summaries and descriptions, glossary, vocabulary, centered and projected embeddings), plus fine-tuning behind `SKAI_E2E_FINETUNE=1`, record the payloads, and verify they pass
- [ ] 4.2 Add a "What fit learns" section to every guide page and a fit table to the README, update the migration page for `k_shot`, and verify `pdm docs build` passes
- [ ] 4.3 Run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order, and verify they all pass
