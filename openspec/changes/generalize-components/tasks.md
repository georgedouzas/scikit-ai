# Tasks

## 1. Core

- [ ] 1.1 Declare input kind, output kind and operations as class variables of the bases, resolve the backend task from `operation` at `fit`, add the multimodal reader (dicts and DataFrame rows with required keys) and list-valued sample checks, and move frame sampling into the video reader, with tests for every reader, missing keys, and list samples reaching ordinary components
- [ ] 1.2 Add the backend tasks and request types of the new operations, merge `summarize`, `translate` and `describe_*` into `generate_text`, and implement every task on `dummy`, with tests for deterministic, correctly typed outputs

## 2. Classes

- [ ] 2.1 `MultimodalClassifier` and `VideoEmbedder`, with tests for their scenarios
- [ ] 2.2 The five extractors with nested and list fields and feature columns, with tests for the `extraction` scenarios
- [ ] 2.3 The four segmenters with every `by`, including local segmentation without a provider and the vocabulary of timed speech, with tests for the `segmentation` scenarios
- [ ] 2.4 The five text generators with the instruct and transcribe operations, examples, `max_length='auto'`, `learn_terms`, the vocabulary and fine-tuning, with tests for the `text-generation` scenarios
- [ ] 2.5 The twelve media generators with every operation, the local operations, the learned voice, style and portrait, and the multimodal key checks, with tests for the `generation` scenarios

## 3. Primitives, presets and recipes

- [ ] 3.1 `MediaUnion`, `Field`, `ForEach` and `TopK`, with tests for the `pipeline-primitives` scenarios, including batching, parameter routing, cloning and `Pipeline(memory=...)`
- [ ] 3.2 `skai.presets` with every listed preset and its instruction table, with tests that each preset returns the documented class
- [ ] 3.3 `skai.recipes` with the 16 pipelines, with one `dummy` test per recipe requirement of the `pipelines` capability

## 4. Removal

- [ ] 4.1 Delete `skai.text`, `skai.description` and `skai.transcription` and every removed 0.2 class, move their tests to the new classes, and verify the module-layout scenarios, including the grid naming test

## 5. Providers

- [ ] 5.1 Implement `generate_text`, `extract_fields`, `transcribe_image`, `detect_objects` and `find_moments` for every input kind on OpenAI and LiteLLM through prompting, and on Hugging Face where a local model exists, with fake-client tests
- [ ] 5.2 Implement the Hugging Face models of segmentation, diarization, separation, enhancement, upscaling, background removal, music and lip sync, with fake-module tests
- [ ] 5.3 Port the end-to-end suite to the new classes, add OpenAI scenarios for every operation OpenAI supports and for the recipes whose steps it supports, record the payloads, and verify they pass
- [ ] 5.4 Add Hugging Face end-to-end scenarios for the new local models that fit on the development machine, behind `SKAI_E2E_HF=1`, and verify they pass
- [ ] 5.5 Run the Sora scenarios behind `SKAI_E2E_VIDEO=1`, and verify the defaults that need other credentials or larger hardware (Gemini, Anthropic, Cohere, Qwen2-Audio-7B, Wan, InstructPix2Pix, Stable Video Diffusion), carried over from `add-roadmap-components` (blocked on access)

## 6. Docs and release

- [ ] 6.1 Rewrite the guide around the grid, add a presets page with the instruction table and a recipes page with every recipe as an executed example, update the README tables, and write the 0.3 migration page mapping every removed name, and verify `pdm docs build` passes in strict mode
- [ ] 6.2 Update the constitution's Project Profile, then run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order and verify they all pass

## Workflow follow-up

- Archive the change, which retires the `description`, `text-processing` and `transcription` specs, and release 0.3.0.
