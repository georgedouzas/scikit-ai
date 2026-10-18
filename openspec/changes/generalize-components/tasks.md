# Tasks

## 1. Core

- [ ] 1.1 Compute `task` and the reader from `input` and `operation` at `fit`, add the `'record'` reader and list-valued sample checks, and move frame sampling into the video reader, with tests for every input type, invalid `input`, and list samples reaching ordinary components
- [ ] 1.2 Add the backend tasks and request types of the new operations, merge `summarize`, `translate` and `describe_*` into `generate_text` with a fallback to 0.2 custom backends, and implement every task on `dummy`, with tests for deterministic, correctly typed outputs

## 2. General components

- [ ] 2.1 `Classifier` for every input type, including records and fine-tuning, with tests for the `classification` scenarios
- [ ] 2.2 `TextGenerator` with instructions, records, examples, `max_length='auto'`, `learn_terms` and fine-tuning, with tests for the `text-generation` scenarios
- [ ] 2.3 `Extractor` for every input type with nested and list fields and feature columns, with tests for the `extraction` scenarios
- [ ] 2.4 `Embedder` for every input type, including video as the mean of frame embeddings, with tests for the `embedding` scenarios
- [ ] 2.5 `Transcriber` for audio, video and images, with speakers and layout, with tests for the `transcription` scenarios
- [ ] 2.6 `Segmenter` with image boxes and masks, video shots and moments, audio silence, speakers and moments, and text splitting, with tests for the `segmentation` scenarios, including local segmentation without a provider
- [ ] 2.7 `ImageGenerator`, `AudioGenerator` and `VideoGenerator` with every operation, the learned voice, style and portrait, with tests for the `generation` scenarios
- [ ] 2.8 `Composer` with concatenation, mixing with ducking, muxing and subtitles, with tests for the `composition` scenarios

## 3. Primitives, presets and recipes

- [ ] 3.1 `MediaUnion`, `Pick`, `ForEach` and `Select`, with tests for the `pipeline-primitives` scenarios, including batching, parameter routing, cloning and `Pipeline(memory=...)`
- [ ] 3.2 `skai.presets` with every listed preset and its instruction table, with tests that each preset returns the documented component
- [ ] 3.3 `skai.recipes` with the 16 pipelines, with one `dummy` test per recipe requirement of the `pipelines` capability

## 4. Providers

- [ ] 4.1 Implement `generate_text`, `extract_fields`, `transcribe_image` and `detect_objects` for every input type on OpenAI and LiteLLM through prompting, and on Hugging Face where a local model exists, with fake-client tests
- [ ] 4.2 Implement the Hugging Face models of segmentation, diarization, separation, enhancement, upscaling, background removal, music and lip sync, with fake-module tests
- [ ] 4.3 Add OpenAI end-to-end scenarios for every component and operation OpenAI supports, and for the recipes that use only OpenAI-supported steps, record the payloads, and verify they pass
- [ ] 4.4 Add Hugging Face end-to-end scenarios for the new local models that fit on the development machine, behind `SKAI_E2E_HF=1`, and verify they pass
- [ ] 4.5 Run the Sora scenarios behind `SKAI_E2E_VIDEO=1`, and verify the defaults that need other credentials or larger hardware (Gemini, Anthropic, Cohere, Qwen2-Audio-7B, Wan, InstructPix2Pix, Stable Video Diffusion), carried over from `add-roadmap-components` (blocked on access)

## 5. Deprecation and docs

- [ ] 5.1 Turn every 0.2 class into a deprecated subclass in its 0.2 module with one `FutureWarning`, with tests for the `deprecations` scenarios and that 0.2 code from the docs of 0.2 still runs
- [ ] 5.2 Rewrite the guide around the 10 components, add a presets page with the instruction table and a recipes page with every recipe as an executed example, update the README tables and the migration page, and verify `pdm docs build` passes in strict mode
- [ ] 5.3 Update the constitution's Project Profile and the module-layout tests, then run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order and verify they all pass

## Workflow follow-up

- Archive the change, which retires the `description` and `text-processing` specs, and release 0.3.0.
