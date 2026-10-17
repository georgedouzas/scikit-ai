# Spec Delta

## ADDED Requirements

### Requirement: Fine-tuning task
Backends that support fine-tuning SHALL implement a `fine_tune` task that takes the task name and the training examples, trains a model, waits for it within `finetune_timeout` seconds, and returns the model id. They SHALL declare their minimum number of examples.

#### Scenario: Timeout
- **WHEN** a fine-tuning job does not finish within `finetune_timeout`
- **THEN** `fit` raises `TimeoutError` naming the job id, so the user can resume it later with `model=<id>`

#### Scenario: Dummy fine-tuning
- **WHEN** the `dummy` backend fine-tunes on any examples
- **THEN** it returns a deterministic model id without waiting
