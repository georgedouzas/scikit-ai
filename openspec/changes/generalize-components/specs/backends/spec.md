# Spec Delta

## MODIFIED Requirements

### Requirement: Open-source and cloud coverage
Every operation of every component SHALL be supported by at least one open-source model, locally or through a hosted provider, and by every cloud provider that offers it. The documentation SHALL list, per component and operation, every supported provider and default model, and operations that no cloud provider offers SHALL be marked as open-source only.

#### Scenario: Documented backend matrix
- **WHEN** a user reads a component's API documentation
- **THEN** it lists every operation with its supported providers, the default model per provider, and whether each runs locally or in the cloud

### Requirement: Backend selected by string
Every component SHALL accept `backend` as a string `"<provider>"` or `"<provider>:<model>"`. A provider-only string SHALL use that provider's documented default model for the component's task.

#### Scenario: Provider and model string
- **WHEN** `TextGenerator(input='image', instructions='Describe the image.', backend="openai:gpt-5-mini")` is fitted
- **THEN** the component uses the OpenAI provider with model `gpt-5-mini`

#### Scenario: Provider-only string
- **WHEN** `Transcriber(backend="hf")` is fitted
- **THEN** the component uses the Hugging Face provider's documented default transcription model

### Requirement: Backend selected by object
Every component SHALL accept `backend` as a backend object carrying provider settings (model, credential variable name, client, endpoint, generation options). Backend objects SHALL support `get_params`/`set_params` so nested parameters such as `backend__model` can be tuned with `GridSearchCV`.

#### Scenario: Nested parameter search
- **WHEN** `GridSearchCV` searches `{"backend__model": [m1, m2]}` over a `Classifier` with a backend object
- **THEN** each candidate is fitted with the corresponding model and results are reported per model

#### Scenario: Injected client
- **WHEN** a backend object is given a preconfigured provider client
- **THEN** the component uses that client instead of creating one

### Requirement: Default backend
Every component SHALL define a documented default backend used when `backend=None`. The default SHALL be an open-source model, run locally when a packaged local implementation exists and otherwise through a hosted provider.

#### Scenario: Local default
- **WHEN** `TextGenerator(instructions='Summarize the text.')` is fitted with the local open-source extra installed
- **THEN** it uses the documented local default model without any network credentials

#### Scenario: Default backend without its extra
- **WHEN** a component is fitted with `backend=None` and the required optional dependency is not installed
- **THEN** `ImportError` is raised naming the `pip install scikit-ai[<extra>]` command

### Requirement: Unsupported backend reported at fit
A component SHALL raise `ValueError` at `fit` when the provider is unknown or does not support the component's task, listing the providers that do.

#### Scenario: Provider lacks task
- **WHEN** `VideoGenerator(operation='lip_sync', backend="openai")` is fitted and OpenAI does not support that operation
- **THEN** `ValueError` is raised listing the supported providers

### Requirement: Custom backend registration
The library SHALL provide a public way to register a third-party provider under a name, declaring which tasks it supports, after which the name is usable in backend strings.

#### Scenario: Registered provider used by string
- **WHEN** a user registers provider `"acme"` for the text generation task and fits `TextGenerator(instructions='Summarize the text.', backend="acme:v1")`
- **THEN** the component calls the registered provider with model `v1`
