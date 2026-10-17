# Spec Delta

## Purpose

Lets users run any component on open-source models or cloud APIs through one `backend` parameter, given
either as a short string or as a configurable backend object, without changing the component's interface.

## ADDED Requirements

### Requirement: Backend selected by string
Every component SHALL accept `backend` as a string `"<provider>"` or `"<provider>:<model>"`. A provider-only string SHALL use that provider's documented default model for the component's task.

#### Scenario: Provider and model string
- **WHEN** `ImageDescriptor(backend="openai:gpt-5-mini")` is fitted
- **THEN** the component uses the OpenAI provider with model `gpt-5-mini`

#### Scenario: Provider-only string
- **WHEN** `SpeechTranscriber(backend="hf")` is fitted
- **THEN** the component uses the Hugging Face provider's documented default transcription model

### Requirement: Backend selected by object
Every component SHALL accept `backend` as a backend object carrying provider settings (model, credentials, client, endpoint, generation options). Backend objects SHALL support `get_params`/`set_params` so nested parameters such as `backend__model` can be tuned with `GridSearchCV`.

#### Scenario: Nested parameter search
- **WHEN** `GridSearchCV` searches `{"backend__model": [m1, m2]}` over a `TextClassifier` with a backend object
- **THEN** each candidate is fitted with the corresponding model and results are reported per model

#### Scenario: Injected client
- **WHEN** a backend object is given a preconfigured provider client
- **THEN** the component uses that client instead of creating one

### Requirement: Default backend
Every component SHALL define a documented default backend used when `backend=None`. The default SHALL be an open-source model, run locally when a packaged local implementation exists and otherwise through a hosted provider.

#### Scenario: Local default
- **WHEN** `TextSummarizer()` is fitted with the local open-source extra installed
- **THEN** it uses the documented local default model without any network credentials

#### Scenario: Default backend without its extra
- **WHEN** a component is fitted with `backend=None` and the required optional dependency is not installed
- **THEN** `ImportError` is raised naming the `pip install scikit-ai[<extra>]` command

### Requirement: Open-source and cloud coverage
Every first-release component SHALL support at least one cloud API provider and at least one open-source model. An open-source model MAY run locally or through a hosted provider.

#### Scenario: Documented backend matrix
- **WHEN** a user reads a component's API documentation
- **THEN** it lists every supported provider, the default model per provider, and whether each runs locally or in the cloud

### Requirement: Unsupported backend reported at fit
A component SHALL raise `ValueError` at `fit` when the provider is unknown or does not support the component's task, listing the providers that do.

#### Scenario: Provider lacks task
- **WHEN** `TalkingAvatarGenerator(backend="openai")` is fitted and OpenAI does not support that task
- **THEN** `ValueError` is raised listing the supported providers

### Requirement: Credentials handling
Cloud backends SHALL read credentials from explicit backend-object arguments first, then from the provider's standard environment variables. Credentials SHALL NOT appear in `repr`, `get_params` output, logs, warnings or exception messages.

#### Scenario: Missing credentials
- **WHEN** a cloud backend is fitted with no credentials available
- **THEN** an error names the environment variable to set, without printing any secret

#### Scenario: Secret hidden from repr
- **WHEN** a backend object constructed with `api_key="sk-123"` is printed
- **THEN** the output does not contain `sk-123`

### Requirement: Custom backend registration
The library SHALL provide a public way to register a third-party provider under a name, declaring which tasks it supports, after which the name is usable in backend strings.

#### Scenario: Registered provider used by string
- **WHEN** a user registers provider `"acme"` for the summarization task and fits `TextSummarizer(backend="acme:v1")`
- **THEN** the component calls the registered provider with model `v1`

### Requirement: Offline test backend
The library SHALL ship a `"dummy"` provider that supports every task, needs no network or model weights, and returns deterministic, valid-typed outputs, for use in tests and documentation.

#### Scenario: Dummy backend in a pipeline
- **WHEN** any component is fitted and used with `backend="dummy"` without network access
- **THEN** it returns outputs of the documented type and shape, identical across runs
