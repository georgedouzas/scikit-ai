# Spec Delta

## MODIFIED Requirements

### Requirement: Open-source and cloud coverage
Every component SHALL support at least one cloud API provider and at least one open-source model. An open-source model MAY run locally or through a hosted provider. `AudioEmbedder` is the one exception, with open-source models only, until a cloud provider offers audio embeddings.

#### Scenario: Documented backend matrix
- **WHEN** a user reads a component's API documentation
- **THEN** it lists every supported provider, the default model per provider, and whether each runs locally or in the cloud
