# Spec Delta

## ADDED Requirements

### Requirement: Summarizer learns from reference summaries
`TextSummarizer.fit(X, y)`, with reference summaries as `y`, SHALL select `k_shot` text and summary pairs as `examples_`, and, when `max_length` is `None`, SHALL set `max_length_` to the 90th percentile of the word counts of the references.

#### Scenario: Learned length
- **WHEN** `TextSummarizer(max_length=None, backend='dummy')` is fitted with references of 8 to 12 words
- **THEN** `max_length_` is between 8 and 12 and predictions have at most `max_length_` words

#### Scenario: Pairs are sent as examples
- **WHEN** it is fitted with reference summaries and `k_shot=2`
- **THEN** every summary request carries two example pairs

### Requirement: Translator learns from reference translations
`TextTranslator.fit(X, y)`, with reference translations as `y`, SHALL select `k_shot` text and translation pairs as `examples_`, and SHALL learn a `glossary_` of terms that appear in at least two references, which is sent with every request so terms are translated consistently.

#### Scenario: Glossary from references
- **WHEN** it is fitted on pairs where a product name keeps the same translation in two references
- **THEN** `glossary_` maps that name to its translation

### Requirement: Text components can be fine-tuned
`TextSummarizer` and `TextTranslator` SHALL support `adaptation='finetune'` from texts and their references.

#### Scenario: Fine-tuned summarizer
- **WHEN** `TextSummarizer(adaptation='finetune', backend='dummy')` is fitted with 20 texts and their summaries
- **THEN** `model_` is set and predictions are produced by that model
