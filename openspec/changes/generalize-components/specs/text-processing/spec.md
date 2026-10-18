# Spec Delta

## REMOVED Requirements

### Requirement: TextSummarizer
**Reason**: Tasks are text generation named by input and output.
**Migration**: Use `TextToTextGenerator(instructions='Summarize ...', max_length=...)`.

### Requirement: TextTranslator
**Reason**: Tasks are text generation named by input and output.
**Migration**: Use `TextToTextGenerator(instructions='Translate into <language>.', learn_terms=True)`.

### Requirement: Summarizer learns from reference summaries
**Reason**: Examples and a learned length are part of `TextToTextGenerator`.
**Migration**: Use `max_length='auto'` and fit with reference summaries.

### Requirement: Translator learns from reference translations
**Reason**: Examples and a glossary are part of `TextToTextGenerator`.
**Migration**: Use `learn_terms=True` and fit with reference translations.

### Requirement: Text components can be fine-tuned
**Reason**: Fine-tuning is part of every text generator.
**Migration**: Use `TextToTextGenerator(adaptation='finetune')`.

### Requirement: TextExtractor
**Reason**: Extractors have their own capability for every input type.
**Migration**: See `TextExtractor` in the `extraction` capability.

### Requirement: TextExtractor returns feature columns
**Reason**: Feature columns are part of every extractor.
**Migration**: See the `extraction` capability.
