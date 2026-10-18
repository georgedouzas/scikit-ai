# Spec Delta

## REMOVED Requirements

### Requirement: TextSummarizer
**Reason**: Summaries are text generation with instructions.
**Migration**: Use `TextGenerator(instructions='Summarize ...', max_length=...)`, or `skai.presets.summarizer(max_length=...)`.

### Requirement: TextTranslator
**Reason**: Translations are text generation with instructions.
**Migration**: Use `TextGenerator(instructions='Translate into <language>.', learn_terms=True)`, or `skai.presets.translator(target_language)`.

### Requirement: Summarizer learns from reference summaries
**Reason**: Examples and a learned length are part of `TextGenerator`.
**Migration**: Use `TextGenerator(max_length='auto')` fitted with reference summaries.

### Requirement: Translator learns from reference translations
**Reason**: Examples and a glossary are part of `TextGenerator`.
**Migration**: Use `TextGenerator(learn_terms=True)` fitted with reference translations.

### Requirement: Text components can be fine-tuned
**Reason**: Fine-tuning is part of `TextGenerator`.
**Migration**: Use `TextGenerator(adaptation='finetune', ...)`.

### Requirement: TextExtractor
**Reason**: Extraction works on every input type.
**Migration**: Use `skai.extraction.Extractor(fields=...)`.

### Requirement: TextExtractor returns feature columns
**Reason**: Feature columns are part of `Extractor`.
**Migration**: Use `Extractor.transform` and `get_feature_names_out`.
