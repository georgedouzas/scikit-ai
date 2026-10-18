# Spec Delta

## ADDED Requirements

### Requirement: MultimodalClassifier
The library SHALL provide `MultimodalClassifier` in `skai.classification`, a classifier whose samples are multimodal, with the labels, instructions, few-shot and fine-tuning behavior of the other classifiers, sending every named part of a sample to the backend in one request.

#### Scenario: Post with image and caption
- **WHEN** `MultimodalClassifier(labels=['safe', 'unsafe'], backend='dummy').fit()` predicts on `[{'image': photo, 'caption': 'hello'}]`
- **THEN** it returns one label from `labels`, and the request carries the image and the caption, each named by its key
