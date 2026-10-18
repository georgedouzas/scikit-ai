# Spec Delta

## Purpose

Defines how components compose in scikit-learn pipelines: which outputs feed which inputs, the chains the library
supports and documents, and how fitting, caching and tuning behave across steps.

## ADDED Requirements

### Requirement: Step outputs are valid next-step inputs
The output of every generator SHALL be accepted as input by every component whose input type matches it, in every `output_format`: text outputs by text components, audio by audio components, images by image components, and videos by video components.

#### Scenario: Paths between steps
- **WHEN** `SpeechSynthesizer(output_format='path')` is followed by `SpeechTranscriber` in a pipeline
- **THEN** the transcriber reads the written files and returns one transcript per text

#### Scenario: Mismatched types
- **WHEN** `ImageGenerator` is followed by `TextSummarizer`
- **THEN** `predict` raises `InvalidSampleError` naming the first sample

### Requirement: Transcripts feed text components
A transcriber (`SpeechTranscriber` or `VideoTranscriber`) followed by `TextClassifier`, `TextSummarizer`, `TextTranslator`, `TextExtractor` or `TextEmbedder` SHALL form a working pipeline, covering call classification, meeting summaries, subtitle translation and form filling from speech.

#### Scenario: Summarize videos
- **WHEN** a pipeline of `VideoTranscriber` followed by `TextSummarizer` predicts on two videos
- **THEN** it returns two summaries

#### Scenario: Classify calls
- **WHEN** a pipeline of `SpeechTranscriber` followed by `TextClassifier` is fitted on recordings and labels
- **THEN** it predicts one label per recording

### Requirement: Descriptions feed text components
A descriptor (`ImageDescriptor`, `AudioDescriptor` or `VideoDescriptor`) followed by `TextClassifier`, `TextExtractor`, `TextEmbedder` or `TextTranslator` SHALL form a working pipeline, so media can be classified, searched or indexed through its descriptions.

#### Scenario: Extract product attributes from photos
- **WHEN** a pipeline of `ImageDescriptor` followed by `TextExtractor(fields={'color': '...', 'material': '...'})` predicts on images
- **THEN** it returns one dict per image with the keys `color` and `material`

### Requirement: Speech translation
A pipeline of `SpeechTranscriber`, `TextTranslator` and `SpeechSynthesizer` SHALL translate speech into speech in the target language.

#### Scenario: English speech to Greek speech
- **WHEN** the pipeline with `TextTranslator(target_language='Greek')` predicts on two recordings
- **THEN** it returns two `Audio` outputs

### Requirement: Narration of media
A descriptor followed by `SpeechSynthesizer`, optionally with `TextTranslator` between them, SHALL narrate images, sounds or videos, for example as spoken alt text.

#### Scenario: Spoken alt text
- **WHEN** a pipeline of `ImageDescriptor` followed by `SpeechSynthesizer` predicts on three images
- **THEN** it returns three `Audio` outputs

### Requirement: Talking presenter
A pipeline of `SpeechSynthesizer` followed by `TalkingAvatarGenerator`, optionally preceded by `TextTranslator` or `TextSummarizer`, SHALL turn texts into videos of a portrait that speaks them.

#### Scenario: Multilingual presenter
- **WHEN** a pipeline of `TextTranslator(target_language='Spanish')`, `SpeechSynthesizer` and `TalkingAvatarGenerator(avatar=portrait)` predicts on two scripts
- **THEN** it returns two videos whose durations match their synthesized speech

### Requirement: Visual generation chains
`ImageGenerator` followed by `ImageEditor`, `ImageAnimator`, or `ImageEditor` and then `ImageAnimator` SHALL form working pipelines from prompts to edited images and short videos.

#### Scenario: Prompt to animated scene
- **WHEN** a pipeline of `ImageGenerator`, `ImageEditor(instruction='make it night')` and `ImageAnimator` predicts on one prompt
- **THEN** it returns one video

### Requirement: Embeddings feed scikit-learn estimators
Every embedder SHALL work as the first step before any scikit-learn estimator that takes a numeric matrix, including classifiers, regressors, clustering and `NearestNeighbors`, and `FeatureUnion` SHALL combine embedders that take the same input type.

#### Scenario: Image search
- **WHEN** `NearestNeighbors` is fitted on `ImageEmbedder` vectors of a gallery and queried with `TextEmbedder` vectors from the same joint model
- **THEN** it returns the indices of the closest images for every text

#### Scenario: Combined text features
- **WHEN** a `FeatureUnion` of two `TextEmbedder`s with different backends feeds `LogisticRegression`
- **THEN** the classifier is fitted on the concatenated vectors

### Requirement: Pipeline steps run once per fit
Fitting a pipeline SHALL run each generator step once over the training samples, and a generator that is the final step SHALL call its backend during `fit` only to learn from the references it is given.

#### Scenario: One backend call per sample
- **WHEN** a pipeline of `SpeechTranscriber` and `TextClassifier` is fitted on 10 recordings
- **THEN** the transcription backend receives exactly 10 requests

### Requirement: Cached pipeline steps
Components whose backend is given as a string or `None` SHALL be hashable by `joblib`, so `Pipeline(memory=...)` reuses the cached outputs of generator steps when the pipeline is fitted again on the same samples with the same parameters.

#### Scenario: Refit with cache
- **WHEN** a cached pipeline of `ImageDescriptor` and `TextClassifier` is fitted twice on the same images
- **THEN** the description backend receives requests only during the first fit

### Requirement: Tuning across steps
Parameters of every step and of every step's backend SHALL be tunable with `GridSearchCV` using scikit-learn's `<step>__<parameter>` names.

#### Scenario: Tune two steps
- **WHEN** `GridSearchCV` searches `transcriber__language` and `textclassifier__backend__model` over a transcriber and classifier pipeline
- **THEN** every combination is fitted and scored
