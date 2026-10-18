# Spec Delta

## ADDED Requirements

### Requirement: Recipe: brief to explainer video
`explainer_video` SHALL turn a brief into one narrated, subtitled video with background music: a storyboard extractor, one image animated per scene, narration per scene, concatenation, music mixed under the narration, and burned subtitles.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.explainer_video(backend='dummy').fit(None).predict(['How solar panels work'])` is called
- **THEN** it returns one video with an audio track and subtitles

### Requirement: Recipe: articles to podcast
`podcast` SHALL turn articles into one audio episode: summaries, a two-host script, speech in two voices joined in order, and generated intro music mixed in.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.podcast(backend='dummy').fit(None).predict([article])` is called
- **THEN** it returns one `Audio` episode per article

### Requirement: Recipe: post to vertical clip
`vertical_clip` SHALL turn a post into a short vertical video: a shortened text, the explainer steps, reframing to 9:16 and burned subtitles.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.vertical_clip(backend='dummy').fit(None).predict([post])` is called
- **THEN** it returns one video whose height is 16/9 of its width

### Requirement: Recipe: story to read-along video
`read_along` SHALL turn a story into a video whose characters look the same on every page, using one set of reference images for every generated image and word-timed subtitles.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.read_along(backend='dummy').fit(None).predict([story])` is called
- **THEN** it returns one video and every image request carries the same reference images

### Requirement: Recipe: product photos to catalog and ads
`product_catalog` SHALL turn product photos into catalog records (attributes and marketing copy from background-free photos), and `product_ad` SHALL turn them into short promotional clips (edited scenes, upscaled and animated).

#### Scenario: Runs on the dummy backend
- **WHEN** both recipes predict on two product photos on `dummy`
- **THEN** the catalog returns two records with the attribute fields and the ad recipe returns two videos

### Requirement: Recipe: dubbing in the original voice
`dubbing` SHALL turn a video into the same video speaking another language in the original speaker's voice, lip-synced, with the original music and effects kept and subtitles in the new language.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.dubbing(target_language='Greek', backend='dummy').fit(None).predict([video])` is called
- **THEN** it returns one video as long as the input whose voice request carries the separated vocals

### Requirement: Recipe: image text translation
`image_translation` SHALL redraw the text of images, such as comics or screenshots, in another language: transcribed text, translated, its regions masked and edited.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.image_translation(target_language='Greek', backend='dummy').fit(None).predict([image])` is called
- **THEN** it returns one image of the input size

### Requirement: Recipe: meeting to briefing
`meeting_briefing` SHALL turn a meeting recording into a record with `minutes`, a list of `action_items` with owners, and `video`, a short briefing read by a talking head.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.meeting_briefing(avatar=portrait, backend='dummy').fit(None).predict([recording])` is called
- **THEN** it returns one record with the keys `minutes`, `action_items` and `video`

### Requirement: Recipe: call analytics
`call_topics` SHALL label the topic and sentiment of every speaker turn of recorded calls and cluster the turns by meaning.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.call_topics(n_clusters=3, backend='dummy').fit(recordings).predict(recordings)` is called
- **THEN** it returns one list of turn records per call, each with `speaker`, `topic`, `sentiment` and `cluster`

### Requirement: Recipe: invoices to anomaly scores
`invoice_anomalies` SHALL turn invoice scans into vendor, amount and date fields and score them with an isolation forest.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.invoice_anomalies(backend='dummy').fit(scans).predict(scans)` is called
- **THEN** it returns one anomaly label per scan

### Requirement: Recipe: searching a video library by text
`video_search` SHALL index the shots of a video library and return, for every text query, the closest shots.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.video_search(backend='dummy').fit(videos)` is queried with `kneighbors_texts(['a goal'], n_neighbors=2)`
- **THEN** it returns two shot records per query, each with its video index, `start` and `end`

### Requirement: Recipe: multimodal moderation
`moderation` SHALL classify posts given as records of an image and a caption by combining safety labels of both into one calibrated classifier trained on labeled posts.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.moderation(backend='dummy').fit(posts, labels).predict_proba(posts)` is called
- **THEN** it returns one probability row per post

### Requirement: Recipe: audio description track
`audio_description` SHALL add a spoken description of every shot to a video, placed in the pauses of the dialogue, plus subtitles of the dialogue.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.audio_description(backend='dummy').fit(None).predict([video])` is called
- **THEN** it returns one video as long as the input

### Requirement: Recipe: lecture to study pack
`study_pack` SHALL turn a lecture video into a record with `notes`, `key_terms`, `quiz` and `recap`, a narrated audio summary.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.study_pack(backend='dummy').fit(None).predict([lecture])` is called
- **THEN** it returns one record with the keys `notes`, `key_terms`, `quiz` and `recap`

### Requirement: Recipe: visual quality control
`quality_control` SHALL classify product photos as defective or not by combining a visual question answer and detected defect regions into one classifier trained on labeled photos.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.quality_control(backend='dummy').fit(photos, labels).predict(photos)` is called
- **THEN** it returns one label per photo

### Requirement: Recipe: highlight reel
`highlights` SHALL turn a long video into a reel of its `top_k` most exciting shots in their original order, with generated music mixed in.

#### Scenario: Runs on the dummy backend
- **WHEN** `skai.recipes.highlights(top_k=3, backend='dummy').fit(None).predict([match])` is called
- **THEN** it returns one video made of at most three shots

## MODIFIED Requirements

### Requirement: Step outputs are valid next-step inputs
The output of every generator SHALL be accepted as input by every component whose input type matches it, in every `output_format`: text outputs by text components, audio by audio components, images by image components, and videos by video components.

#### Scenario: Paths between steps
- **WHEN** `AudioGenerator(output_format='path')` is followed by `Transcriber` in a pipeline
- **THEN** the transcriber reads the written files and returns one transcript per text

#### Scenario: Mismatched types
- **WHEN** `ImageGenerator` is followed by `TextGenerator(instructions='Summarize.')`
- **THEN** `predict` raises `InvalidSampleError` naming the first sample

### Requirement: Embeddings feed scikit-learn estimators
Every `Embedder` SHALL work as the first step before any scikit-learn estimator that takes a numeric matrix, including classifiers, regressors, clustering and `NearestNeighbors`, and `FeatureUnion` SHALL combine embedders of the same input type.

#### Scenario: Image search
- **WHEN** `NearestNeighbors` is fitted on `Embedder(input='image')` vectors of a gallery and queried with `Embedder()` vectors from the same joint model
- **THEN** it returns the indices of the closest images for every text

#### Scenario: Combined text features
- **WHEN** a `FeatureUnion` of two `Embedder`s with different backends feeds `LogisticRegression`
- **THEN** the classifier is fitted on the concatenated vectors

### Requirement: Pipeline steps run once per fit
Fitting a pipeline SHALL run each generator step once over the training samples, and a generator that is the final step SHALL call its backend during `fit` only to learn from the references it is given.

#### Scenario: One backend call per sample
- **WHEN** a pipeline of `Transcriber` and `Classifier` is fitted on 10 recordings
- **THEN** the transcription backend receives exactly 10 requests

### Requirement: Cached pipeline steps
Components whose backend is given as a string or `None` SHALL be hashable by `joblib`, so `Pipeline(memory=...)` reuses the cached outputs of generator steps when the pipeline is fitted again on the same samples with the same parameters.

#### Scenario: Refit with cache
- **WHEN** a cached pipeline of `TextGenerator(input='image', instructions='Describe the image.')` and `Classifier` is fitted twice on the same images
- **THEN** the description backend receives requests only during the first fit

### Requirement: Tuning across steps
Parameters of every step and of every step's backend SHALL be tunable with `GridSearchCV` using scikit-learn's `<step>__<parameter>` names.

#### Scenario: Tune two steps
- **WHEN** `GridSearchCV` searches `transcriber__language` and `textclassifier__backend__model` over a transcriber and classifier pipeline
- **THEN** every combination is fitted and scored

## REMOVED Requirements

### Requirement: Transcripts feed text components
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.

### Requirement: Descriptions feed text components
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.

### Requirement: Speech translation
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.

### Requirement: Narration of media
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.

### Requirement: Talking presenter
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.

### Requirement: Visual generation chains
**Reason**: The chain is now a recipe built from the general components.
**Migration**: Use the matching function of `skai.recipes`, or the chain of general components it documents.
