# Recipes

`skai.recipes` holds functions that return configured pipelines, from a brief to an explainer video or from a video to
the same video dubbed in another language. Every recipe returns an ordinary scikit-learn `Pipeline` built only from
the public components, the primitives of `skai.pipeline` and scikit-learn, so its steps can be inspected, changed
with `set_params`, cloned and tuned. `video_search` returns a `Retriever` instead, since it answers queries.

Every recipe takes a `backend`, passed to every step that uses one, and keyword arguments for its main choices. The
examples run on `dummy` with tiny inputs. Replace `backend='dummy'` with a provider, or leave it out to use `hf`, to
get real results. Some steps need a provider that offers their operation, as [Backends](backends.md) lists.

```python exec="true" source="above" result="text"
from skai import recipes

pipeline = recipes.explainer_video(backend='dummy')
print([name for name, _ in pipeline.steps])
pipeline.set_params(plan__scenes__instructions='Split the text into two scenes.')
print(pipeline.get_params()['plan__scenes__instructions'])
```

The examples share these inputs: a three-second video with one shot per second and a tone as its track, a recording
of two seconds, and small photos.

```python exec="true" source="above" result="text" session="recipes"
import numpy as np
from PIL import Image
from skai import recipes
from skai.core import Audio, Video

frames = np.zeros((24, 48, 64, 3), dtype=np.uint8)
for index in range(24):
    frames[index] = (index // 8) * 90
tone = Audio((0.3 * np.sin(np.arange(48000) / 10)).astype(np.float32), 16000)
video = Video(frames, fps=8, audio=tone)
recording = (0.3 * np.sin(np.arange(32000) / 10), 16000)
photos = [Image.new('RGB', (64, 48), color) for color in ('red', 'blue')]
print(video, recording[0].shape, photos[0].size)
```

## Brief to explainer video

`explainer_video` turns a brief into one narrated, subtitled video with music. A storyboard from `TextExtractor`
splits the brief into scenes, every scene gets a generated image animated into a clip and a spoken narration, the
clips and narrations are joined, music is mixed under the narration, and the narration is burned in as subtitles.

```python exec="true" source="above" result="text" session="recipes"
explainer = recipes.explainer_video(size=(64, 48), scene_seconds=1, backend='dummy')
(result,) = explainer.fit(None).predict(['How solar panels work'])
print(result, result.audio, result.subtitles is not None)
```

## Articles to podcast

`podcast` turns an article into one audio episode: a summary, a script of two hosts, every line spoken in the voice of
its host, the lines joined in order, and generated music mixed under them.

```python exec="true" source="above" result="text" session="recipes"
podcast = recipes.podcast(voices=('alloy', 'echo'), music_seconds=2, backend='dummy')
print(podcast.fit(None).predict(['Bees pollinate flowers in spring.']))
```

## Post to vertical clip

`vertical_clip` turns a post into a short vertical video: a shortened script, the steps of the explainer video,
reframing to 9:16 following the main subject, and burned subtitles.

```python exec="true" source="above" result="text" session="recipes"
vertical = recipes.vertical_clip(size=(64, 48), scene_seconds=1, backend='dummy')
(clip,) = vertical.fit(None).predict(['A post about running'])
height, width = clip.frames.shape[1:3]
print(clip, round(height / width, 2))
```

## Story to read-along video

`read_along` turns a story into a video whose characters, described by `characters`, look the same on every page.
`fit` generates one reference image of the characters, which every page image keeps consistent, and the narration is
shown word by word.

```python exec="true" source="above" result="text" session="recipes"
story = recipes.read_along(characters='a girl and her dog', size=(64, 48), page_seconds=1, backend='dummy')
print(story.fit(None).predict(['A girl and her dog walk to the sea.']))
```

## Product photos to catalog and ads

`product_catalog` removes the background of every photo and extracts its attributes and a short marketing text.
`product_ad` edits every photo into a scene, upscales it and animates it into a promotional clip.

```python exec="true" source="above" result="text" session="recipes"
records = recipes.product_catalog(backend='dummy').fit(None).predict(photos)
print(list(records[0]))
print(recipes.product_ad(duration=1, backend='dummy').fit(None).predict(photos))
```

## Dubbing in the original voice

`dubbing` turns a video into the same video speaking another language in the original voice. The speech is
transcribed and translated, the vocals and the accompaniment are separated, the translation is spoken in the voice of
the vocals, the video is lip-synced to it, the accompaniment is mixed back, and subtitles in the new language are
burned in.

```python exec="true" source="above" result="text" session="recipes"
(dubbed,) = recipes.dubbing(target_language='Greek', backend='dummy').fit(None).predict([video])
print(dubbed, dubbed.duration == video.duration)
```

## Image text translation

`image_translation` redraws the text of images, such as comics or screenshots, in another language. The text of
every image is transcribed and translated into an edit instruction, the text regions are masked, and the image is
edited inside the mask.

```python exec="true" source="above" result="text" session="recipes"
(translated,) = recipes.image_translation(target_language='Greek', backend='dummy').fit(None).predict(photos[:1])
print(translated.size)
```

## Meeting to briefing

`meeting_briefing` turns a meeting recording into a dict with `minutes`, a list of `action_items` with owners, and
`video`, a short briefing read by a talking head.

```python exec="true" source="above" result="text" session="recipes"
(briefing,) = recipes.meeting_briefing(avatar=photos[0], backend='dummy').fit(None).predict([recording])
print(list(briefing), briefing['action_items'][0], briefing['video'])
```

## Call analytics

`call_topics` labels the topic and sentiment of every speaker turn of recorded calls and clusters the turns by
meaning. It is fitted on recordings, and returns one list of turn records per call.

```python exec="true" source="above" result="text" session="recipes"
calls = [recording, (0.2 * np.cos(np.arange(48000) / 7), 16000)]
turns = recipes.call_topics(n_clusters=2, backend='dummy').fit(calls).predict(calls)
print([sorted(turn) for turn in turns[0]])
```

## Invoices to anomaly scores

`invoice_anomalies` extracts the vendor, amount and date of every invoice scan and scores them with an
`IsolationForest`, which returns `1` for normal and `-1` for anomalous scans.

```python exec="true" source="above" result="text" session="recipes"
scans = [Image.new('RGB', (64, 48), color) for color in ('red', 'blue', 'green', 'white')]
print(recipes.invoice_anomalies(backend='dummy').fit(scans).predict(scans))
```

## Searching a video library by text

`video_search` returns a `Retriever` that splits every video into shots, embeds the shots and the text queries with
one joint text and image model, CLIP on Hugging Face by default, and returns the closest shots of every query with
the index of their video as `sample`, their `start` and `end`, and their `distance`.

```python exec="true" source="above" result="text" session="recipes"
search = recipes.video_search(backend='dummy').fit([video, video])
(shots,) = search.kneighbors(['a goal'], n_neighbors=2)
print([(shot['sample'], shot['start'], shot['end']) for shot in shots])
```

## Multimodal moderation

`moderation` classifies posts given as multimodal samples of an `image` and a `caption`, judging both in one request,
with labeled posts as few-shot examples and probabilities calibrated by cross-validation.

```python exec="true" source="above" result="text" session="recipes"
posts = [{'image': photos[index % 2], 'caption': f'caption {index}'} for index in range(4)]
moderation = recipes.moderation(cv=2, backend='dummy').fit(posts, ['safe', 'unsafe'] * 2)
print(moderation.predict_proba(posts).shape)
```

## Audio description track

`audio_description` adds a spoken description of every shot to a video, placed at the start of the shot, with the
dialogue lowered while it plays, plus subtitles of the dialogue.

```python exec="true" source="above" result="text" session="recipes"
(described,) = recipes.audio_description(max_length=5, backend='dummy').fit(None).predict([video])
print(described, described.duration == video.duration)
```

## Lecture to study pack

`study_pack` turns a lecture video into a dict with `notes`, `key_terms`, a `quiz` and `recap`, a narrated audio
summary.

```python exec="true" source="above" result="text" session="recipes"
(pack,) = recipes.study_pack(n_questions=2, backend='dummy').fit(None).predict([video])
print(list(pack), pack['recap'])
```

## Visual quality control

`quality_control` classifies product photos as defective or not. A visual question answer about defects and the
detected defect regions of every photo are combined into the features of a logistic regression trained on labeled
photos.

```python exec="true" source="above" result="text" session="recipes"
inspection = recipes.quality_control(defects=['scratch', 'dent'], backend='dummy')
print(inspection.fit(photos * 2, ['ok', 'defective'] * 2).predict(photos))
```

## Highlight reel

`highlights` turns a long video into a reel of its `k` most exciting shots in their original order, with generated
music mixed in.

```python exec="true" source="above" result="text" session="recipes"
(reel,) = recipes.highlights(k=2, music_seconds=2, backend='dummy').fit(None).predict([video])
print(reel)
```

The API reference is in [`skai.recipes`](../api/recipes/index.md).
