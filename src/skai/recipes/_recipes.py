"""Return configured pipelines built from the components and primitives."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from datetime import date
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from skai.classification import MultimodalClassifier, TextClassifier, VideoClassifier
from skai.core import Backend, to_object_array
from skai.embedding import TextEmbedder, VideoEmbedder
from skai.extraction import ImageExtractor, TextExtractor
from skai.generation import (
    AudioToAudioGenerator,
    AudioToTextGenerator,
    AudioToVideoGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    MultimodalToAudioGenerator,
    MultimodalToImageGenerator,
    MultimodalToVideoGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
    VideoToAudioGenerator,
    VideoToTextGenerator,
    VideoToVideoGenerator,
)
from skai.pipeline import Field, ForEach, MediaUnion, Retriever, TopK
from skai.presets import INSTRUCTIONS, quiz_writer, storyboarder, summarizer
from skai.segmentation import AudioSegmenter, ImageSegmenter, VideoSegmenter

BackendType = str | Backend | None
NUMBER = re.compile(r'-?\d+(?:[.,]\d+)*')
ISO_DATE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
JOINT_MODEL = 'hf:clip-ViT-B-32'


def _pick_voice(hosts: Any, voices: tuple[str, str]) -> NDArray[np.object_]:
    """Map every host name to the voice of host A or host B."""
    return to_object_array(voices[1] if str(host).strip().upper().endswith('B') else voices[0] for host in hosts)


def _merge_masks(regions: Any) -> NDArray[np.object_]:
    """Merge the masks of the regions of every sample into one mask, or `None` when a sample has no regions."""
    merged = []
    for sample in regions:
        masks = [np.asarray(region['mask'].convert('L')) for region in sample if region.get('mask') is not None]
        merged.append(Image.fromarray(np.maximum.reduce(masks)) if masks else None)
    return to_object_array(merged)


def _to_number(values: Any) -> NDArray[np.float64]:
    """Read the first number of every value, or zero."""
    numbers = []
    for value in np.asarray(values, dtype=object).ravel():
        match = NUMBER.search(str(value))
        numbers.append(float(match.group().replace(',', '')) if match else 0.0)
    return np.asarray(numbers).reshape(-1, 1)


def _to_ordinal(values: Any) -> NDArray[np.float64]:
    """Read the ISO date of every value as a day number, or zero."""
    days = []
    for value in np.asarray(values, dtype=object).ravel():
        match = ISO_DATE.search(str(value))
        try:
            days.append(float(date(*(int(part) for part in match.groups())).toordinal()) if match else 0.0)
        except ValueError:
            days.append(0.0)
    return np.asarray(days).reshape(-1, 1)


def _region_features(regions: Any) -> NDArray[np.float64]:
    """Return the number of regions and their highest score for every sample."""
    rows = [[len(sample), max((region['score'] for region in sample), default=0.0)] for sample in regions]
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def _narrated_video(
    backend: BackendType,
    *,
    size: tuple[int, int],
    scene_seconds: float,
    voice: str | None,
    music: bool,
    aspect_ratio: float | None = None,
    reference_prompt: str | None = None,
    subtitles_by: Literal['speech', 'word'] = 'speech',
) -> list[tuple[str, Any]]:
    """Return the steps that turn a dict with `scenes` and `music` into a narrated, subtitled video."""
    image = TextToImageGenerator(size=size, reference_prompt=reference_prompt, backend=backend)
    clip = ImageToVideoGenerator(duration=scene_seconds, backend=backend)
    video = [Field('scenes'), ForEach(make_pipeline(Field('visual'), image, clip))]
    video.append(VideoToVideoGenerator(operation='concatenate'))
    if aspect_ratio is not None:
        video.append(VideoToVideoGenerator(operation='reframe', aspect_ratio=aspect_ratio, backend=backend))
    speech = TextToAudioGenerator(voice=voice, backend=backend)
    narration = [Field('scenes'), ForEach(make_pipeline(Field('narration'), speech))]
    narration.append(AudioToAudioGenerator(operation='concatenate', gap=0.3))
    media = [('video', make_pipeline(*video)), ('narration', make_pipeline(*narration))]
    tracks = ['narration']
    if music:
        music_audio = TextToAudioGenerator(operation='music', duration=scene_seconds, backend=backend)
        media.append(('music', make_pipeline(Field('music'), music_audio)))
        tracks.append('music')
    mixer = MultimodalToAudioGenerator(duck_under=['music'], duck_for='narration', volumes={'music': 0.5})
    return [
        ('media', MediaUnion(media)),
        (
            'tracks',
            MediaUnion(
                [
                    ('video', Field('video')),
                    ('audio', make_pipeline(Field(tracks), mixer)),
                    ('transcript', make_pipeline(Field('narration'), AudioSegmenter(by=subtitles_by, backend=backend))),
                ],
            ),
        ),
        (
            'muxed',
            MediaUnion(
                [
                    ('video', make_pipeline(Field(['video', 'audio']), MultimodalToVideoGenerator(operation='mux'))),
                    ('transcript', Field('transcript')),
                ],
            ),
        ),
        ('subtitled', MultimodalToVideoGenerator(operation='subtitle')),
    ]


def _plan(backend: BackendType, n_scenes: int | str, music: bool) -> MediaUnion:
    """Return the union that turns a text into a storyboard and a music description."""
    branches: list[tuple[str, Any]] = [('scenes', storyboarder(n_scenes=n_scenes, backend=backend))]
    if music:
        instructions = 'Describe background music that fits a video about the text, in one sentence.'
        branches.append(('music', TextToTextGenerator(instructions=instructions, backend=backend)))
    return MediaUnion(branches)


def explainer_video(
    *,
    n_scenes: int | str = 'a few',
    size: tuple[int, int] = (1280, 720),
    scene_seconds: float = 4.0,
    voice: str | None = None,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that turns a brief into one narrated, subtitled video with music.

    A storyboard splits the brief into scenes, every scene gets a generated image animated into a clip and a spoken
    narration, the clips and narrations are joined, music is mixed under the narration, and the narration is burned
    in as subtitles.

    Args:
        n_scenes:
            The number of scenes, or words such as `'a few'`.

        size:
            The width and height of the video.

        scene_seconds:
            The length of every scene clip in seconds.

        voice:
            The voice of the narration.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one video per brief.
    """
    steps = [('plan', _plan(backend, n_scenes, music=True))]
    return Pipeline(steps + _narrated_video(backend, size=size, scene_seconds=scene_seconds, voice=voice, music=True))


def vertical_clip(
    *,
    max_length: int = 60,
    size: tuple[int, int] = (1280, 720),
    scene_seconds: float = 3.0,
    voice: str | None = None,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that turns a post into a short vertical video with burned subtitles.

    The post is shortened, turned into an explainer video, and reframed to 9:16 following its main subject.

    Args:
        max_length:
            The maximum number of words of the shortened post.

        size:
            The width and height of the generated scenes, before reframing.

        scene_seconds:
            The length of every scene clip in seconds.

        voice:
            The voice of the narration.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one vertical video per post.
    """
    instructions = 'Rewrite the post as a short, punchy script for a vertical video.'
    steps: list[tuple[str, Any]] = [
        ('shorten', TextToTextGenerator(instructions=instructions, max_length=max_length, backend=backend)),
        ('plan', _plan(backend, 'three', music=True)),
    ]
    video = _narrated_video(
        backend,
        size=size,
        scene_seconds=scene_seconds,
        voice=voice,
        music=True,
        aspect_ratio=9 / 16,
    )
    return Pipeline(steps + video)


def read_along(
    *,
    characters: str = 'the main characters of the story, in a friendly picture-book style',
    size: tuple[int, int] = (1024, 768),
    page_seconds: float = 6.0,
    voice: str | None = None,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that turns a story into a read-along video with consistent characters and word subtitles.

    `fit` generates one reference image of the characters, which every page image keeps consistent, and the narration
    is shown word by word.

    Args:
        characters:
            A description of the characters, from which the reference image is generated.

        size:
            The width and height of the pages.

        page_seconds:
            The length of every page in seconds.

        voice:
            The voice of the narration.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one video per story.
    """
    steps = [('plan', _plan(backend, 'one per page', music=False))]
    pages = _narrated_video(
        backend,
        size=size,
        scene_seconds=page_seconds,
        voice=voice,
        music=False,
        reference_prompt=characters,
        subtitles_by='word',
    )
    return Pipeline(steps + pages)


def podcast(
    *,
    voices: tuple[str, str] = ('alloy', 'echo'),
    music_seconds: float = 8.0,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that turns an article into a two-host podcast episode with intro music.

    The article is summarized and written as a conversation of host A and host B, every line is spoken in the voice
    of its host, the lines are joined in order, and generated music is mixed under them.

    Args:
        voices:
            The voices of host A and host B.

        music_seconds:
            The length of the music in seconds.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one audio episode per article.
    """
    script = TextExtractor(
        fields={'lines': [{'host': 'A or B', 'text': 'what the host says'}]},
        instructions='Write a lively podcast conversation between host A and host B that covers the text.',
        backend=backend,
    )
    voice = make_pipeline(Field('host'), FunctionTransformer(_pick_voice, kw_args={'voices': voices}))
    line = make_pipeline(
        MediaUnion([('text', Field('text')), ('voice', voice)]),
        MultimodalToAudioGenerator(operation='speech', backend=backend),
    )
    speech = make_pipeline(script, ForEach(line), AudioToAudioGenerator(operation='concatenate', gap=0.3))
    music_prompt = TextToTextGenerator(
        instructions='Describe short intro music for a podcast about the text, in one sentence.',
        backend=backend,
    )
    music = make_pipeline(
        music_prompt,
        TextToAudioGenerator(operation='music', duration=music_seconds, backend=backend),
    )
    return Pipeline(
        [
            ('summary', summarizer(backend=backend)),
            ('parts', MediaUnion([('speech', speech), ('music', music)])),
            ('mix', MultimodalToAudioGenerator(duck_under=['music'], duck_for='speech', volumes={'music': 0.4})),
        ],
    )


def product_catalog(*, attributes: dict[str, str] | None = None, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that turns product photos into catalog records with marketing copy.

    The background of every photo is removed, and the attributes and a short marketing text are extracted from it.

    Args:
        attributes:
            The attribute fields, as names and descriptions. Name, category, color and material are used when it is
            `None`.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one record per photo.
    """
    fields: dict[str, Any] = attributes or {
        'name': 'product name',
        'category': 'product category',
        'color': 'main color',
        'material': 'main material',
    }
    fields = {**fields, 'copy': 'two sentences of marketing copy for the product'}
    return Pipeline(
        [
            ('cutout', ImageToImageGenerator(operation='remove_background', backend=backend)),
            ('record', ImageExtractor(fields=fields, backend=backend)),
        ],
    )


def product_ad(
    *,
    scene: str = 'Place the product in an attractive lifestyle scene with soft natural light.',
    motion: str = 'a slow camera orbit around the product',
    duration: float = 4.0,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that turns product photos into short promotional clips.

    Every photo is edited into a scene, upscaled and animated.

    Args:
        scene:
            The edit that places the product in a scene.

        motion:
            The motion of the clip.

        duration:
            The length of every clip in seconds.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one video per photo.
    """
    return Pipeline(
        [
            ('scene', ImageToImageGenerator(operation='edit', instructions=scene, backend=backend)),
            ('upscale', ImageToImageGenerator(operation='upscale', backend=backend)),
            ('animate', ImageToVideoGenerator(instructions=motion, duration=duration, backend=backend)),
        ],
    )


def dubbing(*, target_language: str, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that dubs a video into another language in the original voice.

    The speech is transcribed and translated, the vocals and the accompaniment are separated, the translation is spoken
    in the voice of the vocals, the video is lip-synced to it, the accompaniment is mixed back, and subtitles in the
    new language are burned in.

    Args:
        target_language:
            The language of the dubbed video.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one video per video, as long as the input.
    """
    translation = INSTRUCTIONS['translator'].format(target_language=target_language)
    translate = TextToTextGenerator(instructions=translation, backend=backend)
    sources = MediaUnion(
        [
            ('video', 'passthrough'),
            ('text', make_pipeline(VideoToTextGenerator(operation='transcribe', backend=backend), translate)),
            (
                'voice',
                make_pipeline(
                    VideoToAudioGenerator(),
                    AudioToAudioGenerator(operation='separate', stem='vocals', backend=backend),
                ),
            ),
            (
                'accompaniment',
                make_pipeline(
                    VideoToAudioGenerator(),
                    AudioToAudioGenerator(operation='separate', stem='accompaniment', backend=backend),
                ),
            ),
        ],
    )
    speech = MediaUnion(
        [
            ('video', Field('video')),
            (
                'audio',
                make_pipeline(
                    Field(['text', 'voice']),
                    MultimodalToAudioGenerator(operation='speech', backend=backend),
                ),
            ),
            ('accompaniment', Field('accompaniment')),
        ],
    )
    fitted = MediaUnion(
        [
            ('speaking', make_pipeline(Field(['video', 'audio']), MultimodalToVideoGenerator(operation='mux'))),
            ('accompaniment', Field('accompaniment')),
            ('transcript', make_pipeline(Field('audio'), AudioSegmenter(by='speech', backend=backend))),
        ],
    )
    synced = MediaUnion(
        [
            ('video', Field('speaking')),
            (
                'mix',
                make_pipeline(Field(['speaking', 'accompaniment']), MultimodalToAudioGenerator(duck_for='speaking')),
            ),
            ('transcript', Field('transcript')),
        ],
    )
    lipsynced = MediaUnion(
        [
            (
                'video',
                make_pipeline(
                    MediaUnion([('video', Field('video')), ('audio', Field('video'))]),
                    MultimodalToVideoGenerator(operation='lip_sync', backend=backend),
                ),
            ),
            ('audio', Field('mix')),
            ('transcript', Field('transcript')),
        ],
    )
    muxed = MediaUnion(
        [
            ('video', make_pipeline(Field(['video', 'audio']), MultimodalToVideoGenerator(operation='mux'))),
            ('transcript', Field('transcript')),
        ],
    )
    return Pipeline(
        [
            ('sources', sources),
            ('speech', speech),
            ('fitted', fitted),
            ('synced', synced),
            ('lipsynced', lipsynced),
            ('muxed', muxed),
            ('subtitled', MultimodalToVideoGenerator(operation='subtitle')),
        ],
    )


def image_translation(*, target_language: str, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that redraws the text of images, such as comics or screenshots, in another language.

    The text of every image is transcribed and translated into an edit instruction, the text regions are masked, and
    the image is edited inside the mask.

    Args:
        target_language:
            The language of the redrawn text.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one image per image, of the input size.
    """
    instructions = (
        f'Translate the text into {target_language}. Answer with an image edit instruction of the form: Replace the '
        'text with "<translation>", keeping the font, size, color and layout.'
    )
    parts = MediaUnion(
        [
            ('image', 'passthrough'),
            (
                'instructions',
                make_pipeline(
                    ImageToTextGenerator(operation='transcribe', backend=backend),
                    TextToTextGenerator(instructions=instructions, backend=backend),
                ),
            ),
            (
                'mask',
                make_pipeline(
                    ImageSegmenter(by='mask', prompt='written text', backend=backend),
                    FunctionTransformer(_merge_masks),
                ),
            ),
        ],
    )
    return Pipeline([('parts', parts), ('redraw', MultimodalToImageGenerator(backend=backend))])


def meeting_briefing(*, avatar: Any, max_length: int = 80, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that turns a meeting recording into minutes, action items and a talking-head briefing.

    Args:
        avatar:
            The portrait that reads the briefing.

        max_length:
            The maximum number of words of the briefing.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one dict per recording with `minutes`, `action_items` and `video`.
    """
    minutes = TextToTextGenerator(
        instructions='Write the minutes of the meeting: decisions, open questions and key points.',
        backend=backend,
    )
    action_items = TextExtractor(
        fields={'action_items': [{'owner': 'who does it', 'task': 'what to do', 'due': 'when, if stated'}]},
        backend=backend,
    )
    briefing = TextToTextGenerator(
        instructions='Write a short spoken briefing of the meeting for people who missed it.',
        max_length=max_length,
        backend=backend,
    )
    video = make_pipeline(
        briefing,
        TextToAudioGenerator(backend=backend),
        AudioToVideoGenerator(avatar=avatar, backend=backend),
    )
    return Pipeline(
        [
            ('transcript', AudioToTextGenerator(backend=backend)),
            ('briefing', MediaUnion([('minutes', minutes), ('action_items', action_items), ('video', video)])),
        ],
    )


def call_topics(
    *,
    topics: list[str] | None = None,
    n_clusters: int = 5,
    random_state: int | None = 0,
    backend: BackendType = None,
) -> Pipeline:
    """Return a pipeline that labels the topic and sentiment of every speaker turn of calls and clusters the turns.

    Args:
        topics:
            The topic labels. Billing, technical issue, account, cancellation and other are used when it is `None`.

        n_clusters:
            The number of clusters of the turns by meaning.

        random_state:
            Controls the clustering.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, which is fitted on recordings and whose `predict` returns one list of turn records per call,
        each with `speaker`, `text`, `topic`, `sentiment` and `cluster`.
    """
    topics = topics or ['billing', 'technical issue', 'account', 'cancellation', 'other']
    cluster = make_pipeline(
        Field('text'),
        TextEmbedder(backend=backend),
        KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state),
        FunctionTransformer(np.argmin, kw_args={'axis': 1}),
    )
    turn = MediaUnion(
        [
            ('speaker', Field('speaker')),
            ('text', Field('text')),
            ('topic', make_pipeline(Field('text'), TextClassifier(labels=topics, backend=backend))),
            (
                'sentiment',
                make_pipeline(
                    Field('text'),
                    TextClassifier(labels=['positive', 'neutral', 'negative'], backend=backend),
                ),
            ),
            ('cluster', cluster),
        ],
    )
    return Pipeline([('turns', AudioSegmenter(by='speaker', backend=backend)), ('label', ForEach(turn))])


def invoice_anomalies(*, contamination: float | str = 'auto', backend: BackendType = None) -> Pipeline:
    """Return a pipeline that scores invoice scans as normal or anomalous from their vendor, amount and date.

    Args:
        contamination:
            The expected share of anomalies, passed to `IsolationForest`.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, which is fitted on scans and whose `predict` returns `1` for normal and `-1` for anomalous scans.
    """
    fields = {'vendor': 'vendor name', 'amount': 'total amount', 'date': 'invoice date as YYYY-MM-DD'}
    features = ColumnTransformer(
        [
            ('vendor', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [0]),
            ('amount', FunctionTransformer(_to_number), [1]),
            ('date', FunctionTransformer(_to_ordinal), [2]),
        ],
    )
    return Pipeline(
        [
            ('fields', ImageExtractor(fields=fields, backend=backend)),
            ('features', features),
            ('score', IsolationForest(contamination=contamination, random_state=0)),
        ],
    )


def video_search(*, threshold: float = 0.5, n_frames: int = 4, backend: BackendType = None) -> Retriever:
    """Return a retriever that indexes the shots of a video library and finds the closest shots of text queries.

    Shots and texts are embedded with one joint text and image model, CLIP on Hugging Face by default.

    Args:
        threshold:
            The histogram distance between consecutive frames above which a cut is found.

        n_frames:
            The number of frames embedded per shot.

        backend:
            The backend of the embedders. It must embed texts and images in one space.

    Returns:
        The retriever, whose `kneighbors` returns shot records with `sample`, `start`, `end` and `distance`.
    """
    backend = JOINT_MODEL if backend is None else backend
    shot = MediaUnion(
        [
            ('start', Field('start')),
            ('end', Field('end')),
            ('vector', make_pipeline(Field('media'), VideoEmbedder(n_frames=n_frames, center=False, backend=backend))),
        ],
    )
    items = make_pipeline(VideoSegmenter(by='shot', threshold=threshold), ForEach(shot))
    return Retriever(items, TextEmbedder(center=False, backend=backend))


def moderation(*, labels: list[str] | None = None, cv: int = 3, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that classifies posts of an image and a caption, calibrated on labeled posts.

    Every post is classified with its image and caption in one request, with labeled posts as few-shot examples, and
    the scores are calibrated with cross-validation into probabilities.

    Args:
        labels:
            The labels, `['safe', 'unsafe']` when it is `None`.

        cv:
            The number of cross-validation folds of the calibration.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, which is fitted on posts, as dicts with `image` and `caption`, and their labels.
    """
    labels = labels or ['safe', 'unsafe']
    instructions = INSTRUCTIONS['moderator'].format(kind='post, judging its image and caption together,')
    classifier = MultimodalClassifier(labels=labels, instructions=instructions, backend=backend)
    return Pipeline([('moderate', CalibratedClassifierCV(classifier, cv=cv))])


def audio_description(*, max_length: int = 15, voice: str | None = None, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that adds a spoken description of every shot and subtitles of the dialogue to a video.

    Every description starts with its shot, and the dialogue is lowered while it plays.

    Args:
        max_length:
            The maximum number of words of every description.

        voice:
            The voice of the descriptions.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one video per video, as long as the input.
    """
    describe = VideoToTextGenerator(
        instructions='Describe what is shown in one short sentence for a blind viewer.',
        max_length=max_length,
        backend=backend,
    )
    shot = MediaUnion(
        [
            ('start', Field('start')),
            ('media', make_pipeline(Field('media'), describe, TextToAudioGenerator(voice=voice, backend=backend))),
        ],
    )
    sources = MediaUnion(
        [
            ('video', 'passthrough'),
            ('dialogue', VideoToAudioGenerator()),
            ('description', make_pipeline(VideoSegmenter(by='shot'), ForEach(shot))),
            ('transcript', make_pipeline(VideoToAudioGenerator(), AudioSegmenter(by='speech', backend=backend))),
        ],
    )
    mixer = MultimodalToAudioGenerator(duck_under=['dialogue'], duck_for='description')
    tracks = MediaUnion(
        [
            ('video', Field('video')),
            ('audio', make_pipeline(Field(['dialogue', 'description']), mixer)),
            ('transcript', Field('transcript')),
        ],
    )
    muxed = MediaUnion(
        [
            ('video', make_pipeline(Field(['video', 'audio']), MultimodalToVideoGenerator(operation='mux'))),
            ('transcript', Field('transcript')),
        ],
    )
    return Pipeline(
        [
            ('sources', sources),
            ('tracks', tracks),
            ('muxed', muxed),
            ('subtitled', MultimodalToVideoGenerator(operation='subtitle')),
        ],
    )


def study_pack(*, n_questions: int = 5, voice: str | None = None, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that turns a lecture video into notes, key terms, a quiz and a narrated recap.

    Args:
        n_questions:
            The number of quiz questions.

        voice:
            The voice of the recap.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one dict per lecture with `notes`, `key_terms`, `quiz` and `recap`.
    """
    notes = TextToTextGenerator(
        instructions='Write structured study notes of the lecture, with headings and bullet points.',
        backend=backend,
    )
    key_terms = TextExtractor(
        fields={'key_terms': [{'term': 'a key term', 'definition': 'its meaning'}]},
        backend=backend,
    )
    recap = make_pipeline(
        summarizer(max_length=60, backend=backend),
        TextToAudioGenerator(voice=voice, backend=backend),
    )
    return Pipeline(
        [
            ('transcript', VideoToTextGenerator(operation='transcribe', backend=backend)),
            (
                'pack',
                MediaUnion(
                    [
                        ('notes', notes),
                        ('key_terms', key_terms),
                        ('quiz', quiz_writer(n_questions=n_questions, backend=backend)),
                        ('recap', recap),
                    ],
                ),
            ),
        ],
    )


def quality_control(*, defects: list[str] | None = None, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that classifies product photos as defective or not, trained on labeled photos.

    A visual question answer about defects and the detected defect regions of every photo are combined into features
    of a logistic regression.

    Args:
        defects:
            The kinds of defects to detect. Scratches, dents, cracks and stains are used when it is `None`.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, which is fitted on photos and their labels.
    """
    defects = defects or ['scratch', 'dent', 'crack', 'stain']
    question = ImageToTextGenerator(
        instructions=(
            f'Does the product show a defect such as a {", ".join(defects)}? Answer yes or no, then describe it.'
        ),
        backend=backend,
    )
    features = FeatureUnion(
        [
            ('answer', make_pipeline(question, TextEmbedder(backend=backend))),
            (
                'regions',
                make_pipeline(
                    ImageSegmenter(labels=defects, backend=backend),
                    FunctionTransformer(_region_features),
                ),
            ),
        ],
    )
    return Pipeline([('features', features), ('classify', LogisticRegression(max_iter=1000))])


def highlights(*, k: int = 5, music_seconds: float = 30.0, backend: BackendType = None) -> Pipeline:
    """Return a pipeline that turns a long video into a reel of its most exciting shots with music.

    Args:
        k:
            The number of shots of the reel, kept in their original order.

        music_seconds:
            The length of the music in seconds.

        backend:
            The backend of every step that uses one.

    Returns:
        The pipeline, whose `predict` returns one reel per video.
    """
    reel = make_pipeline(
        VideoSegmenter(by='shot'),
        ForEach(Field('media')),
        TopK(VideoClassifier(labels=['exciting', 'calm'], backend=backend), label='exciting', k=k),
        VideoToVideoGenerator(operation='concatenate'),
    )
    music = make_pipeline(
        VideoToTextGenerator(
            instructions='Describe energetic music that fits the video, in one sentence.',
            backend=backend,
        ),
        TextToAudioGenerator(operation='music', duration=music_seconds, backend=backend),
    )
    mixer = MultimodalToAudioGenerator(duck_under=['music'], duck_for='video', volumes={'music': 0.6})
    tracks = MediaUnion([('video', Field('video')), ('audio', make_pipeline(Field(['video', 'music']), mixer))])
    return Pipeline(
        [
            ('parts', MediaUnion([('video', reel), ('music', music)])),
            ('tracks', tracks),
            ('muxed', MultimodalToVideoGenerator(operation='mux')),
        ],
    )
