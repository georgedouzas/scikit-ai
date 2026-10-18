"""Test the recipes of `skai.recipes` on the dummy backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from skai import recipes
from skai.backends import DummyBackend
from skai.core import Audio, ImageGenerationRequest, SpeechRequest, Video
from skai.pipeline import Retriever

FPS = 8
RATE = 16000


class RecordingBackend(DummyBackend):
    """Record the image and speech requests of every clone."""

    images: ClassVar[list[ImageGenerationRequest]] = []
    speech: ClassVar[list[SpeechRequest]] = []

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Record the request and generate the dummy image."""
        RecordingBackend.images.append(request)
        return super().generate_image(request)

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Record the request and generate the dummy speech."""
        RecordingBackend.speech.append(request)
        return super().synthesize_speech(request)


@pytest.fixture
def recording() -> type[RecordingBackend]:
    """Return the recording backend class with empty records."""
    RecordingBackend.images.clear()
    RecordingBackend.speech.clear()
    return RecordingBackend


def _video(seconds: int = 3, audio: bool = True) -> Video:
    """Return a video with one shot per second and a tone as its track."""
    frames = np.zeros((seconds * FPS, 48, 64, 3), dtype=np.uint8)
    for index in range(frames.shape[0]):
        frames[index] = (index // FPS) * 90 % 256
    track = Audio((0.3 * np.sin(np.arange(seconds * RATE) / 10)).astype(np.float32), RATE) if audio else None
    return Video(frames, FPS, track)


def _photo(color: str = 'red') -> Image.Image:
    """Return a small product photo."""
    return Image.new('RGB', (64, 48), color)


def _recording() -> tuple[np.ndarray, int]:
    """Return two seconds of audio."""
    return 0.3 * np.sin(np.arange(2 * RATE) / 10), RATE


def test_recipe_returns_pipeline() -> None:
    """A recipe returns a pipeline whose steps can be changed."""
    pipeline = recipes.explainer_video(backend='dummy')
    assert isinstance(pipeline, Pipeline)
    pipeline.set_params(plan__scenes__instructions='One scene only.')
    assert pipeline.get_params()['plan__scenes__instructions'] == 'One scene only.'
    clone(pipeline)


def test_explainer_video() -> None:
    """A brief becomes one video with an audio track and subtitles."""
    (video,) = recipes.explainer_video(size=(64, 48), backend='dummy').fit(None).predict(['How solar panels work'])
    assert isinstance(video, Video)
    assert video.audio is not None
    assert video.subtitles


def test_podcast() -> None:
    """An article becomes one audio episode."""
    episodes = recipes.podcast(backend='dummy').fit(None).predict(['Bees pollinate flowers in spring.'])
    assert len(episodes) == 1
    assert isinstance(episodes[0], Audio)


def test_podcast_voices_per_host(recording: type[RecordingBackend]) -> None:
    """Every line is spoken in one of the two host voices."""
    recipes.podcast(voices=('ana', 'bob'), backend=recording()).fit(None).predict(['Bees pollinate flowers.'])
    assert {request.voice for request in recording.speech} <= {'ana', 'bob'}
    assert recording.speech


def test_vertical_clip() -> None:
    """A post becomes a video whose height is 16/9 of its width."""
    (video,) = recipes.vertical_clip(size=(64, 48), backend='dummy').fit(None).predict(['A post about running'])
    height, width = video.frames.shape[1:3]
    assert height / width == pytest.approx(16 / 9, rel=0.05)
    assert video.subtitles


def test_read_along(recording: type[RecordingBackend]) -> None:
    """A story becomes one video, and every page image carries the same reference image."""
    (video,) = recipes.read_along(size=(64, 48), backend=recording()).fit(None).predict(['A girl and her dog.'])
    assert isinstance(video, Video)
    pages = [request for request in recording.images if request.reference_images]
    assert len(pages) >= 2
    first = pages[0].reference_images
    assert all(request.reference_images == first for request in pages)


def test_product_catalog_and_ad() -> None:
    """Product photos become records with the attribute fields and promotional clips."""
    photos = [_photo('red'), _photo('blue')]
    records = recipes.product_catalog(backend='dummy').fit(None).predict(photos)
    assert len(records) == 2
    assert set(records[0]) == {'name', 'category', 'color', 'material', 'copy'}
    clips = recipes.product_ad(backend='dummy').fit(None).predict(photos)
    assert all(isinstance(clip, Video) for clip in clips)
    assert len(clips) == 2


def test_dubbing(recording: type[RecordingBackend]) -> None:
    """A video is dubbed as long as the input, with the separated vocals as the voice."""
    source = _video()
    (video,) = recipes.dubbing(target_language='Greek', backend=recording()).fit(None).predict([source])
    assert video.duration == pytest.approx(source.duration)
    assert video.subtitles
    voices = [request.voice for request in recording.speech]
    assert voices
    assert all(isinstance(voice, Audio) for voice in voices)
    assert voices[0].duration == pytest.approx(source.duration, abs=0.1)


def test_image_translation() -> None:
    """An image is redrawn at its size."""
    (image,) = recipes.image_translation(target_language='Greek', backend='dummy').fit(None).predict([_photo()])
    assert image.size == (64, 48)


def test_meeting_briefing() -> None:
    """A recording becomes minutes, action items and a talking-head video."""
    (briefing,) = recipes.meeting_briefing(avatar=_photo(), backend='dummy').fit(None).predict([_recording()])
    assert set(briefing) == {'minutes', 'action_items', 'video'}
    assert isinstance(briefing['action_items'], list)
    assert isinstance(briefing['video'], Video)


def test_call_topics() -> None:
    """Every call becomes turn records with speaker, topic, sentiment and cluster."""
    calls = [_recording(), (0.2 * np.cos(np.arange(3 * RATE) / 7), RATE)]
    turns = recipes.call_topics(n_clusters=3, backend='dummy').fit(calls).predict(calls)
    assert len(turns) == 2
    for call in turns:
        for turn in call:
            assert {'speaker', 'topic', 'sentiment', 'cluster'} <= set(turn)


def test_invoice_anomalies() -> None:
    """Every scan gets one anomaly label."""
    scans = [_photo(color) for color in ('red', 'blue', 'green', 'white')]
    labels = recipes.invoice_anomalies(backend='dummy').fit(scans).predict(scans)
    assert len(labels) == 4
    assert set(labels.tolist()) <= {-1, 1}


def test_video_search() -> None:
    """Text queries return the closest shots with their video index and times."""
    retriever = recipes.video_search(backend='dummy').fit([_video(), _video()])
    assert isinstance(retriever, Retriever)
    (shots,) = retriever.kneighbors(['a goal'], n_neighbors=2)
    assert len(shots) == 2
    assert all({'sample', 'start', 'end'} <= set(shot) for shot in shots)


def test_moderation() -> None:
    """Posts of an image and a caption get one probability row each."""
    posts: list[dict[str, Any]] = [{'image': _photo(), 'caption': f'caption {index}'} for index in range(9)]
    labels = ['safe', 'unsafe', 'safe'] * 3
    proba = recipes.moderation(backend='dummy').fit(posts, labels).predict_proba(posts)
    assert proba.shape == (9, 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_audio_description() -> None:
    """A video gets descriptions and subtitles and keeps its length."""
    source = _video()
    (video,) = recipes.audio_description(backend='dummy').fit(None).predict([source])
    assert video.duration == pytest.approx(source.duration)
    assert video.subtitles


def test_study_pack() -> None:
    """A lecture becomes notes, key terms, a quiz and a recap."""
    (pack,) = recipes.study_pack(backend='dummy').fit(None).predict([_video()])
    assert set(pack) == {'notes', 'key_terms', 'quiz', 'recap'}
    assert isinstance(pack['recap'], Audio)


def test_quality_control() -> None:
    """Labeled photos train a classifier that labels every photo."""
    photos = [_photo('red'), _photo('blue')] * 2
    labels = ['ok', 'defective'] * 2
    predictions = recipes.quality_control(backend='dummy').fit(photos, labels).predict(photos)
    assert len(predictions) == 4
    assert set(predictions.tolist()) <= {'ok', 'defective'}


def test_highlights() -> None:
    """A long video becomes a reel of at most three shots."""
    (reel,) = recipes.highlights(k=3, backend='dummy').fit(None).predict([_video(5)])
    assert isinstance(reel, Video)
    assert reel.duration <= 3 + 1e-9
