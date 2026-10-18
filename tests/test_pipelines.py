"""Test how components combine in scikit-learn pipelines.

The recipes of the pipelines spec are tested in `tests/test_recipes.py`.
"""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

from skai.backends import DummyBackend
from skai.classification import ImageClassifier, TextClassifier
from skai.core import Audio, InvalidSampleError, Video
from skai.embedding import ImageEmbedder, TextEmbedder
from skai.generation import (
    AudioToTextGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
    TextToVideoGenerator,
    VideoToTextGenerator,
)

IMAGES = [Image.new('RGB', (8, 8), color) for color in ('red', 'green', 'blue')]


class CountingBackend(DummyBackend):
    """Count the requests of every task."""

    counts: ClassVar[dict[str, int]] = {}

    def run(self, task, requests):
        """Count the requests and run them like the dummy backend."""
        CountingBackend.counts[task] = CountingBackend.counts.get(task, 0) + len(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def reset_counts():
    """Start every test with no counted requests."""
    CountingBackend.counts = {}


def test_image_search_by_text():
    """Return the indices of the closest images for every text with one joint model."""
    gallery = ImageEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(IMAGES)
    queries = TextEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(['red', 'blue'])
    indices = NearestNeighbors(n_neighbors=1).fit(gallery).kneighbors(queries, return_distance=False)
    assert indices.shape == (2, 1)
    assert set(indices.ravel()) <= {0, 1, 2}


def test_combined_text_features():
    """Fit a classifier on the concatenated vectors of two text embedders."""
    union = FeatureUnion([('a', TextEmbedder(backend='dummy:a')), ('b', TextEmbedder(dimensions=4, backend='dummy:b'))])
    model = make_pipeline(union, LogisticRegression()).fit(['x', 'y', 'z', 'w'], [0, 1, 0, 1])
    assert model[0].transform(['x']).shape == (1, 12)
    assert model[-1].coef_.shape == (1, 12)
    assert model.predict(['x', 'y']).shape == (2,)


def test_embeddings_feed_clustering():
    """Cluster the embeddings of texts."""
    labels = make_pipeline(TextEmbedder(backend='dummy'), KMeans(n_clusters=2, n_init=1, random_state=0))
    assert labels.fit_predict(['a', 'b', 'c', 'd']).shape == (4,)


def test_paths_between_steps(tmp_path):
    """Transcribe the files that the speech generator wrote."""
    pipeline = make_pipeline(
        TextToAudioGenerator(output_format='path', output_dir=tmp_path, backend='dummy'),
        AudioToTextGenerator(backend='dummy'),
    )
    transcripts = pipeline.fit(None).predict(['one', 'two words'])
    assert len(transcripts) == 2
    assert all(transcript.startswith('Speech of') for transcript in transcripts)
    assert len(list(tmp_path.glob('*.wav'))) == 2


@pytest.mark.parametrize('output_format', ['object', 'bytes', 'path'])
def test_every_output_format_feeds_the_next_step(tmp_path, output_format):
    """Accept the images, audio and videos of a generator in every output format."""
    options = {'output_format': output_format, 'output_dir': tmp_path, 'backend': 'dummy'}
    images = make_pipeline(
        TextToImageGenerator(size=(8, 8), **options),
        ImageToImageGenerator(operation='upscale', **options),
        ImageToTextGenerator(instructions='Describe.', backend='dummy'),
    )
    assert images.fit(None).predict(['a cat']).tolist() == ['A 16x16 image.']
    speech = make_pipeline(TextToAudioGenerator(**options), AudioToTextGenerator(backend='dummy'))
    assert speech.fit(None).predict(['hello there'])[0].startswith('Speech of')
    videos = make_pipeline(
        TextToVideoGenerator(duration=1, size=(16, 16), fps=4, **options),
        VideoToTextGenerator(instructions='Describe.', n_frames=2, backend='dummy'),
    )
    assert videos.fit(None).predict(['a boat'])[0].startswith('2 frames')


def test_generated_media_chains():
    """Generate, edit and animate an image from one prompt."""
    pipeline = make_pipeline(
        TextToImageGenerator(size=(16, 16), backend='dummy'),
        ImageToImageGenerator(instructions='make it night', backend='dummy'),
        ImageToVideoGenerator(duration=1, backend='dummy'),
    )
    videos = pipeline.fit(None).predict(['a quiet harbor'])
    assert len(videos) == 1
    assert isinstance(videos[0], Video)


def test_mismatched_types():
    """Name the first sample when an image reaches a text component."""
    pipeline = make_pipeline(
        TextToImageGenerator(size=(4, 4), backend='dummy'),
        TextToTextGenerator(instructions='Summarize the text.', backend='dummy'),
    )
    with pytest.raises(InvalidSampleError, match='Sample 0'):
        pipeline.fit(None).predict(['a cat'])


def test_one_backend_call_per_sample(audio):
    """Send exactly one transcription request per training recording."""
    pipeline = make_pipeline(AudioToTextGenerator(backend=CountingBackend()), TextClassifier(backend='dummy'))
    pipeline.fit([audio] * 10, ['a', 'b'] * 5)
    assert CountingBackend.counts == {'transcribe': 10}


def test_final_generator_calls_the_backend_only_to_learn(audio):
    """Call the backend of a final generator during `fit` only for what it learns from references."""
    pipeline = make_pipeline(
        AudioToTextGenerator(backend='dummy'),
        TextToTextGenerator(instructions='Summarize.', backend=CountingBackend()),
    )
    pipeline.fit([audio] * 4, ['A summary.'] * 4)
    assert CountingBackend.counts == {}
    pipeline.predict([audio])
    assert CountingBackend.counts == {'generate_text': 1}


def test_refit_with_cache(tmp_path):
    """Describe the images only during the first of two identical fits of a cached pipeline."""

    def pipeline():
        """Return a cached pipeline of a describer and a classifier."""
        steps = [
            ('describe', ImageToTextGenerator(instructions='Describe the image.', backend=CountingBackend())),
            ('classify', TextClassifier(backend='dummy')),
        ]
        return Pipeline(steps, memory=str(tmp_path))

    pipeline().fit(IMAGES, ['warm', 'cold', 'cold'])
    first = CountingBackend.counts['generate_text']
    assert first == 3
    pipeline().fit(IMAGES, ['warm', 'cold', 'cold'])
    assert CountingBackend.counts['generate_text'] == first


def test_refit_with_cache_and_string_backend(tmp_path, monkeypatch):
    """Hash components whose backend is a string, reusing cached outputs."""
    calls = []
    run = DummyBackend.run

    def counting_run(self, task, requests):
        """Count the requests and run them like the dummy backend."""
        calls.extend(requests)
        return run(self, task, requests)

    monkeypatch.setattr(DummyBackend, 'run', counting_run)
    steps = [
        ('describe', ImageToTextGenerator(instructions='Describe the image.', backend='dummy')),
        ('classify', TextClassifier(backend=CountingBackend())),
    ]
    Pipeline(steps, memory=str(tmp_path)).fit(IMAGES, ['warm', 'cold', 'cold'])
    described = len(calls)
    assert described == 3
    Pipeline(steps, memory=str(tmp_path)).fit(IMAGES, ['warm', 'cold', 'cold'])
    assert len(calls) == described


def test_parameters_are_tuned_across_steps(audio):
    """Fit and score every combination of parameters of two steps."""
    recordings = [Audio(audio.samples[: 1600 * (index + 1)], audio.sample_rate) for index in range(6)]
    pipeline = Pipeline(
        [
            ('transcriber', AudioToTextGenerator(backend='dummy')),
            ('classifier', TextClassifier(backend=DummyBackend())),
        ],
    )
    grid = {'transcriber__language': ['en', 'el'], 'classifier__backend__model': ['m1', 'm2']}
    search = GridSearchCV(pipeline, grid, cv=2).fit(recordings, ['a', 'b'] * 3)
    assert len(search.cv_results_['params']) == 4


def test_image_classifier_after_generator():
    """Classify generated images."""
    pipeline = make_pipeline(
        TextToImageGenerator(size=(8, 8), backend='dummy'),
        ImageClassifier(labels=['a', 'b'], backend='dummy'),
    )
    assert set(pipeline.fit(None).predict(['x', 'y'])) <= {'a', 'b'}
    assert np.asarray(pipeline.predict(['x'])).shape == (1,)
