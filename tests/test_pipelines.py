"""Test the pipelines that the library supports, one test per requirement of the pipelines spec."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

from skai.backends import DummyBackend
from skai.classification import TextClassifier
from skai.core import Audio, InvalidSampleError, Video
from skai.description import ImageDescriptor
from skai.embedding import ImageEmbedder, TextEmbedder
from skai.generation import ImageAnimator, ImageEditor, ImageGenerator, SpeechSynthesizer, TalkingAvatarGenerator
from skai.text import TextExtractor, TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber, VideoTranscriber

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


def test_written_files_feed_the_next_step(tmp_path):
    """Transcribe the files that the synthesizer wrote."""
    pipeline = make_pipeline(
        SpeechSynthesizer(output_format='path', output_dir=tmp_path, backend='dummy'),
        SpeechTranscriber(backend='dummy'),
    )
    transcripts = pipeline.fit(None).predict(['one', 'two words'])
    assert len(transcripts) == 2
    assert len(list(tmp_path.glob('*.wav'))) == 2


def test_mismatched_types_are_rejected():
    """Name the first sample when an image reaches a text component."""
    pipeline = make_pipeline(ImageGenerator(size=(4, 4), backend='dummy'), TextSummarizer(backend='dummy'))
    with pytest.raises(InvalidSampleError, match='Sample 0 must be a string'):
        pipeline.fit(None).predict(['a cat'])


def test_transcripts_feed_text_components(audio):
    """Summarize videos and classify recordings through their transcripts."""
    videos = [Video(np.zeros((10, 4, 4, 3), dtype=np.uint8), fps=5, audio=audio)] * 2
    pipeline = make_pipeline(VideoTranscriber(backend='dummy'), TextSummarizer(backend='dummy'))
    summaries = pipeline.fit(None).predict(videos)
    assert len(summaries) == 2
    calls = make_pipeline(SpeechTranscriber(backend='dummy'), TextClassifier(backend='dummy'))
    assert calls.fit([audio, audio], ['sales', 'support']).predict([audio]).shape == (1,)


def test_descriptions_feed_text_extraction():
    """Extract attributes from the descriptions of product photos."""
    fields = {'color': 'main color', 'material': 'main material'}
    pipeline = make_pipeline(ImageDescriptor(backend='dummy'), TextExtractor(fields=fields, backend='dummy'))
    records = pipeline.fit(IMAGES).predict(IMAGES)
    assert [set(record) for record in records] == [{'color', 'material'}] * 3


def test_speech_is_translated_into_speech(audio):
    """Turn speech into speech in another language."""
    pipeline = make_pipeline(
        SpeechTranscriber(backend='dummy'),
        TextTranslator(target_language='Greek', backend='dummy'),
        SpeechSynthesizer(backend='dummy'),
    )
    speech = pipeline.fit([audio]).predict([audio, audio])
    assert all(isinstance(output, Audio) for output in speech)


def test_media_is_narrated():
    """Speak the description of every image."""
    pipeline = make_pipeline(ImageDescriptor(backend='dummy'), SpeechSynthesizer(backend='dummy'))
    assert all(isinstance(output, Audio) for output in pipeline.fit(IMAGES).predict(IMAGES))


def test_talking_presenter_speaks_translated_scripts():
    """Return one video per script, as long as its synthesized speech."""
    synthesizer = SpeechSynthesizer(backend='dummy')
    pipeline = make_pipeline(
        TextTranslator(target_language='Spanish', backend='dummy'),
        synthesizer,
        TalkingAvatarGenerator(avatar=Image.new('RGB', (16, 16)), backend='dummy'),
    )
    scripts = ['Hello and welcome.', 'Thank you for watching our short video.']
    videos = pipeline.fit(None).predict(scripts)
    speech = synthesizer.predict(TextTranslator(target_language='Spanish', backend='dummy').fit().predict(scripts))
    assert [video.duration for video in videos] == pytest.approx([audio.duration for audio in speech], abs=1 / 25)


def test_prompts_become_animated_scenes():
    """Generate, edit and animate an image from one prompt."""
    pipeline = make_pipeline(
        ImageGenerator(size=(16, 16), backend='dummy'),
        ImageEditor(instruction='make it night', backend='dummy'),
        ImageAnimator(duration=1, backend='dummy'),
    )
    videos = pipeline.fit(None).predict(['a quiet harbor'])
    assert len(videos) == 1
    assert isinstance(videos[0], Video)


def test_embeddings_feed_scikit_learn_estimators():
    """Search images with text and learn from combined text features."""
    gallery = ImageEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(IMAGES)
    queries = TextEmbedder(normalize=True, center=False, backend='dummy:joint').fit().transform(['red', 'blue'])
    indices = NearestNeighbors(n_neighbors=1).fit(gallery).kneighbors(queries, return_distance=False)
    assert indices.shape == (2, 1)
    union = FeatureUnion([('a', TextEmbedder(backend='dummy:a')), ('b', TextEmbedder(dimensions=4, backend='dummy:b'))])
    model = make_pipeline(union, LogisticRegression()).fit(['x', 'y', 'z', 'w'], [0, 1, 0, 1])
    assert model[0].transform(['x']).shape == (1, 12)


def test_steps_run_once_per_fit(audio):
    """Send exactly one transcription request per training recording."""
    pipeline = make_pipeline(SpeechTranscriber(backend=CountingBackend()), TextClassifier(backend='dummy'))
    pipeline.fit([audio] * 10, ['a', 'b'] * 5)
    assert CountingBackend.counts['transcribe'] == 10


def test_cached_steps_are_reused(tmp_path):
    """Describe the images only during the first of two identical fits."""
    pipeline = Pipeline(
        [('describe', ImageDescriptor(backend=CountingBackend())), ('classify', TextClassifier(backend='dummy'))],
        memory=str(tmp_path),
    )
    pipeline.fit(IMAGES, ['warm', 'cold', 'cold'])
    first = CountingBackend.counts['describe_image']
    pipeline.fit(IMAGES, ['warm', 'cold', 'cold'])
    assert CountingBackend.counts['describe_image'] == first


def test_parameters_are_tuned_across_steps(audio):
    """Fit and score every combination of parameters of two steps."""
    recordings = [Audio(audio.samples[: 1600 * (index + 1)], audio.sample_rate) for index in range(6)]
    pipeline = Pipeline(
        [('transcriber', SpeechTranscriber(backend='dummy')), ('classifier', TextClassifier(backend=DummyBackend()))],
    )
    grid = {'transcriber__language': ['en', 'el'], 'classifier__backend__model': ['m1', 'm2']}
    search = GridSearchCV(pipeline, grid, cv=2).fit(recordings, ['a', 'b'] * 3)
    assert len(search.cv_results_['params']) == 4
