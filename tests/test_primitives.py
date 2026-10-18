"""Test the pipeline primitives, one test or more per scenario of the pipeline-primitives spec."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline

from skai.backends import DummyBackend
from skai.classification import TextClassifier, VideoClassifier
from skai.core import Audio, InvalidSampleError, Video
from skai.embedding import TextEmbedder, VideoEmbedder
from skai.generation import (
    ImageToTextGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
    VideoToTextGenerator,
)
from skai.pipeline import Field, ForEach, MediaUnion, Retriever, TopK
from skai.segmentation import VideoSegmenter

SPEECH = Audio(np.zeros(16000, dtype=np.float32), 16000)
STORYBOARDS = [
    [{'visual': f'scene {index}', 'narration': f'Line {index}.'} for index in range(3)],
    [{'visual': f'shot {index}', 'narration': f'Words {index}.'} for index in range(2)],
]


def clip(value, frames=8):
    """Return a short video of one gray level with a silent track."""
    return Video(np.full((frames, 8, 8, 3), value, dtype=np.uint8), fps=8, audio=SPEECH)


class CountingBackend(DummyBackend):
    """Count the calls and requests of every task."""

    calls: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the size of the call and run it like the dummy backend."""
        CountingBackend.calls.append((task, len(requests)))
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_calls():
    """Start every test with no recorded calls."""
    CountingBackend.calls = []


def translated_speech():
    """Return the union of the original video and its translated speech."""
    audio = make_pipeline(
        VideoToTextGenerator(operation='transcribe', backend='dummy'),
        TextToTextGenerator(instructions='Translate into Greek.', backend='dummy'),
        TextToAudioGenerator(backend='dummy'),
    )
    return MediaUnion([('video', 'passthrough'), ('audio', audio)])


def test_video_and_translated_speech_side_by_side():
    """Return one dict per video with the original video and the new audio."""
    videos = [clip(0), clip(100, frames=16)]
    samples = translated_speech().fit(videos).transform(videos)
    assert len(samples) == 2
    assert all(set(sample) == {'video', 'audio'} for sample in samples)
    assert [sample['video'] for sample in samples] == videos
    assert all(isinstance(sample['audio'], Audio) for sample in samples)


def test_union_fit_transform_equals_fit_then_transform():
    """Return the same samples from `fit_transform` and from `fit` then `transform`."""
    videos = [clip(0), clip(50)]
    fitted = translated_speech().fit_transform(videos)
    assert [sample['audio'].duration for sample in fitted] == [
        sample['audio'].duration for sample in translated_speech().fit(videos).transform(videos)
    ]


def test_union_routes_parameters_and_clones():
    """Route `<branch>__<parameter>` to the branches and clone like any estimator."""
    union = translated_speech()
    params = union.get_params()
    assert params['audio__texttotextgenerator__instructions'] == 'Translate into Greek.'
    union.set_params(audio__texttotextgenerator__instructions='Translate into French.')
    assert union.transformer_list[1][1][1].instructions == 'Translate into French.'
    copy = clone(union)
    assert copy is not union
    assert copy.get_params()['audio__texttotextgenerator__instructions'] == 'Translate into French.'
    with pytest.raises(ValueError, match='not unique'):
        MediaUnion([('a', 'passthrough'), ('a', 'passthrough')]).fit(['x'])


def test_union_takes_predict_of_branches_without_transform():
    """Use the predictions of a branch that has no `transform`, and predict like it transforms."""
    texts = ['good movie', 'bad movie', 'great film']
    union = MediaUnion([('text', 'passthrough'), ('label', TextClassifier(labels=['pos', 'neg'], backend='dummy'))])
    samples = union.fit(texts).predict(texts)
    assert [sample['text'] for sample in samples] == texts
    assert all(sample['label'] in {'pos', 'neg'} for sample in samples)
    assert [sample['label'] for sample in union.fit_transform(texts)] == [sample['label'] for sample in samples]


def test_one_field():
    """Return the values of one field in order, without fitting."""
    samples = [{'visual': 'a boat', 'narration': 'At sea.'}, {'visual': 'a car', 'narration': 'On land.'}]
    assert Field('visual').fit_transform(samples).tolist() == ['a boat', 'a car']
    assert Field('narration').transform(samples).tolist() == ['At sea.', 'On land.']


def test_several_fields():
    """Return dicts of the listed fields."""
    samples = [{'a': 1, 'b': 2, 'c': 3}]
    assert Field(['a', 'c']).fit().transform(samples).tolist() == [{'a': 1, 'c': 3}]


def test_missing_field():
    """Name the missing field and the index of the sample."""
    with pytest.raises(KeyError, match=r'Sample 1.*`visual`'):
        Field('visual').transform([{'visual': 'a'}, {'narration': 'b'}])
    with pytest.raises(KeyError, match=r'Sample 0.*`b`'):
        Field(['a', 'b']).transform([{'a': 1}])


def scenes(**params):
    """Return a step that draws the visual of every scene."""
    return ForEach(
        make_pipeline(Field('visual'), TextToImageGenerator(size=(64, 64), backend=CountingBackend(), **params)),
    )


def test_every_scene():
    """Return lists of three and two images, from one batch of five requests to the backend."""
    step = scenes(batch_size=8).fit(STORYBOARDS)
    CountingBackend.calls = []
    images = step.transform(STORYBOARDS)
    assert [len(item) for item in images] == [3, 2]
    assert all(isinstance(image, Image.Image) and image.size == (64, 64) for item in images for image in item)
    assert CountingBackend.calls == [('generate_image', 5)]


def test_for_each_runs_every_item_once_in_fit_transform():
    """Run every item once in `fit_transform`, through one call of the inner estimator."""
    images = scenes(batch_size=8).fit_transform(STORYBOARDS)
    assert [len(item) for item in images] == [3, 2]
    assert CountingBackend.calls == [('generate_image', 5)]


def test_for_each_fits_on_all_items_without_targets():
    """Fit the inner estimator on the items of all samples with `y=None`."""
    texts = [['We use Kubernetes.', 'And gRPC.'], ['Over S3.']]
    step = ForEach(TextToTextGenerator(instructions='Summarize.', backend='dummy')).fit(texts, ['ignored', 'too'])
    assert step.estimator_.examples_ == ()
    assert step.predict(texts).tolist() == step.transform(texts).tolist()


def test_empty_lists():
    """Return an empty list for an empty sample, and for all samples when every one is empty."""
    images = scenes().fit(STORYBOARDS).transform([STORYBOARDS[1], []])
    assert [len(item) for item in images] == [2, 0]
    assert scenes().fit_transform([[], []]).tolist() == [[], []]
    assert scenes().fit([[]]).transform([[]]).tolist() == [[]]


def test_for_each_needs_lists():
    """Reject samples that are not lists, naming the index."""
    with pytest.raises(InvalidSampleError, match='Sample 1'):
        scenes().fit_transform([STORYBOARDS[0], 'a scene'])


def test_for_each_is_cached(tmp_path):
    """Reuse the cached outputs of a `ForEach` step when a pipeline with `memory` is fitted again."""

    def pipeline():
        """Return a cached pipeline that draws and then describes every scene."""
        describe = ForEach(ImageToTextGenerator(instructions='Describe.', backend='dummy'))
        return Pipeline([('draw', scenes(batch_size=8)), ('describe', describe)], memory=str(tmp_path))

    first = pipeline().fit(STORYBOARDS)
    assert CountingBackend.calls == [('generate_image', 5)]
    second = pipeline().fit(STORYBOARDS)
    assert CountingBackend.calls == [('generate_image', 5)]
    assert [len(item) for item in second.predict(STORYBOARDS)] == [3, 2]
    assert first.named_steps['draw'].estimator_ is not None


def test_best_moments():
    """Keep the two clips with the highest `exciting` probability, in their original order."""
    clips = [clip(value) for value in (0, 40, 80, 120, 160)]
    classifier = VideoClassifier(labels=['exciting', 'calm'], backend='dummy')
    fitted = clone(classifier).fit()
    scores = fitted.predict_proba(clips)[:, fitted.classes_.tolist().index('exciting')]
    expected = sorted(np.argsort(-scores)[:2].tolist())
    kept = TopK(classifier, label='exciting', k=2).fit().transform([clips])
    assert len(kept) == 1
    assert [int(item.frames[0, 0, 0, 0]) for item in kept[0]] == [(0, 40, 80, 120, 160)[index] for index in expected]


def test_top_k_with_a_callable_and_threshold():
    """Score items with a callable and keep the best, or those above `threshold`, in their order."""
    samples = [[3, 9, 1, 7], [5], []]
    assert TopK(float, k=2).fit().transform(samples).tolist() == [[9, 7], [5], []]
    assert TopK(float, threshold=4).fit().transform(samples).tolist() == [[9, 7], [5], []]
    assert TopK(float, k=1, threshold=8).fit().transform(samples).tolist() == [[9], [], []]


def test_top_k_needs_k_or_threshold_and_label():
    """Reject a `TopK` without `k` or `threshold`, and a classifier scorer without `label`."""
    with pytest.raises(ValueError, match='`k` or `threshold`'):
        TopK(float).fit()
    with pytest.raises(ValueError, match='label'):
        TopK(VideoClassifier(labels=['a', 'b'], backend='dummy'), k=1).fit()


def test_shots_found_by_text():
    """Return two shot records per query, each with `sample`, `start` and `end`."""
    frames = np.concatenate([np.zeros((8, 8, 8, 3), np.uint8), np.full((8, 8, 8, 3), 255, np.uint8)])
    videos = [Video(frames, fps=8), Video(frames[::-1].copy(), fps=8)]
    shot = MediaUnion(
        [
            ('start', Field('start')),
            ('end', Field('end')),
            ('vector', make_pipeline(Field('media'), VideoEmbedder(n_frames=2, center=False, backend='dummy:joint'))),
        ],
    )
    items = make_pipeline(VideoSegmenter(by='shot'), ForEach(shot))
    retriever = Retriever(items, TextEmbedder(center=False, backend='dummy:joint')).fit(videos)
    neighbors = retriever.kneighbors(['a goal', 'a corner'], n_neighbors=2)
    assert len(neighbors) == 2
    for records in neighbors:
        assert len(records) == 2
        assert all({'sample', 'start', 'end'} <= set(record) for record in records)
        assert all(record['sample'] in {0, 1} and record['start'] <= record['end'] for record in records)
        assert records[0]['distance'] <= records[1]['distance']


def test_retriever_indexes_one_vector_per_sample():
    """Index one vector per sample and cap the neighbors at the number of items."""
    texts = ['a goal', 'a corner', 'a foul']
    retriever = Retriever(TextEmbedder(center=False, backend='dummy'), TextEmbedder(backend='dummy')).fit(texts)
    (records,) = retriever.kneighbors(['a goal'], n_neighbors=10)
    assert len(records) == 3
    assert records[0]['sample'] == 0
    assert records[0]['distance'] == pytest.approx(0.0, abs=1e-9)
