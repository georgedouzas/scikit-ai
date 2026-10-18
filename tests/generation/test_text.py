"""Test the text generators, one test or more per scenario of the text-generation spec."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend
from skai.classification import TextClassifier
from skai.core import Audio, Backend, Transcript, UnusedFitDataWarning, Video
from skai.generation import (
    AudioToTextGenerator,
    ImageToTextGenerator,
    MultimodalToTextGenerator,
    TextToTextGenerator,
    VideoToTextGenerator,
)

TEXTS = [f'text number {index} about topic {index % 3}' for index in range(12)]
SUMMARIES = [f'topic {index % 3}' for index in range(12)]
IMAGES = [Image.new('RGB', (size, size)) for size in (8, 16, 24, 32)]
SPEAKING = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10, audio=Audio(np.zeros(32000), 16000))
SILENT = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)


class RecordingBackend(DummyBackend):
    """Keep every request and fine-tuning request the backend receives."""

    requests: ClassVar[list] = []
    tasks: ClassVar[list] = []
    tunings: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.tasks.extend([task] * len(requests))
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)

    def fine_tune(self, request):
        """Keep the request and fine-tune like the dummy backend."""
        RecordingBackend.tunings.append(request)
        return super().fine_tune(request)


class VerboseBackend(Backend):
    """Return texts that are longer than requested."""

    def generate_text(self, request):
        """Return the text twice."""
        return f'{request.sample} {request.sample}'


class NoContextBackend(DummyBackend):
    """Take no context."""

    accepts_context = False


@pytest.fixture(autouse=True)
def clear_records():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []
    RecordingBackend.tasks = []
    RecordingBackend.tunings = []


def test_summarize_translate_and_describe():
    """Return an object array with one string per sample from instructed generators."""
    summaries = TextToTextGenerator(instructions='Summarize in one sentence.', backend='dummy').fit().predict(TEXTS[:3])
    translations = TextToTextGenerator(instructions='Translate into Greek.', backend='dummy').fit().predict(TEXTS[:2])
    descriptions = ImageToTextGenerator(instructions='Describe the image.', backend='dummy').fit().predict(IMAGES[:2])
    for outputs, count in [(summaries, 3), (translations, 2), (descriptions, 2)]:
        assert isinstance(outputs, np.ndarray)
        assert outputs.dtype == object
        assert outputs.shape == (count,)
        assert all(isinstance(output, str) for output in outputs)
    assert descriptions.tolist() == ['A 8x8 image.', 'A 16x16 image.']


@pytest.mark.parametrize(
    ('generator', 'samples'),
    [
        (TextToTextGenerator(instructions='Summarize.', backend='dummy'), TEXTS[:2]),
        (ImageToTextGenerator(instructions='Describe.', backend='dummy'), IMAGES[:2]),
        (AudioToTextGenerator(backend='dummy'), [(np.zeros(8000), 16000)]),
        (VideoToTextGenerator(instructions='Describe.', backend='dummy'), [SPEAKING]),
        (MultimodalToTextGenerator(instructions='Caption.', backend='dummy'), [{'a': 'x', 'b': IMAGES[0]}]),
    ],
)
def test_predict_equals_transform(generator, samples):
    """Return equal outputs from `predict` and `transform`."""
    generator.fit()
    assert generator.predict(samples).tolist() == generator.transform(samples).tolist()


@pytest.mark.parametrize(
    'generator',
    [
        TextToTextGenerator(backend='dummy'),
        ImageToTextGenerator(backend='dummy'),
        AudioToTextGenerator(operation='instruct', backend='dummy'),
        VideoToTextGenerator(backend='dummy'),
        MultimodalToTextGenerator(backend='dummy'),
    ],
)
def test_missing_instructions(generator):
    """Reject instructed generation without `instructions` at `fit`."""
    with pytest.raises(InvalidParameterError, match='instructions'):
        generator.fit()


def test_instruct_is_the_default_operation_except_for_audio():
    """Default to `instruct` for every text generator except `AudioToTextGenerator`."""
    for generator_class in (TextToTextGenerator, ImageToTextGenerator, VideoToTextGenerator, MultimodalToTextGenerator):
        assert generator_class().operation == 'instruct'
        assert next(iter(generator_class.operations)) == 'instruct'
    assert AudioToTextGenerator().operation == 'transcribe'
    assert list(AudioToTextGenerator.operations) == ['transcribe', 'instruct']
    assert list(ImageToTextGenerator.operations) == ['instruct', 'transcribe']
    assert list(VideoToTextGenerator.operations) == ['instruct', 'transcribe']


def test_multimodal_context():
    """Send the image and the text of a multimodal sample in one request, each named by its key."""
    image = Image.new('RGB', (8, 8))
    generator = MultimodalToTextGenerator(instructions='Write a caption.', backend=RecordingBackend()).fit()
    caption = generator.predict([{'photo': image, 'notes': 'taken in Crete'}])
    assert caption.shape == (1,)
    (request,) = RecordingBackend.requests
    assert set(request.sample) == {'photo', 'notes'}
    assert isinstance(request.sample['photo'], Image.Image)
    assert request.sample['notes'] == 'taken in Crete'
    assert request.instructions == 'Write a caption.'


def test_multimodal_dataframe_rows():
    """Read every row of a DataFrame as one sample keyed by its columns."""
    frame = pd.DataFrame({'image': [Image.new('RGB', (8, 8)), Image.new('RGB', (4, 4))], 'caption': ['one', 'two']})
    outputs = MultimodalToTextGenerator(instructions='Caption.', backend=RecordingBackend()).fit().predict(frame)
    assert outputs.tolist() == ['image: A 8x8 image. caption: one', 'image: A 4x4 image. caption: two']
    assert all(set(request.sample) == {'image', 'caption'} for request in RecordingBackend.requests)


def test_instructed_audio_and_video():
    """Describe audio and the sampled frames of videos, with or without an audio track."""
    audio = AudioToTextGenerator(operation='instruct', instructions='Describe.', backend='dummy').fit()
    assert audio.predict([(np.zeros(8000), 16000)]).tolist() == ['0.50 seconds of audio.']
    video = VideoToTextGenerator(instructions='Describe.', n_frames=4, backend='dummy').fit()
    assert video.predict([SPEAKING, SILENT]).tolist() == ['4 frames of a 2.00 second video.'] * 2


def test_speech_video_and_page_transcripts(audio):
    """Return one transcript string per sample from the transcribe operation."""
    speech = AudioToTextGenerator(backend='dummy').fit().predict([audio, audio])
    assert speech.tolist() == ['Speech of 1.00 seconds.'] * 2
    video = VideoToTextGenerator(operation='transcribe', backend='dummy').fit().predict([SPEAKING])
    assert video.tolist() == ['Speech of 2.00 seconds.']
    page = ImageToTextGenerator(operation='transcribe', format='markdown', backend='dummy').fit().predict(IMAGES[:1])
    assert page.tolist() == ['# Text of a 8x8 image.']
    plain = ImageToTextGenerator(operation='transcribe', backend='dummy').fit().predict(IMAGES[:1])
    assert plain.tolist() == ['Text of a 8x8 image.']


def test_page_transcription_ignores_references():
    """Learn nothing from references when transcribing the text of images."""
    ocr = ImageToTextGenerator(operation='transcribe', backend='dummy').fit(IMAGES[:2], ['Some Text', 'More Text'])
    assert ocr.examples_ == ()
    assert ocr.predict(IMAGES[:1]).tolist() == ['Text of a 8x8 image.']


def test_transcription_takes_language(audio):
    """Send the language with every transcription request."""
    AudioToTextGenerator(language='el', backend=RecordingBackend()).fit().predict([audio])
    assert RecordingBackend.requests[-1].language == 'el'


def test_silent_video():
    """Return an empty transcript for a video without an audio track, without calling the backend for it."""
    transcripts = (
        VideoToTextGenerator(operation='transcribe', backend=RecordingBackend()).fit().predict([SILENT, SPEAKING])
    )
    assert transcripts.tolist() == ['', 'Speech of 2.00 seconds.']
    assert len(RecordingBackend.requests) == 1
    assert VideoToTextGenerator(operation='transcribe', backend=RecordingBackend()).fit().predict(
        [SILENT],
    ).tolist() == ['']
    assert len(RecordingBackend.requests) == 1


@pytest.mark.parametrize('generator_class', [AudioToTextGenerator, VideoToTextGenerator])
def test_domain_terms(audio, generator_class):
    """Learn the domain terms of reference transcripts and send them with every request."""
    sample = audio if generator_class is AudioToTextGenerator else SPEAKING
    transcriber = generator_class(operation='transcribe', backend=RecordingBackend())
    transcriber.fit([sample], ['We run Kubernetes over gRPC.'])
    assert set(transcriber.vocabulary_) == {'Kubernetes', 'gRPC'}
    transcriber.predict([sample, sample])
    assert [request.vocabulary for request in RecordingBackend.requests] == [transcriber.vocabulary_] * 2


def test_vocabulary_is_ordered_and_capped(audio):
    """Order the vocabulary by frequency and keep at most `max_vocabulary` terms."""
    references = ['We use Kubernetes.', 'We run Kubernetes over gRPC.', 'Our Kubernetes, gRPC and S3.']
    transcriber = AudioToTextGenerator(max_vocabulary=2, backend='dummy').fit([audio] * 3, references)
    assert transcriber.vocabulary_ == ('Kubernetes', 'gRPC')
    assert AudioToTextGenerator(backend='dummy').fit().vocabulary_ == ()


def test_vocabulary_warns_without_context(audio):
    """Warn at `fit` when the backend cannot take the learned vocabulary."""
    with pytest.warns(UnusedFitDataWarning, match='vocabulary is not used'):
        AudioToTextGenerator(backend=NoContextBackend()).fit([audio], ['We run Kubernetes.'])


def test_learned_length_and_examples():
    """Learn the length from the references and send `k_shot` example pairs with every request."""
    references = [' '.join(['word'] * length) for length in (8, 9, 10, 11, 12)]
    generator = TextToTextGenerator(instructions='Summarize.', max_length='auto', k_shot=2, backend=RecordingBackend())
    generator.fit(TEXTS[:5], references)
    assert 8 <= generator.max_length_ <= 12
    outputs = generator.predict([' '.join(['token'] * 50), 'short text'])
    assert all(len(output.split()) <= generator.max_length_ for output in outputs)
    assert [len(request.examples) for request in RecordingBackend.requests] == [2, 2]
    assert all(reference in references for _, reference in RecordingBackend.requests[0].examples)


def test_auto_length_without_references():
    """Set no length limit when `max_length='auto'` has no references to learn from."""
    assert TextToTextGenerator(instructions='Summarize.', max_length='auto', backend='dummy').fit().max_length_ is None


def test_integer_length_cuts_outputs():
    """Cut outputs that the backend returns longer than `max_length`."""
    text = ' '.join(f'word{index}' for index in range(200))
    summary = TextToTextGenerator(instructions='Summarize.', max_length=50, backend='dummy').fit().predict([text])[0]
    assert len(summary.split()) <= 50
    generator = TextToTextGenerator(instructions='Summarize.', max_length=3, backend=VerboseBackend()).fit()
    assert generator.predict(['one two']).tolist() == ['one two one']
    assert generator.max_length_ == 3


def test_examples_are_capped_and_reproducible():
    """Pass at most `max_examples` pairs, the same ones for the same `random_state`."""

    def examples(**params):
        """Return the examples of a summarizer fitted on the texts."""
        return TextToTextGenerator(instructions='Summarize.', backend='dummy', **params).fit(TEXTS, SUMMARIES).examples_

    assert len(examples(max_examples=3, random_state=0)) == 3
    assert examples(max_examples=3, random_state=0) == examples(max_examples=3, random_state=0)
    assert examples(k_shot=0) == ()
    assert examples(k_shot=[1, 4])[1] == (TEXTS[4], SUMMARIES[4])


def test_media_generators_keep_examples(audio, image):
    """Keep sample and reference pairs as examples for instructed audio and image generators."""
    describer = AudioToTextGenerator(operation='instruct', instructions='Describe.', k_shot=1, random_state=0)
    assert len(describer.set_params(backend='dummy').fit([audio, audio], ['Rain.', 'Wind.']).examples_) == 1
    images = [image, Image.new('RGB', (8, 8)), Image.new('RGB', (16, 16))]
    descriptions = ['A red block.', 'A black block.', 'A larger black block.']
    generator = ImageToTextGenerator(instructions='Describe.', k_shot=2, random_state=0, backend=RecordingBackend())
    generator.fit(images, descriptions).predict([image])
    assert len(RecordingBackend.requests[-1].examples) == 2


def test_consistent_product_name():
    """Map a product name that both translations keep, and send the glossary with every request."""
    sources = ['Open SkaiBox now.', 'SkaiBox is ready.', 'Hello there.']
    references = ['Ouvrez SkaiBox.', 'SkaiBox est prête.', 'Bonjour.']
    translator = TextToTextGenerator(
        instructions='Translate into French.',
        learn_terms=True,
        backend=RecordingBackend(),
    )
    translator.fit(sources, references)
    assert translator.glossary_ == (('SkaiBox', 'SkaiBox'),)
    translator.predict(['Close SkaiBox.'])
    assert RecordingBackend.requests[-1].glossary == (('SkaiBox', 'SkaiBox'),)
    plain = TextToTextGenerator(instructions='Translate into French.', backend='dummy').fit(sources, references)
    assert plain.glossary_ == ()


def test_fine_tuned_generator():
    """Train on every text and reference, and send later requests to the new model without examples."""
    texts = [f'review number {index}' for index in range(20)]
    references = [text.split()[-1] for text in texts]
    generator = TextToTextGenerator(instructions='Summarize.', adaptation='finetune', backend=RecordingBackend())
    generator.fit(texts, references)
    assert generator.model_.startswith('dummy-finetuned-')
    assert generator.backend_.model == generator.model_
    assert generator.examples_ == ()
    tuning = RecordingBackend.tunings[-1]
    assert tuning.task == 'generate_text'
    assert len(tuning.examples) == 20
    assert tuning.examples[0][1] == references[0]
    generator.predict(texts[:2])
    assert [request.examples for request in RecordingBackend.requests] == [(), ()]


def test_fine_tuned_image_generator():
    """Fine-tune an instructed image generator with one image in every training prompt."""
    images = [Image.new('RGB', (4, 4), (index, 0, 0)) for index in range(20)]
    references = [f'Shade {index}.' for index in range(20)]
    generator = ImageToTextGenerator(instructions='Describe.', adaptation='finetune', backend=RecordingBackend())
    generator.fit(images, references)
    assert len(RecordingBackend.tunings[-1].examples[0][0].images) == 1
    assert generator.predict(images[:1])[0]


def test_fine_tuning_needs_references_and_instruct():
    """Reject fine-tuning without references or for transcription."""
    with pytest.raises(ValueError, match='reference outputs'):
        TextToTextGenerator(instructions='Summarize.', adaptation='finetune', backend='dummy').fit()
    with pytest.raises(ValueError, match="only with `operation='instruct'`"):
        AudioToTextGenerator(adaptation='finetune', backend='dummy').fit()


def test_fitted_state_changes_requests():
    """Send different requests after a fit with data than after a fit without it."""
    TextToTextGenerator(instructions='Summarize.', backend=RecordingBackend()).fit().predict(['new text'])
    TextToTextGenerator(instructions='Summarize.', backend=RecordingBackend()).fit(TEXTS, SUMMARIES).predict(
        ['new text'],
    )
    zero_shot, few_shot = RecordingBackend.requests
    assert zero_shot.examples == ()
    assert few_shot.examples
    assert zero_shot != few_shot


def test_references_scored_as_targets():
    """Learn from training references and score against test references in every fold."""

    def overlap(references, summaries):
        """Return the share of summaries that contain their reference's topic number."""
        pairs = zip(references, summaries, strict=True)
        return np.mean([reference.split()[-1] in summary for reference, summary in pairs])

    generator = TextToTextGenerator(instructions='Summarize the text.', backend='dummy')
    scores = cross_val_score(generator, TEXTS, SUMMARIES, cv=3, scoring=make_scorer(overlap))
    assert scores.shape == (3,)
    assert np.all(scores == 1.0)


def test_mismatched_targets():
    """Reject references whose length differs from the samples, naming both lengths."""
    with pytest.raises(ValueError, match='3 samples but `y` has 2 references'):
        TextToTextGenerator(instructions='Summarize.', backend='dummy').fit(TEXTS[:3], SUMMARIES[:2])


def test_pipeline_target_is_not_read_as_references(audio):
    """Learn no vocabulary from class labels in an intermediate transcriber, and let the classifier learn them."""
    pipeline = make_pipeline(AudioToTextGenerator(backend='dummy'), TextClassifier(backend='dummy'))
    pipeline.fit([audio, audio], ['Kubernetes', 'Docker'])
    assert pipeline[0].vocabulary_ == ()
    assert pipeline[1].classes_.tolist() == ['Docker', 'Kubernetes']
    assert pipeline.predict([audio]).shape == (1,)


def test_supervised_final_step(audio):
    """Learn from references in the final text generator and not in the intermediate transcriber."""
    pipeline = make_pipeline(
        AudioToTextGenerator(backend='dummy'),
        TextToTextGenerator(instructions='Summarize the text.', backend='dummy'),
    )
    pipeline.fit([audio, audio], ['First Summary.', 'Second Summary.'])
    assert pipeline[0].vocabulary_ == ()
    assert [reference for _, reference in pipeline[1].examples_] == ['First Summary.', 'Second Summary.']


def test_use_y_override(audio):
    """Learn in `fit_transform` with `use_y=True`, and never with `use_y=False`."""
    transcriber = AudioToTextGenerator(use_y=True, backend='dummy')
    transcriber.fit_transform([audio], ['We use Kubernetes.'])
    assert transcriber.vocabulary_ == ('Kubernetes',)
    pipeline = make_pipeline(
        AudioToTextGenerator(use_y=True, backend='dummy'),
        TextToTextGenerator(instructions='Summarize.', backend='dummy'),
    )
    pipeline.fit([audio], ['We use Kubernetes.'])
    assert pipeline[0].vocabulary_ == ('Kubernetes',)
    assert AudioToTextGenerator(use_y=False, backend='dummy').fit([audio], ['We use Kubernetes.']).vocabulary_ == ()


def test_descriptions_feed_text_classifier():
    """Classify images and audio through their descriptions in a pipeline."""
    pipeline = make_pipeline(
        ImageToTextGenerator(instructions='Describe.', backend='dummy'),
        TextClassifier(backend='dummy'),
    )
    predictions = pipeline.fit(IMAGES, ['small', 'small', 'large', 'large']).predict(IMAGES)
    assert len(predictions) == 4
    assert set(predictions) <= {'small', 'large'}
    sounds = [(np.zeros(size, dtype=np.float32), 16000) for size in (8000, 16000)]
    describer = AudioToTextGenerator(operation='instruct', instructions='Describe.', backend='dummy')
    assert set(make_pipeline(describer, TextClassifier(backend='dummy')).fit(sounds, ['a', 'b']).predict(sounds)) <= {
        'a',
        'b',
    }


def test_transcripts_are_text_not_records(audio):
    """Return plain strings, not transcript objects, from transcription."""
    output = AudioToTextGenerator(backend='dummy').fit().predict([audio])[0]
    assert isinstance(output, str)
    assert not isinstance(output, Transcript)
