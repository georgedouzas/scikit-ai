"""Run chains of components on the OpenAI backend as scikit-learn pipelines."""

import re
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from skai import presets
from skai.classification import TextClassifier
from skai.core import Audio
from skai.embedding import TextEmbedder
from skai.generation import AudioToTextGenerator, ImageToTextGenerator, TextToAudioGenerator

DESCRIBE = 'Describe the image in one sentence.'
GREEK = re.compile('[Ͱ-Ͽ]')
TALK = (
    'Welcome to the weekly update. This week the team finished the new billing system, which now sends invoices '
    'automatically at the end of every month. Next week we will start testing the mobile app with ten customers.'
)


def test_descriptions_feed_text_classifier(openai_backend, shared_images):
    """Classify images through their descriptions."""
    pipeline = make_pipeline(
        ImageToTextGenerator(instructions=DESCRIBE, backend=openai_backend),
        TextClassifier(labels=['fruit', 'vehicle'], backend=openai_backend),
    )
    assert pipeline.fit(shared_images).predict(shared_images).tolist() == ['fruit', 'vehicle']


def test_speech_is_transcribed_summarized_and_translated(openai_backend):
    """Turn spoken English into a short Greek summary."""
    speech = TextToAudioGenerator(backend=openai_backend).fit().predict([TALK])
    pipeline = make_pipeline(
        AudioToTextGenerator(language='en', backend=openai_backend),
        presets.summarizer(max_length=20, backend=openai_backend),
        presets.translator('Greek', backend=openai_backend),
    )
    summary = pipeline.fit(speech).predict(speech)[0]
    assert GREEK.search(summary)
    assert len(summary.split()) <= 40


def test_files_pass_between_steps(openai_backend, tmp_path):
    """Transcribe speech that the synthesizer wrote to files."""
    pipeline = make_pipeline(
        TextToAudioGenerator(output_format='path', output_dir=tmp_path, backend=openai_backend),
        AudioToTextGenerator(language='en', backend=openai_backend),
    )
    transcripts = pipeline.fit(None).predict(['Paris is the capital of France.'])
    assert 'paris' in transcripts[0].lower()
    assert len(list(Path(tmp_path).glob('*.wav'))) == 1


def test_descriptions_are_narrated(openai_backend, shared_images):
    """Speak the description of every image."""
    pipeline = make_pipeline(
        ImageToTextGenerator(instructions=DESCRIBE, backend=openai_backend),
        TextToAudioGenerator(backend=openai_backend),
    )
    speech = pipeline.fit(shared_images).predict(shared_images)
    assert len(speech) == 2
    assert all(isinstance(audio, Audio) and audio.duration > 1 for audio in speech)


def test_description_embeddings_feed_logistic_regression(openai_backend, shared_images):
    """Learn a scikit-learn classifier on the embeddings of image descriptions."""
    pipeline = make_pipeline(
        ImageToTextGenerator(instructions=DESCRIBE, backend=openai_backend),
        TextEmbedder(backend=openai_backend),
        LogisticRegression(),
    )
    labels = ['fruit', 'vehicle']
    assert pipeline.fit(shared_images, labels).predict(shared_images).tolist() == labels
