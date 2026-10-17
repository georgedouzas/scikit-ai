"""Test the OpenAI backend against a fake client."""

import base64
import json

import numpy as np
import pytest
from PIL import Image

from skai.backends import OpenAIBackend
from skai.classification import ImageClassifier, TextClassifier
from skai.core import (
    Audio,
    ExtractionRequest,
    FineTuneError,
    InvalidResponseError,
    SampleError,
    UnsupportedBackendError,
)
from skai.description import ImageDescriptor
from skai.embedding import TextEmbedder
from skai.generation import ImageGenerator, SpeechSynthesizer
from skai.text import TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber
from tests.fakes import Recorder, chat_response, install_module, namespace, png_of, round_trip, wav_of


def make_client(answer='positive'):
    """Return a fake OpenAI client whose chat model always gives `answer`."""
    transcription = namespace(
        text=' hello world ',
        segments=[namespace(start=0.0, end=1.0, text=' hello world')],
        words=[namespace(start=0.0, end=0.4, word='hello'), namespace(start=0.4, end=1.0, word='world')],
    )
    return namespace(
        chat=namespace(completions=namespace(create=Recorder(lambda **kwargs: chat_response(answer)))),
        embeddings=namespace(create=Recorder(lambda **kwargs: namespace(data=[namespace(embedding=[0.5, 0.5, 0.0])]))),
        audio=namespace(
            speech=namespace(create=Recorder(lambda **kwargs: namespace(content=wav_of(0.5)))),
            transcriptions=namespace(create=Recorder(lambda **kwargs: transcription)),
        ),
        images=namespace(
            generate=Recorder(
                lambda **kwargs: namespace(
                    data=[namespace(b64_json=base64.b64encode(png_of((16, 8))).decode())],
                ),
            ),
        ),
    )


def last_messages(client):
    """Return the messages of the last chat call."""
    return client.chat.completions.create.calls[-1]['messages']


def test_connect_creates_client_from_environment(monkeypatch):
    """Create the client with the API key and base URL."""
    created = {}
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: created.update(kwargs) or make_client())
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    backend = OpenAIBackend(base_url='http://localhost:1234/v1').connect()
    assert created == {'api_key': 'sk-test', 'base_url': 'http://localhost:1234/v1'}
    assert backend.supported_tasks() >= {'summarize', 'transcribe', 'generate_image'}


def test_unpickled_backend_reconnects(monkeypatch):
    """Create the client again on first use after it was dropped."""
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: make_client('restored'))
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    summarizer = round_trip(TextSummarizer(backend='openai').fit())
    assert summarizer.predict(['text'])[0] == 'restored'


def test_text_classifier_parses_named_label():
    """Prompt with the labels and turn the named label into a one-hot score."""
    client = make_client('Positive.')
    classifier = TextClassifier(labels=['positive', 'negative'], backend=OpenAIBackend(client=client)).fit()
    assert classifier.predict(['I love it']).tolist() == ['positive']
    np.testing.assert_array_equal(classifier.predict_proba(['I love it']), [[0.0, 1.0]])
    system, user = last_messages(client)
    assert 'exactly one label' in system['content']
    assert '- positive' in user['content'][0]['text']
    assert user['content'][0]['text'].endswith('Input: I love it\nOutput:')


def test_text_classifier_finds_label_in_sentence():
    """Find the one label that a longer answer names."""
    client = make_client('I would say it is negative overall')
    classifier = TextClassifier(labels=['positive', 'negative'], backend=OpenAIBackend(client=client)).fit()
    assert classifier.predict(['meh']).tolist() == ['negative']


def test_text_classifier_includes_examples_and_descriptions():
    """Put label descriptions, instructions and few-shot examples in the prompt."""
    client = make_client('1')
    classifier = TextClassifier(
        labels={0: 'negative', 1: 'positive'},
        instructions='Be strict.',
        k_shot=[0, 1],
        backend=OpenAIBackend(client=client),
    )
    classifier.fit(['bad', 'good'], [0, 1])
    assert classifier.predict(['fine']).tolist() == [1]
    text = last_messages(client)[1]['content'][0]['text']
    assert '- 0: negative' in text
    assert 'Be strict.' in text
    assert 'Example 1:\nInput: bad\nOutput: 0' in text
    assert 'Example 2:\nInput: good\nOutput: 1' in text


def test_text_classifier_rejects_ambiguous_answer():
    """Report an answer that names no single label."""
    classifier = TextClassifier(labels=['positive', 'negative'], backend=OpenAIBackend(client=make_client('unsure')))
    with pytest.raises(SampleError, match='does not name exactly one') as error:
        classifier.fit().predict(['meh'])
    assert isinstance(error.value.__cause__, InvalidResponseError)


def test_text_classifier_parses_multilabel_answers():
    """Turn comma-separated labels and `none` into indicator rows."""
    y = np.array([[1, 0, 1], [0, 1, 0]])
    for answer, expected in [('a, c', [1, 0, 1]), ('none', [0, 0, 0])]:
        client = make_client(answer)
        classifier = TextClassifier(labels=['a', 'b', 'c'], backend=OpenAIBackend(client=client)).fit(['x', 'y'], y)
        assert classifier.predict(['z']).tolist() == [expected]
        assert 'separated by commas' in last_messages(client)[0]['content']


def test_options_are_forwarded():
    """Pass `options` to every API call."""
    client = make_client('short')
    TextSummarizer(backend=OpenAIBackend(client=client, options={'temperature': 0})).fit().predict(['long text'])
    assert client.chat.completions.create.calls[-1]['temperature'] == 0


def test_summarizer_and_translator_prompt_the_model():
    """Ask for the summary length, the style and the target language."""
    client = make_client(' answer ')
    backend = OpenAIBackend(model='gpt-test', client=client)
    assert TextSummarizer(max_length=20, style='formal', backend=backend).fit().predict(['text'])[0] == 'answer'
    assert 'at most 20 words. Write it in this style: formal.' in last_messages(client)[1]['content'][0]['text']
    assert client.chat.completions.create.calls[-1]['model'] == 'gpt-test'
    TextTranslator(target_language='Greek', source_language='English', backend=backend).fit().predict(['hi'])
    assert 'from English into Greek' in last_messages(client)[1]['content'][0]['text']


def test_image_descriptor_sends_image():
    """Send the image as a data URL with the detail instruction."""
    client = make_client('A red square.')
    descriptor = ImageDescriptor(
        detail='detailed',
        instructions='Name the color.',
        backend=OpenAIBackend(client=client),
    )
    assert descriptor.fit().predict([Image.new('RGB', (4, 4), 'red')]).tolist() == ['A red square.']
    content = last_messages(client)[1]['content']
    assert 'one detailed paragraph. Name the color.' in content[0]['text']
    assert content[1]['image_url']['url'].startswith('data:image/png;base64,')


def test_image_classifier_sends_example_images():
    """Send the example images before the image to classify."""
    client = make_client('dog')
    images = [Image.new('RGB', (4, 4), color) for color in ('red', 'blue')]
    classifier = ImageClassifier(k_shot=2, random_state=0, backend=OpenAIBackend(client=client))
    classifier.fit(images, ['cat', 'dog'])
    assert classifier.predict([images[0]]).tolist() == ['dog']
    content = last_messages(client)[1]['content']
    assert len(content) == 4
    assert '<image 3>' in content[0]['text']


def test_embedder_requests_dimensions():
    """Ask the embedding model for the requested dimensions."""
    client = make_client()
    embeddings = TextEmbedder(dimensions=3, backend=OpenAIBackend(client=client)).fit().transform(['a'])
    np.testing.assert_allclose(embeddings, [[0.5, 0.5, 0.0]])
    assert client.embeddings.create.calls[-1]['dimensions'] == 3
    assert client.embeddings.create.calls[-1]['model'] == 'text-embedding-3-small'


def test_synthesizer_uses_voice():
    """Request WAV speech in the chosen voice."""
    client = make_client()
    speech = SpeechSynthesizer(voice='nova', backend=OpenAIBackend(client=client)).fit().predict(['hello'])
    assert isinstance(speech[0], Audio)
    assert speech[0].duration == pytest.approx(0.5)
    call = client.audio.speech.create.calls[-1]
    assert (call['voice'], call['response_format'], call['input']) == ('nova', 'wav', 'hello')
    SpeechSynthesizer(backend=OpenAIBackend(client=client, voice='echo')).fit().predict(['hi'])
    assert client.audio.speech.create.calls[-1]['voice'] == 'echo'


def test_synthesizer_cannot_clone_voices(audio):
    """Reject reference audio and list the backends that clone voices."""
    synthesizer = SpeechSynthesizer(voice=audio.to_bytes(), backend=OpenAIBackend(client=make_client()))
    with pytest.raises(UnsupportedBackendError, match=r"cannot clone voices.*\['dummy', 'replicate'\]"):
        synthesizer.fit()


def test_image_generator_requests_size():
    """Request the size and the styled prompt."""
    client = make_client()
    generator = ImageGenerator(size=(16, 8), style='flat', backend=OpenAIBackend(client=client)).fit()
    assert generator.predict(['a cat'])[0].size == (16, 8)
    call = client.images.generate.calls[-1]
    assert (call['size'], call['prompt']) == ('16x8', 'a cat. Style: flat')


@pytest.mark.parametrize(
    ('timestamps', 'expected'),
    [
        (None, 'hello world'),
        ('segment', [{'start': 0.0, 'end': 1.0, 'text': 'hello world'}]),
        ('word', [{'start': 0.0, 'end': 0.4, 'text': 'hello'}, {'start': 0.4, 'end': 1.0, 'text': 'world'}]),
    ],
)
def test_transcriber_reads_text_and_timestamps(audio, timestamps, expected):
    """Request verbose output for timestamps and read the segments or words."""
    client = make_client()
    transcriber = SpeechTranscriber(language='en', timestamps=timestamps, backend=OpenAIBackend(client=client)).fit()
    assert transcriber.predict([audio])[0] == expected
    assert transcriber.transform([audio]).tolist() == ['hello world']
    call = client.audio.transcriptions.create.calls[-1]
    assert call['language'] == 'en'
    assert call['file'][0] == 'audio.wav'
    assert call.get('timestamp_granularities') == ([timestamps] if timestamps else None)


def test_transcriber_sends_vocabulary_as_prompt(audio):
    """Send the learned vocabulary as the transcription prompt."""
    client = make_client()
    transcriber = SpeechTranscriber(backend=OpenAIBackend(client=client)).fit([audio], ['We run Kubernetes over gRPC.'])
    transcriber.predict([audio])
    assert client.audio.transcriptions.create.calls[-1]['prompt'] == 'Kubernetes, gRPC'


def fine_tuning_client(statuses, model='ft:gpt-4.1-mini:skai'):
    """Return a fake client whose fine-tuning job goes through the given statuses."""
    client = make_client('positive')
    job = {'id': 'ftjob-1', 'fine_tuned_model': model, 'error': 'bad data'}
    jobs = iter([namespace(status=status, **job) for status in statuses])
    client.files = namespace(create=Recorder(lambda **kwargs: namespace(id='file-1')))
    client.fine_tuning = namespace(
        jobs=namespace(
            create=Recorder(lambda **kwargs: next(jobs)),
            retrieve=Recorder(lambda job_id: next(jobs)),
        ),
    )
    return client


def test_fine_tuning_uploads_examples_and_waits():
    """Upload the chat examples, poll the job, and send later requests to the new model."""
    client = fine_tuning_client(['queued', 'running', 'succeeded'])
    backend = OpenAIBackend(client=client, poll_interval=0)
    texts = [f'text {index}' for index in range(10)]
    classifier = TextClassifier(adaptation='finetune', backend=backend).fit(texts, ['positive', 'negative'] * 5)
    assert classifier.model_ == 'ft:gpt-4.1-mini:skai'
    name, content = client.files.create.calls[-1]['file']
    lines = [json.loads(line) for line in content.decode().splitlines()]
    assert name == 'train.jsonl'
    assert len(lines) == 10
    assert lines[0]['messages'][-1] == {'role': 'assistant', 'content': 'positive'}
    assert client.fine_tuning.jobs.create.calls[-1]['model'] == 'gpt-4.1-mini-2025-04-14'
    assert len(client.fine_tuning.jobs.retrieve.calls) == 2
    classifier.predict(['new text'])
    assert client.chat.completions.create.calls[-1]['model'] == 'ft:gpt-4.1-mini:skai'


def test_fine_tuning_reports_failed_jobs():
    """Raise `FineTuneError` with the job id and its error."""
    backend = OpenAIBackend(client=fine_tuning_client(['running', 'failed']), poll_interval=0)
    with pytest.raises(FineTuneError, match='ftjob-1 ended with status `failed` \\(bad data\\)'):
        TextSummarizer(adaptation='finetune', backend=backend).fit([f't {i}' for i in range(10)], ['s'] * 10)


def test_fine_tuning_times_out_with_job_id():
    """Raise `TimeoutError` naming the job when it takes too long."""
    backend = OpenAIBackend(client=fine_tuning_client(['running'] * 3), finetune_timeout=0, poll_interval=0)
    with pytest.raises(TimeoutError, match='ftjob-1 did not finish'):
        TextSummarizer(adaptation='finetune', backend=backend).fit([f't {i}' for i in range(10)], ['s'] * 10)


def test_json_prompts_use_json_mode():
    """Ask for a JSON object when the prompt needs JSON output."""
    client = make_client('{"name": "Ada"}')
    backend = OpenAIBackend(model='gpt-5-mini', client=client)
    fields = backend.extract_fields(ExtractionRequest('Ada.', (('name', 'person name'),)))
    assert fields == {'name': 'Ada'}
    assert client.chat.completions.create.calls[-1]['response_format'] == {'type': 'json_object'}


class RefusedError(Exception):
    """Stand in for the error that OpenAI raises when an account cannot fine-tune."""

    status_code = 403
    message = 'OpenAI is winding down the fine-tuning platform.'


def test_fine_tuning_reports_refused_jobs():
    """Raise `FineTuneError` with OpenAI's reason when it refuses the job."""

    def refuse(**kwargs):
        """Refuse every job."""
        raise RefusedError

    client = fine_tuning_client([])
    client.fine_tuning.jobs.create = refuse
    backend = OpenAIBackend(client=client, poll_interval=0)
    with pytest.raises(FineTuneError, match='OpenAI refused the fine-tuning job: OpenAI is winding down'):
        TextSummarizer(adaptation='finetune', backend=backend).fit([f't {i}' for i in range(10)], ['s'] * 10)


def test_fine_tuning_keeps_other_errors():
    """Let errors that are not refusals propagate unchanged."""

    def fail(**kwargs):
        """Fail like a network error."""
        msg = 'connection reset'
        raise ConnectionError(msg)

    client = fine_tuning_client([])
    client.files.create = fail
    with pytest.raises(ConnectionError, match='connection reset'):
        TextSummarizer(adaptation='finetune', backend=OpenAIBackend(client=client)).fit(['t'] * 10, ['s'] * 10)
