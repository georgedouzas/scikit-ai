"""Test the Hugging Face backend against fake `transformers`, `diffusers` and `sentence_transformers` modules."""

import contextlib

import numpy as np
import pytest
from PIL import Image

from skai.backends import HFBackend
from skai.classification import AudioClassifier, ImageClassifier, TextClassifier
from skai.core import (
    Audio,
    AudioOperationRequest,
    DiarizationRequest,
    MissingCredentialError,
    RegionRequest,
    TranscriptionRequest,
)
from skai.embedding import AudioEmbedder, ImageEmbedder, TextEmbedder
from skai.generation import (
    AudioToAudioGenerator,
    AudioToTextGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
    TextToVideoGenerator,
)
from skai.segmentation import AudioSegmenter, ImageSegmenter
from tests.fakes import Recorder, install_module, namespace

RMBG_REVISION = '2ceba5a5efaec153162aedea169f76caf9b46cf8'
CREDENTIAL_VARIABLE = 'SKAI_TEST_HF_TOKEN'
FAKE_CREDENTIAL = 'not-a-secret'
STEM_LEVELS = {'drums': 1.0, 'bass': 2.0, 'other': 3.0, 'vocals': 4.0}


def zero_shot(text, candidate_labels, multi_label):
    """Score the candidates in reverse order of appearance."""
    scores = np.linspace(0.6, 0.1, len(candidate_labels))
    return {'labels': list(reversed(candidate_labels)), 'scores': scores.tolist()}


PIPELINES = {
    'text-generation': lambda messages, **kwargs: [{'generated_text': 'Short summary.'}],
    'image-text-to-text': lambda text, **kwargs: [{'generated_text': 'A small image.'}],
    'zero-shot-classification': zero_shot,
    'zero-shot-image-classification': lambda image, candidate_labels: [
        {'label': label, 'score': score}
        for label, score in zip(candidate_labels, np.linspace(0.2, 0.8, len(candidate_labels)), strict=True)
    ],
    'text-to-speech': lambda text, **kwargs: {'audio': np.zeros((1, 8000)), 'sampling_rate': 16000},
    'automatic-speech-recognition': lambda inputs, **kwargs: {
        'text': ' hi there',
        'chunks': [{'timestamp': (0.0, 0.5), 'text': ' hi'}, {'timestamp': (0.5, None), 'text': ' there'}],
    },
    'zero-shot-object-detection': lambda image, candidate_labels, **kwargs: [
        {'label': label, 'score': 0.9, 'box': {'xmin': -3.4, 'ymin': 2.6, 'xmax': 50.2, 'ymax': 20.0}}
        for label in candidate_labels
    ],
    'image-to-image': lambda image, **kwargs: image.resize((image.width * 2, image.height * 2)),
    'image-segmentation': lambda image, return_mask=False, **kwargs: (
        Image.new('L', (4, 4), 200) if return_mask else [{'mask': Image.new('L', image.size, 90)}]
    ),
    'text-to-audio': lambda text, **kwargs: {'audio': np.ones((1, 1, 40000)), 'sampling_rate': 32000},
}


@pytest.fixture
def transformers(monkeypatch):
    """Install a fake `transformers` module that records every pipeline it creates."""
    calls = {}
    options = {}
    pipelines = {}

    def pipeline(task, model, device, **kwargs):
        """Create a recorded fake pipeline, keeping the extra arguments it was created with."""
        calls[task, model] = calls.get((task, model), 0) + 1
        options[task, model] = kwargs
        recorder = Recorder(PIPELINES[task])
        recorder.tokenizer = namespace(get_prompt_ids=lambda text, return_tensors: ('prompt', text))
        pipelines[task] = recorder
        return recorder

    module = install_module(monkeypatch, 'transformers', pipeline=pipeline)
    module.created = calls
    module.options = options
    module.pipelines = pipelines
    return module


@pytest.fixture
def torch(monkeypatch):
    """Install a fake `torch` module whose tensors are NumPy arrays."""
    return install_module(
        monkeypatch,
        'torch',
        no_grad=contextlib.nullcontext,
        from_numpy=lambda array: namespace(to=lambda device: array, array=array),
        device=lambda name: name,
        Generator=lambda: namespace(manual_seed=lambda seed: ('generator', seed)),
    )


def test_prompted_tasks_use_chat_and_vision_models(transformers):
    """Prompt the chat model for text and the vision model for images, caching each pipeline."""
    summarizer = TextToTextGenerator(instructions='Summarize the text.', backend='hf').fit()
    assert summarizer.predict(['a', 'b']).tolist() == ['Short summary.'] * 2
    assert summarizer.predict(['c']).tolist() == ['Short summary.']
    assert transformers.created[('text-generation', HFBackend.default_models['generate_text'])] == 1
    translator = TextToTextGenerator(instructions='Translate into French.', backend='hf').fit()
    assert translator.predict(['a'])[0] == 'Short summary.'
    descriptor = ImageToTextGenerator(instructions='Describe the image.', backend='hf').fit()
    assert descriptor.predict([Image.new('RGB', (4, 4))])[0] == 'A small image.'
    assert ('image-text-to-text', HFBackend.default_models['generate_text:image']) in transformers.created


def test_audio_text_generation_uses_audio_language_model(transformers, audio):
    """Prompt the audio-language model with the audio at its sampling rate and decode the new tokens."""
    calls = {}

    class FakeProcessor:
        """Render the chat, encode the inputs and decode the output."""

        feature_extractor = namespace(sampling_rate=8000)

        def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
            """Keep the conversation and return the prompt text."""
            calls['conversation'] = conversation
            return 'prompt'

        def __call__(self, text, audio, sampling_rate, return_tensors):
            """Keep the audio and return three input ids."""
            calls['audio'] = audio
            return {'input_ids': np.zeros((1, 3))}

        def batch_decode(self, generated, skip_special_tokens):
            """Decode only the generated tokens."""
            calls['generated'] = generated.shape
            return [' Birds singing. ']

    transformers.AutoProcessor = namespace(from_pretrained=lambda model: FakeProcessor())
    transformers.Qwen2AudioForConditionalGeneration = namespace(
        from_pretrained=lambda model, device_map: namespace(generate=lambda **kwargs: np.zeros((1, 5))),
    )
    generator = AudioToTextGenerator(operation='instruct', instructions='Describe.', max_length=4, backend='hf')
    assert generator.fit().predict([audio]).tolist() == ['Birds singing.']
    assert calls['audio'][0].shape == (8000,)
    assert calls['generated'] == (1, 2)
    text = calls['conversation'][0]['content'][1]['text']
    assert 'Instructions: Describe.\nUse at most 4 words.' in text
    assert calls['conversation'][0]['content'][0]['type'] == 'audio'
    assert all(HFBackend.supports(task, 'audio') for task in ('generate_text', 'extract_fields', 'find_moments'))


def test_text_classifier_uses_zero_shot_scores(transformers):
    """Return the zero-shot scores, matching descriptions back to labels."""
    classifier = TextClassifier(labels={'pos': 'positive', 'neg': 'negative'}, backend='hf').fit()
    proba = classifier.predict_proba(['great'])
    np.testing.assert_allclose(proba, [[0.1 / 0.7, 0.6 / 0.7]])
    assert classifier.predict(['great']).tolist() == ['pos']


def test_image_classifier_uses_clip_scores(transformers):
    """Return the zero-shot image scores."""
    classifier = ImageClassifier(labels=['cat', 'dog'], backend='hf').fit()
    np.testing.assert_allclose(classifier.predict_proba([Image.new('RGB', (4, 4))]), [[0.2, 0.8]])


def test_embedder_uses_sentence_transformers(transformers, monkeypatch):
    """Encode texts with a cached sentence-transformers model on the chosen device."""
    created = []

    def sentence_transformer(model, device):
        """Create a fake encoder."""
        created.append((model, device))
        return namespace(encode=lambda texts, **kwargs: np.ones((len(texts), 5)))

    install_module(monkeypatch, 'sentence_transformers', SentenceTransformer=sentence_transformer)
    embedder = TextEmbedder(dimensions=3, backend=HFBackend(device='cpu')).fit()
    np.testing.assert_allclose(embedder.transform(['a', 'b']), np.ones((2, 3)))
    assert created == [('sentence-transformers/all-MiniLM-L6-v2', 'cpu')]


def test_synthesizer_returns_mono_audio(transformers):
    """Squeeze the pipeline audio into mono samples."""
    speech = TextToAudioGenerator(backend='hf').fit().predict(['hello'])[0]
    assert speech.samples.shape == (8000,)
    assert speech.sample_rate == 16000


def test_image_generator_seeds_diffusion(transformers, monkeypatch):
    """Seed the generator and use one-step settings for the default turbo model."""
    calls = []

    class FakePipeline:
        """Record the generation arguments."""

        def to(self, device):
            """Return the pipeline on a device."""
            self.device = device
            return self

        def __call__(self, **kwargs):
            """Return an image of the requested size."""
            calls.append(kwargs)
            return namespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']))])

    install_module(
        monkeypatch,
        'diffusers',
        AutoPipelineForText2Image=namespace(from_pretrained=lambda model: FakePipeline()),
    )
    install_module(monkeypatch, 'torch', Generator=lambda: namespace(manual_seed=lambda seed: ('generator', seed)))
    generator = TextToImageGenerator(size=(12, 8), negative_prompt='blur', random_state=0, backend='hf').fit()
    assert generator.predict(['a cat'])[0].size == (12, 8)
    assert calls[-1]['num_inference_steps'] == 1
    assert calls[-1]['negative_prompt'] == 'blur'
    assert calls[-1]['generator'][0] == 'generator'
    TextToImageGenerator(size=(8, 8), backend='hf:other/model').fit().predict(['a dog'])
    assert 'num_inference_steps' not in calls[-1]


def test_transcriber_reads_chunks(transformers, audio):
    """Read the text and the timed chunks, ending an open chunk at the audio end."""
    transcript = HFBackend().transcribe(TranscriptionRequest(audio, language='en', timestamps='word'))
    assert [(segment.start, segment.end, segment.text) for segment in transcript.segments] == [
        (0.0, 0.5, 'hi'),
        (0.5, 1.0, 'there'),
    ]
    call = transformers.pipelines['automatic-speech-recognition'].calls[-1]
    assert call['return_timestamps'] == 'word'
    assert call['generate_kwargs'] == {'language': 'en'}
    assert AudioToTextGenerator(backend='hf').fit().predict([audio]).tolist() == ['hi there']


def test_transcriber_sends_vocabulary_as_prompt_ids(transformers, audio):
    """Send the learned vocabulary as Whisper prompt ids."""
    AudioToTextGenerator(backend='hf').fit([audio], ['We run Kubernetes.']).predict([audio])
    call = transformers.pipelines['automatic-speech-recognition'].calls[-1]
    assert call['generate_kwargs']['prompt_ids'] == ('prompt', 'Kubernetes')


def test_audio_classification_uses_clap(transformers, monkeypatch, audio):
    """Score audio resampled to 48 kHz with the zero-shot audio pipeline."""
    monkeypatch.setitem(
        PIPELINES,
        'zero-shot-audio-classification',
        lambda samples, candidate_labels: [
            {'label': label, 'score': score} for label, score in zip(candidate_labels, [0.3, 0.7], strict=True)
        ],
    )
    classifier = AudioClassifier(labels=['rain', 'wind'], backend='hf').fit()
    np.testing.assert_allclose(classifier.predict_proba([audio]), [[0.3, 0.7]])
    samples = transformers.pipelines['zero-shot-audio-classification'].calls[-1]['args'][0]
    assert samples.shape == (48000,)


def test_image_and_audio_embeddings(transformers, monkeypatch, audio):
    """Embed images with CLIP through sentence-transformers and audio with CLAP."""
    install_module(
        monkeypatch,
        'sentence_transformers',
        SentenceTransformer=lambda model, device: namespace(encode=lambda items, **kwargs: np.ones((len(items), 6))),
    )
    assert ImageEmbedder(backend='hf').fit().transform([Image.new('RGB', (4, 4))]).shape == (1, 6)
    features = namespace(detach=lambda: namespace(cpu=lambda: namespace(numpy=lambda: np.ones(5))))
    transformers.ClapProcessor = namespace(from_pretrained=lambda model: lambda **kwargs: {'input_features': 1})
    transformers.ClapModel = namespace(
        from_pretrained=lambda model: namespace(get_audio_features=lambda **kwargs: [features]),
    )
    assert AudioEmbedder(backend='hf').fit().transform([audio]).shape == (1, 5)


def test_video_generation_edit_and_animation(transformers, monkeypatch):
    """Generate, edit and animate with diffusion pipelines, returning frames of the requested sizes."""

    class FakeDiffusion:
        """Return frames or images shaped by the call."""

        def to(self, device):
            """Return the pipeline on a device."""
            return self

        def __call__(self, *args, **kwargs):
            """Return float frames for video calls and an image for edit calls."""
            if 'num_frames' in kwargs:
                height, width = kwargs.get('height', 576), kwargs.get('width', 1024)
                return namespace(frames=[[np.full((height, width, 3), 0.5)] * kwargs['num_frames']])
            return namespace(images=[Image.new('RGB', (64, 64))])

    install_module(
        monkeypatch,
        'diffusers',
        DiffusionPipeline=namespace(from_pretrained=lambda model: FakeDiffusion()),
        StableDiffusionInstructPix2PixPipeline=namespace(from_pretrained=lambda model: FakeDiffusion()),
        StableVideoDiffusionPipeline=namespace(from_pretrained=lambda model: FakeDiffusion()),
    )
    install_module(monkeypatch, 'torch', Generator=lambda: namespace(manual_seed=lambda seed: seed))
    video = TextToVideoGenerator(duration=1, size=(32, 16), backend='hf').fit().predict(['a boat'])[0]
    assert video.frames.shape == (17, 16, 32, 3)
    assert video.frames.dtype == np.uint8
    editor = ImageToImageGenerator(operation='edit', instructions='snow', backend='hf').fit()
    assert editor.predict([Image.new('RGB', (8, 6))])[0].size == (8, 6)
    animation = ImageToVideoGenerator(duration=1, backend='hf').fit().predict([Image.new('RGB', (12, 8))])[0]
    assert animation.frames.shape[1:3] == (8, 12)


def test_detect_objects_uses_owl_and_clamps_boxes(transformers):
    """Detect the labels with OWLv2, or the prompt without labels, and clamp the boxes to the image."""
    segmenter = ImageSegmenter(labels=['cat', 'dog'], backend='hf').fit()
    regions = segmenter.predict([Image.new('RGB', (40, 30))])[0]
    assert [(region['label'], region['box']) for region in regions] == [
        ('cat', (0, 3, 40, 20)),
        ('dog', (0, 3, 40, 20)),
    ]
    assert ('zero-shot-object-detection', 'google/owlv2-base-patch16-ensemble') in transformers.created
    HFBackend().detect_objects(RegionRequest(Image.new('RGB', (8, 8)), prompt='a red car'))
    assert transformers.pipelines['zero-shot-object-detection'].calls[-1]['candidate_labels'] == ['a red car']


def test_segment_image_prompts_sam_with_detected_boxes(transformers, torch):
    """Prompt SAM with the OWLv2 boxes and return a grayscale mask of the image size per region."""
    calls = {}

    class FakeInputs(dict):
        """Hold the processor outputs and move them to a device."""

        def to(self, device):
            """Return the inputs on a device."""
            calls['device'] = device
            return self

    def process(image, input_boxes, return_tensors):
        """Keep the boxes and return the processed inputs."""
        calls['boxes'] = input_boxes
        return FakeInputs(original_sizes=[[30, 40]], reshaped_input_sizes=[[768, 1024]])

    def post_process_masks(masks, original_sizes, reshaped_input_sizes):
        """Return one boolean mask per box at the original size."""
        mask = np.zeros((1, 30, 40), dtype=bool)
        mask[0, 5:10, 5:20] = True
        return [np.stack([mask] * masks.shape[0])]

    processor = Recorder(process)
    processor.image_processor = namespace(post_process_masks=post_process_masks)
    sam = namespace(to=lambda device: lambda **kwargs: namespace(pred_masks=namespace(cpu=lambda: np.zeros((2, 1)))))
    transformers.SamProcessor = namespace(from_pretrained=Recorder(lambda model: processor))
    transformers.SamModel = namespace(from_pretrained=Recorder(lambda model: sam))
    regions = ImageSegmenter(by='mask', labels=['cat', 'dog'], backend='hf').fit().predict([Image.new('RGB', (40, 30))])
    masks = [region['mask'] for region in regions[0]]
    assert [(mask.mode, mask.size) for mask in masks] == [('L', (40, 30))] * 2
    assert np.asarray(masks[0])[7, 10] == 255
    assert np.asarray(masks[0])[0, 0] == 0
    assert calls['boxes'] == [[[0, 3, 40, 20], [0, 3, 40, 20]]]
    assert transformers.SamModel.from_pretrained.calls[-1]['args'] == ('facebook/sam-vit-base',)


@pytest.fixture
def pyannote(monkeypatch):
    """Install a fake `pyannote.audio` module whose pipeline finds two speaker turns."""
    turns = [(namespace(start=0.0, end=0.5), 'A', 'SPEAKER_00'), (namespace(start=0.5, end=1.0), 'B', 'SPEAKER_01')]
    annotation = namespace(itertracks=lambda yield_label: iter(turns))
    diarizer = Recorder(lambda inputs: annotation)
    diarizer.to = Recorder(lambda device: diarizer)
    module = install_module(
        monkeypatch,
        'pyannote.audio',
        Pipeline=namespace(from_pretrained=Recorder(lambda model, use_auth_token: diarizer)),
    )
    module.diarizer = diarizer
    return module


def test_diarize_reads_token_and_transcribes_turns(transformers, torch, pyannote, monkeypatch, audio):
    """Load pyannote with the token of `token_env`, and transcribe every speaker turn with Whisper."""
    monkeypatch.setenv(CREDENTIAL_VARIABLE, FAKE_CREDENTIAL)
    backend = HFBackend('pyannote/other-diarization', device='cpu', token_env=CREDENTIAL_VARIABLE)
    regions = AudioSegmenter(by='speaker', backend=backend).fit().predict([audio])[0]
    assert [(region['speaker'], region['start'], region['end'], region['text']) for region in regions] == [
        ('SPEAKER_00', 0.0, 0.5, 'hi there'),
        ('SPEAKER_01', 0.5, 1.0, 'hi there'),
    ]
    assert pyannote.Pipeline.from_pretrained.calls[-1] == {
        'args': ('pyannote/other-diarization',),
        'use_auth_token': FAKE_CREDENTIAL,
    }
    assert pyannote.diarizer.to.calls[-1]['args'] == ('cpu',)
    assert pyannote.diarizer.calls[-1]['args'][0]['sample_rate'] == 16000
    asr = transformers.pipelines['automatic-speech-recognition']
    assert [call['args'][0]['raw'].shape for call in asr.calls] == [(8000,), (8000,)]
    assert ('automatic-speech-recognition', 'openai/whisper-small') in transformers.created


def test_diarize_without_token_raises(pyannote, monkeypatch, audio):
    """Raise a credential error when the token variable is not set."""
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(MissingCredentialError, match='HF_TOKEN'):
        HFBackend().diarize(DiarizationRequest(audio))


class FakeSeparator:
    """Return every stem at a constant level and keep the waveform."""

    sources = ('drums', 'bass', 'other', 'vocals')

    def to(self, device):
        """Return the model on a device."""
        return self

    def __call__(self, waveform):
        """Return the four stems as stereo constants of the waveform length."""
        self.waveform = waveform
        stems = np.stack([np.full((2, waveform.shape[-1]), STEM_LEVELS[name]) for name in self.sources])
        return [namespace(cpu=lambda: namespace(numpy=lambda: stems))]


@pytest.fixture
def separator(transformers, torch, monkeypatch):
    """Install a fake `torchaudio` module whose Hybrid Demucs bundle runs at 44.1 kHz."""
    model = FakeSeparator()
    bundle = namespace(sample_rate=44100, get_model=lambda: model)
    install_module(monkeypatch, 'torchaudio', pipelines=namespace(HDEMUCS_HIGH_MUSDB_PLUS=bundle))
    return model


@pytest.mark.parametrize(('stem', 'level'), [('vocals', 4.0), ('accompaniment', 6.0)])
def test_separate_audio_uses_hdemucs_at_its_rate(separator, audio, stem, level):
    """Separate stereo audio at the bundle rate and return the mono stem at the input length and rate."""
    result = AudioToAudioGenerator(operation='separate', stem=stem, backend='hf').fit().predict([audio])[0]
    assert (result.samples.shape, result.sample_rate) == ((16000,), 16000)
    assert separator.waveform.shape == (1, 2, 44100)
    np.testing.assert_allclose(result.samples[100:-100], level * audio.samples.std(), rtol=1e-2)


def test_enhance_audio_reduces_noise_and_keeps_the_tone(transformers):
    """Lower the power of the noise and keep a tone, at the input length and rate."""
    rate = 16000
    times = np.arange(2 * rate) / rate
    noise = 0.05 * np.random.default_rng(0).standard_normal(times.size)
    tone = np.where(times >= 1, 0.5 * np.sin(2 * np.pi * 1000 * times), 0.0)
    denoiser = AudioToAudioGenerator(operation='enhance', backend='hf').fit()
    result = denoiser.predict([Audio((tone + noise).astype(np.float32), rate)])[0]
    assert (result.samples.shape, result.sample_rate) == ((2 * rate,), rate)
    quiet, loud = slice(2000, 14000), slice(18000, 30000)
    assert np.mean(result.samples[quiet] ** 2) < 0.25 * np.mean(noise[quiet] ** 2)
    assert np.mean((result.samples[loud] - tone[loud]) ** 2) < 0.5 * np.mean(noise[loud] ** 2)
    short = Audio(np.full(100, 0.5, dtype=np.float32), rate)
    np.testing.assert_allclose(HFBackend().enhance_audio(AudioOperationRequest(short)).samples, short.samples)


@pytest.mark.parametrize(
    ('scale', 'model'),
    [
        (2, 'caidas/swin2SR-classical-sr-x2-64'),
        (3, 'caidas/swin2SR-classical-sr-x2-64'),
        (4, 'caidas/swin2SR-classical-sr-x4-64'),
    ],
)
def test_upscale_image_uses_swin2sr_and_exact_size(transformers, scale, model):
    """Upscale with the Swin2SR model of the scale, or x2 and a resize, to exactly `scale` times the size."""
    upscaler = ImageToImageGenerator(operation='upscale', scale=scale, backend='hf').fit()
    assert upscaler.predict([Image.new('RGB', (10, 6))])[0].size == (10 * scale, 6 * scale)
    assert ('image-to-image', model) in transformers.created


def test_upscale_image_by_one_returns_the_image(transformers):
    """Return the image itself at scale one, without loading a model."""
    upscaler = ImageToImageGenerator(operation='upscale', scale=1, backend='hf').fit()
    assert upscaler.predict([Image.new('RGB', (10, 6))])[0].size == (10, 6)
    assert not transformers.created


def test_segment_image_without_detections_returns_no_regions(transformers, monkeypatch):
    """Return no regions, without loading SAM, when nothing is detected."""
    monkeypatch.setitem(PIPELINES, 'zero-shot-object-detection', lambda image, candidate_labels: [])
    assert ImageSegmenter(by='mask', backend='hf').fit().predict([Image.new('RGB', (8, 8))])[0] == []


def test_remove_background_pins_rmbg_revision(transformers):
    """Load RMBG with remote code at the pinned revision and use its mask as the alpha of the input size."""
    remover = ImageToImageGenerator(operation='remove_background', backend='hf').fit()
    cutout = remover.predict([Image.new('RGB', (8, 6), 'red')])[0]
    assert (cutout.mode, cutout.size) == ('RGBA', (8, 6))
    assert cutout.getpixel((3, 3)) == (255, 0, 0, 200)
    assert transformers.options['image-segmentation', 'briaai/RMBG-1.4'] == {
        'trust_remote_code': True,
        'revision': RMBG_REVISION,
    }
    other = ImageToImageGenerator(operation='remove_background', backend='hf:other/segmenter').fit()
    assert other.predict([Image.new('RGB', (8, 6))])[0].getpixel((0, 0))[3] == 90
    assert transformers.options['image-segmentation', 'other/segmenter'] == {}


def test_generate_music_uses_musicgen_tokens_for_duration(transformers):
    """Ask MusicGen for 50 tokens per second, seed it, and pad the audio to the duration."""
    transformers.set_seed = Recorder(lambda seed: None)
    generator = TextToAudioGenerator(operation='music', duration=2, random_state=0, backend='hf').fit()
    music = generator.predict(['calm piano'])[0]
    assert (music.samples.shape, music.sample_rate) == ((64000,), 32000)
    assert music.samples[-1] == 0.0
    call = transformers.pipelines['text-to-audio'].calls[-1]
    assert call['forward_params'] == {'do_sample': True, 'max_new_tokens': 100}
    assert len(transformers.set_seed.calls) == 1
    assert ('text-to-audio', 'facebook/musicgen-small') in transformers.created


def test_generate_sound_uses_audioldm2(transformers, torch, monkeypatch):
    """Generate with AudioLDM2 for the duration with a seeded generator, trimmed to the duration."""
    calls = []
    created = []

    def pipeline(prompt, **kwargs):
        """Return audio longer than requested."""
        calls.append(kwargs)
        return namespace(audios=np.ones((1, 50000)))

    def from_pretrained(model):
        """Create the fake pipeline."""
        created.append(model)
        return namespace(to=lambda device: pipeline)

    install_module(monkeypatch, 'diffusers', AudioLDM2Pipeline=namespace(from_pretrained=from_pretrained))
    sound = TextToAudioGenerator(operation='sound', duration=3, random_state=0, backend='hf').fit().predict(['rain'])[0]
    assert (sound.samples.shape, sound.sample_rate) == ((48000,), 16000)
    assert calls[-1]['audio_length_in_s'] == 3.0
    assert calls[-1]['generator'][0] == 'generator'
    assert created == ['cvssp/audioldm2']


def test_new_tasks_are_supported():
    """Declare the new tasks with a default model, except enhancement, which needs none."""
    tasks = {'detect_objects', 'segment_image', 'diarize', 'separate_audio', 'upscale_image', 'remove_background'}
    tasks |= {'generate_music', 'generate_sound'}
    assert tasks | {'enhance_audio'} <= HFBackend.supported_tasks()
    assert tasks <= HFBackend.default_models.keys()
