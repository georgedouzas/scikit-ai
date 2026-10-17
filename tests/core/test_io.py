"""Test reading samples and writing generated media."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skai.audio import SpeechTranscriber, TalkingAvatarGenerator
from skai.core import (
    Audio,
    Backend,
    InvalidSampleError,
    MediaReadError,
    SampleFailureWarning,
    Video,
    read_audio,
    read_image,
    read_video,
)
from skai.image import ImageDescriptor
from skai.text import ImageGenerator, SpeechSynthesizer, TextSummarizer, TextTranslator

TEXTS = ['one two three', 'four five six', 'seven eight nine']


@pytest.mark.parametrize(
    'X',
    [
        TEXTS,
        tuple(TEXTS),
        np.array(TEXTS),
        np.array(TEXTS).reshape(-1, 1),
        pd.Series(TEXTS),
        pd.DataFrame({'text': TEXTS}),
    ],
)
def test_predict_accepts_sample_containers(X):
    """Accept lists, tuples, arrays, Series and single-column inputs."""
    summaries = TextSummarizer(max_length=1, backend='dummy').fit().predict(X)
    assert summaries.tolist() == ['one', 'four', 'seven']


def test_predict_rejects_multi_column_input():
    """Reject two-dimensional inputs with more than one column."""
    summarizer = TextSummarizer(backend='dummy').fit()
    with pytest.raises(ValueError, match='one column of samples'):
        summarizer.predict(pd.DataFrame({'a': TEXTS, 'b': TEXTS}))


def test_predict_rejects_non_string_text():
    """Name the index of the first sample that is not a string."""
    summarizer = TextSummarizer(backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match='Sample 1 must be a string'):
        summarizer.predict(['ok', 3])
    assert issubclass(InvalidSampleError, TypeError)


def test_predict_returns_text_object_array():
    """Return text outputs as a one-dimensional object array of strings."""
    translations = TextTranslator(target_language='French', backend='dummy').fit().predict([*TEXTS, 'ten'])
    assert translations.shape == (4,)
    assert translations.dtype == object
    assert all(isinstance(text, str) for text in translations)


def test_read_image_accepts_every_form(server, image, png_bytes):
    """Read images from a path, a URL, bytes, a PIL image and arrays."""
    directory, url = server
    (directory / 'image.png').write_bytes(png_bytes)
    samples = [
        directory / 'image.png',
        str(directory / 'image.png'),
        f'{url}/image.png',
        png_bytes,
        image,
        np.asarray(image),
        np.zeros((8, 8), dtype=np.uint8),
    ]
    sizes = [read_image(sample).size for sample in samples]
    assert sizes == [(32, 24)] * 6 + [(8, 8)]


def test_predict_accepts_mixed_image_forms(server, image, png_bytes):
    """Describe a path, a URL and a PIL image in one call."""
    directory, url = server
    (directory / 'image.png').write_bytes(png_bytes)
    descriptions = ImageDescriptor(backend='dummy').fit().predict([directory / 'image.png', f'{url}/image.png', image])
    assert len(descriptions) == 3


def test_read_image_reports_missing_file(tmp_path):
    """Name the index and the path of a missing file."""
    path = tmp_path / 'missing.png'
    with pytest.raises(MediaReadError, match=r'Sample 3: cannot read file .*missing\.png'):
        read_image(path, 3)
    assert issubclass(MediaReadError, ValueError)


def test_read_image_reports_undecodable_bytes():
    """Report bytes that are not an image."""
    with pytest.raises(MediaReadError, match='cannot decode the image'):
        read_image(b'not an image')


def test_read_image_reports_bad_array():
    """Report arrays that cannot be converted to an image."""
    with pytest.raises(MediaReadError, match='cannot convert an array'):
        read_image(np.zeros((2, 2, 7), dtype=np.float64))


def test_read_image_reports_unsupported_type():
    """Report samples of an unsupported type."""
    with pytest.raises(MediaReadError, match='cannot read an image from int'):
        read_image(5)


def test_read_image_reports_failed_download(server):
    """Report URLs that cannot be downloaded."""
    _, url = server
    with pytest.raises(MediaReadError, match='cannot download'):
        read_image(f'{url}/missing.png')


def test_read_audio_accepts_every_form(tmp_path, audio):
    """Read audio from a tuple, an `Audio` object, bytes and a path."""
    path = audio.save(tmp_path / 'audio.wav')
    samples = [(audio.samples, audio.sample_rate), audio, audio.to_bytes(), path]
    assert [round(read_audio(sample).duration, 2) for sample in samples] == [1.0] * 4


def test_read_audio_reports_bad_input():
    """Report audio that cannot be decoded or has an unsupported type."""
    with pytest.raises(MediaReadError, match='cannot decode the audio'):
        read_audio(b'not audio')
    with pytest.raises(MediaReadError, match='cannot read audio from list'):
        read_audio([1, 2])


def test_predict_accepts_array_audio(audio):
    """Transcribe audio given as a samples and sample rate tuple."""
    transcripts = SpeechTranscriber(backend='dummy').fit().predict([(audio.samples, 16000)])
    assert transcripts.tolist() == ['Speech of 1.00 seconds.']


def test_read_video_accepts_every_form(tmp_path):
    """Read videos from a path, bytes and a `Video` object."""
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'mp4 bytes')
    video = Video(np.zeros((2, 4, 4, 3), dtype=np.uint8), fps=2)
    assert read_video(path).to_bytes() == b'mp4 bytes'
    assert read_video(b'raw').to_bytes() == b'raw'
    assert read_video(video) is video
    with pytest.raises(MediaReadError, match='cannot read a video from float'):
        read_video(1.5)


def test_predict_ignores_unreadable_sample_when_asked(tmp_path, image):
    """Return `None` for a sample that cannot be read when `on_error='ignore'`."""
    descriptor = ImageDescriptor(backend='dummy', on_error='ignore').fit()
    with pytest.warns(UserWarning, match='sample 1'):
        descriptions = descriptor.predict([image, tmp_path / 'missing.png'])
    assert descriptions[0] is not None
    assert descriptions[1] is None


def test_predict_raises_on_unreadable_sample(tmp_path):
    """Raise the read error naming the index when `on_error='raise'`."""
    with pytest.raises(MediaReadError, match='Sample 0'):
        ImageDescriptor(backend='dummy').fit().predict([tmp_path / 'missing.png'])


def test_predict_returns_image_objects():
    """Return PIL images for `output_format='object'`."""
    images = ImageGenerator(size=(16, 16), backend='dummy').fit().predict(['a cat'])
    assert isinstance(images[0], Image.Image)


def test_predict_returns_encoded_bytes():
    """Return PNG bytes for images and WAV bytes for audio for `output_format='bytes'`."""
    png = ImageGenerator(size=(16, 16), output_format='bytes', backend='dummy').fit().predict(['a cat'])[0]
    wav = SpeechSynthesizer(output_format='bytes', backend='dummy').fit().predict(['hi'])[0]
    assert read_image(png).size == (16, 16)
    assert wav[:4] == b'RIFF'
    assert isinstance(Audio.from_bytes(wav), Audio)


def test_predict_writes_new_files_in_order(tmp_path, audio, image):
    """Write one new file per output under `output_dir` and return the paths in input order."""
    generator = TalkingAvatarGenerator(avatar=image, output_format='path', output_dir=tmp_path / 'out', backend='dummy')
    short = Audio(audio.samples[:8000], audio.sample_rate)
    paths = generator.fit().predict([audio, short])
    assert all(Path(path).parent == tmp_path / 'out' for path in paths)
    assert len(set(paths)) == 2
    durations = [Video(data=Path(path).read_bytes()).duration for path in paths]
    assert durations == pytest.approx([1.0, 0.52], abs=0.05)
    again = generator.predict([audio])
    assert again[0] not in paths


def test_predict_writes_to_temporary_directory():
    """Write files to a new temporary directory when `output_dir` is `None`."""
    paths = ImageGenerator(size=(8, 8), output_format='path', backend='dummy').fit().predict(['a', 'b'])
    assert all(Path(path).exists() for path in paths)
    assert Path(paths[0]).parent == Path(paths[1]).parent


def test_predict_keeps_missing_outputs_when_writing(tmp_path):
    """Keep `None` for failed samples in every output format."""

    class FailingBackend(Backend):
        """Fail on prompts that ask for failure."""

        def generate_image(self, request):
            """Fail or return a small image."""
            if request.prompt == 'fail':
                msg = 'boom'
                raise RuntimeError(msg)
            return Image.new('RGB', request.size)

    for output_format in ['object', 'bytes', 'path']:
        generator = ImageGenerator(
            size=(4, 4),
            output_format=output_format,
            output_dir=tmp_path,
            backend=FailingBackend(),
            on_error='ignore',
        )
        with pytest.warns(SampleFailureWarning, match='sample 1'):
            outputs = generator.fit().predict(['ok', 'fail'])
        assert outputs[1] is None
        assert outputs[0] is not None


def test_audio_round_trips_between_components():
    """Feed the in-memory audio of the synthesizer to the transcriber."""
    speech = SpeechSynthesizer(backend='dummy').fit().predict(['hello there'])
    transcripts = SpeechTranscriber(backend='dummy').fit().predict(speech)
    assert transcripts[0].startswith('Speech of')
