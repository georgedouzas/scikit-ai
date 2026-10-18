"""Run the recipes whose every step the OpenAI backend supports.

The other recipes need operations that OpenAI does not offer: music (`explainer_video`, `vertical_clip`, `podcast`,
`highlights`), reference images (`read_along`), background removal (`product_catalog`), upscaling (`product_ad`),
separation and lip sync (`dubbing`), masks (`image_translation`), talking heads (`meeting_briefing`), diarization
(`call_topics`) and joint text and image embeddings (`video_search`).
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from skai import recipes
from skai.core import Audio, Video
from skai.generation import TextToAudioGenerator

LECTURE = (
    'Photosynthesis is how plants make food. Leaves capture sunlight and use it to turn water and carbon dioxide '
    'into sugar, and they release oxygen.'
)
INVOICES = [
    ('Acme Office Supplies', '120.00', '2024-03-04'),
    ('Acme Office Supplies', '115.50', '2024-03-11'),
    ('Acme Office Supplies', '124.90', '2024-03-18'),
    ('Acme Office Supplies', '118.20', '2024-03-25'),
    ('Acme Office Supplies', '121.70', '2024-04-01'),
    ('Zeta Offshore Holdings', '98000.00', '1999-12-31'),
]


def _invoice(vendor, amount, day):
    """Draw a plain invoice scan without a provider."""
    image = Image.new('RGB', (480, 240), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for row, line in enumerate(['INVOICE', f'Vendor: {vendor}', f'Date: {day}', f'Total: {amount} EUR']):
        draw.text((24, 24 + 48 * row), line, fill='black', font=font)
    return image


def test_study_pack(openai_backend, shared_images):
    """Turn a lecture video into notes, key terms, a quiz and a spoken recap."""
    speech = TextToAudioGenerator(backend=openai_backend).fit().predict([LECTURE])[0]
    frames = np.repeat(np.asarray(shared_images[0].convert('RGB').resize((64, 64)))[np.newaxis], 8, axis=0)
    lecture = Video(frames, fps=8 / speech.duration, audio=speech)
    pack = recipes.study_pack(n_questions=2, backend=openai_backend).fit(None).predict([lecture])[0]
    assert set(pack) == {'notes', 'key_terms', 'quiz', 'recap'}
    assert 'photosynthesis' in pack['notes'].lower()
    assert isinstance(pack['key_terms'], list)
    assert pack['key_terms']
    assert 'photosynthesis' in ' '.join(str(term).lower() for term in pack['key_terms'])
    assert '?' in pack['quiz']
    assert isinstance(pack['recap'], Audio)
    assert pack['recap'].duration > 1


def test_audio_description(openai_backend, scenes_video):
    """Add spoken shot descriptions and dialogue subtitles to a video, keeping its length."""
    video = recipes.audio_description(backend=openai_backend).fit(None).predict([scenes_video])[0]
    assert isinstance(video, Video)
    assert video.duration == pytest.approx(scenes_video.duration, abs=0.2)
    assert video.audio is not None
    assert 'harvest' in video.subtitles.lower()


def test_invoice_anomalies(openai_backend):
    """Flag the invoice whose vendor, amount and date differ from the others."""
    scans = [_invoice(*invoice) for invoice in INVOICES]
    pipeline = recipes.invoice_anomalies(contamination=1 / len(scans), backend=openai_backend)
    labels = pipeline.fit(scans).predict(scans)
    assert labels.shape == (len(scans),)
    assert set(labels.tolist()) <= {-1, 1}
    assert labels[-1] == -1


def test_moderation(openai_backend, shared_images):
    """Calibrate the moderation of posts of an image and a caption on labeled posts."""
    apple, car = shared_images
    posts = [
        {'image': apple, 'caption': 'Fresh apples at the farmers market this weekend!'},
        {'image': car, 'caption': 'Park here again and I will smash your windows and hurt you.'},
        {'image': car, 'caption': 'My new car, ready for a family road trip.'},
        {'image': apple, 'caption': 'Whoever lives here, I know where you sleep. You will pay.'},
    ]
    labels = ['safe', 'unsafe', 'safe', 'unsafe']
    pipeline = recipes.moderation(cv=2, backend=openai_backend).fit(posts, labels)
    proba = pipeline.predict_proba(posts)
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert set(pipeline.predict(posts).tolist()) <= {'safe', 'unsafe'}


def test_quality_control(openai_backend, shared_images):
    """Train a classifier on defect answers and regions of labeled photos."""
    labels = ['ok', 'defective']
    pipeline = recipes.quality_control(backend=openai_backend).fit(shared_images, labels)
    assert pipeline.predict(shared_images).tolist() == labels
