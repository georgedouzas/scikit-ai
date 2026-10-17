"""Shared building blocks of scikit-ai: media types, requests, backends and component bases."""

from ._backend import TASKS, Backend, PromptedBackend
from ._errors import (
    InvalidResponseError,
    InvalidSampleError,
    MediaReadError,
    MissingCredentialError,
    MissingExtraError,
    SampleError,
    SampleFailureWarning,
    UnsupportedBackendError,
)
from ._estimator import BaseClassifier, BaseEmbedder, BaseGenerator, BaseMediaEstimator, BaseMediaGenerator
from ._io import encode_png, read_audio, read_image, read_video, to_data_url, to_object_array
from ._media import Audio, Video
from ._prompts import Prompt, build_image_prompt
from ._registry import list_backends, register
from ._requests import (
    AvatarRequest,
    ClassificationRequest,
    DescriptionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    Segment,
    SpeechRequest,
    SummaryRequest,
    Transcript,
    TranscriptionRequest,
    TranslationRequest,
)
from ._utils import import_optional, read_env

__all__: list[str] = [
    'TASKS',
    'Audio',
    'AvatarRequest',
    'Backend',
    'BaseClassifier',
    'BaseEmbedder',
    'BaseGenerator',
    'BaseMediaEstimator',
    'BaseMediaGenerator',
    'ClassificationRequest',
    'DescriptionRequest',
    'EmbeddingRequest',
    'ImageGenerationRequest',
    'InvalidResponseError',
    'InvalidSampleError',
    'MediaReadError',
    'MissingCredentialError',
    'MissingExtraError',
    'Prompt',
    'PromptedBackend',
    'SampleError',
    'SampleFailureWarning',
    'Segment',
    'SpeechRequest',
    'SummaryRequest',
    'Transcript',
    'TranscriptionRequest',
    'TranslationRequest',
    'UnsupportedBackendError',
    'Video',
    'build_image_prompt',
    'encode_png',
    'import_optional',
    'list_backends',
    'read_audio',
    'read_env',
    'read_image',
    'read_video',
    'register',
    'to_data_url',
    'to_object_array',
]
