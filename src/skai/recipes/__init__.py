"""Return configured pipelines that turn inputs into finished results.

Every recipe is a function that returns an ordinary scikit-learn `Pipeline` built only from the public components,
the primitives of `skai.pipeline` and scikit-learn, so its steps can be inspected, changed with `set_params`, cloned
and tuned. Every recipe takes a `backend`, passed to every step that uses one, and keyword arguments for its main
choices. `video_search` returns a `Retriever`, since it answers queries.

Examples:
    >>> from skai import recipes
    >>> pipeline = recipes.study_pack(backend='dummy')
    >>> [name for name, _ in pipeline.steps]
    ['transcript', 'pack']
"""

from ._recipes import (
    audio_description,
    call_topics,
    dubbing,
    explainer_video,
    highlights,
    image_translation,
    invoice_anomalies,
    meeting_briefing,
    moderation,
    podcast,
    product_ad,
    product_catalog,
    quality_control,
    read_along,
    study_pack,
    vertical_clip,
    video_search,
)

__all__: list[str] = [
    'audio_description',
    'call_topics',
    'dubbing',
    'explainer_video',
    'highlights',
    'image_translation',
    'invoice_anomalies',
    'meeting_briefing',
    'moderation',
    'podcast',
    'product_ad',
    'product_catalog',
    'quality_control',
    'read_along',
    'study_pack',
    'vertical_clip',
    'video_search',
]
