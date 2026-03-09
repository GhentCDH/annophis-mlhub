"""RemoteAnnotator that translates HuggingFace Inference API responses.

The HF Inference API has its own schema — this annotator demonstrates
translating a foreign API into konekaare's internal model.

Usage in konekaare.toml:

    [[annotator]]
    name = "hf-ner"
    annotation_type = "ner"
    class_path = "konekaare.annotators.huggingface.HuggingFaceAnnotator"
    base_url = "https://router.huggingface.co"
    model = "dslim/bert-base-NER"
    token = "hf_..."

Or with class-level defaults:

    class MyHfAnnotator(HuggingFaceAnnotator):
        name = "hf-ner"
        annotation_type = "ner"
        model = "dslim/bert-base-NER"
        token = "hf_..."
"""

import logging

from konekaare.annotators.remote import RemoteAnnotator
from konekaare.models import AnnotationRequest, AnnotationResult, Span

logger = logging.getLogger(__name__)


class HuggingFaceAnnotator(RemoteAnnotator):
    """Annotator backed by a HuggingFace Inference API NER model.

    Translates HF's response format::

        [{"entity_group": "PER", "word": "Alice", "start": 0, "end": 5, "score": 0.99}]

    into konekaare Spans::

        [{"label": "PER", "text": "Alice", "start": 0, "end": 5}]
    """

    base_url: str = "https://router.huggingface.co"
    model: str = "dslim/bert-base-NER"
    token: str = ""

    def __init__(self, model: str | None = None, token: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        if token is not None:
            self.token = token

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        client = await self.get_client()
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = await client.post(
            f"/models/{self.model}",
            json={"inputs": request.text},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        entities = resp.json()

        # Translate HF schema → konekaare Span
        spans = [
            Span(
                start=e["start"],
                end=e["end"],
                label=e["entity_group"],
                text=request.text[e["start"] : e["end"]],
            )
            for e in entities
        ]

        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=spans,
        )
