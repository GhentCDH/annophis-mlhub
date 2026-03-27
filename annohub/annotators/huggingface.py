"""RemoteAnnotator that translates HuggingFace Inference API responses.

The HF Inference API has its own schema — this annotator demonstrates
translating a foreign API into annohub's internal model.

Usage in annohub.toml:

    [[annotator]]
    name = "hf-ner"
    annotation_type = "ner"
    class_path = "annohub.annotators.huggingface.HuggingFaceAnnotator"
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

import asyncio
import logging
from contextlib import asynccontextmanager

from annohub.annotators.remote import RemoteAnnotator
from annohub.models import AnnotationResult, AnnotatorInfo, Document, Span, WsInputUnit, WsOutputUnit

logger = logging.getLogger(__name__)


class _HfSession:
    """Streaming session that bridges the WsSession interface over HTTP to the HF API."""

    def __init__(self, annotator: "HuggingFaceAnnotator"):
        self._annotator = annotator
        self._queue: asyncio.Queue[WsOutputUnit | None] = asyncio.Queue()
        self._tasks: set[asyncio.Task] = set()

    async def send(self, unit: WsInputUnit) -> None:
        task = asyncio.create_task(self._process(unit))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process(self, unit: WsInputUnit) -> None:
        result = await self._annotator.annotate(unit.document)
        await self._queue.put(
            WsOutputUnit(
                id=unit.id,
                annotator=result.annotator,
                annotation_type=result.annotation_type,
                spans=result.spans,
            )
        )

    async def __aiter__(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def _close(self):
        if self._tasks:
            await asyncio.gather(*self._tasks)
        await self._queue.put(None)


class HuggingFaceAnnotator(RemoteAnnotator):
    """Annotator backed by a HuggingFace Inference API NER model.

    Translates HF's response format::

        [{"entity_group": "PER", "word": "Alice", "start": 0, "end": 5, "score": 0.99}]

    into annohub Spans::

        [{"label": "PER", "text": "Alice", "start": 0, "end": 5}]
    """

    description: str = "HuggingFace Inference API NER model."
    labels: list[str] = []
    base_url: str = "https://router.huggingface.co"
    model: str = "dslim/bert-base-NER"
    token: str = ""

    def __init__(self, model: str | None = None, token: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        if token is not None:
            self.token = token

    async def annotate(self, doc: Document) -> AnnotationResult:
        client = await self.get_client()
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = await client.post(
            f"/models/{self.model}",
            json={"inputs": doc.text},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        entities = resp.json()

        spans = [
            Span(
                start=e["start"],
                end=e["end"],
                label=e["entity_group"],
                text=doc.text[e["start"] : e["end"]],
            )
            for e in entities
        ]

        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=spans,
        )

    @asynccontextmanager
    async def stream_session(self):
        session = _HfSession(self)
        try:
            yield session
        finally:
            await session._close()

    async def info(self) -> AnnotatorInfo:
        return AnnotatorInfo(
            name=self.name,
            annotation_type=self.annotation_type,
            kind="remote",
            description=self.description,
            labels=self.labels,
            contract=self.contract,
        )
