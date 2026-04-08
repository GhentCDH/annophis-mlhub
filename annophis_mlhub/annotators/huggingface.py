"""RemoteAnnotator that translates HuggingFace Inference API responses.

The HF Inference API has its own schema — this annotator demonstrates
translating a foreign API into LIF annotations.

Usage in mlhub.toml:

    [[annotator]]
    name = "hf-ner"
    annotation_type = "ner"
    class_path = "annophis_mlhub.annotators.huggingface.HuggingFaceAnnotator"
    base_url = "https://router.huggingface.co"
    model = "dslim/bert-base-NER"
    token = "hf_..."
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from annophis_mlhub.annotators.descriptors import build_descriptor_node
from annophis_mlhub.annotators.remote import RemoteAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument
from annophis_mlhub.models import WsInputUnit, WsOutputUnit

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
        annotations = await self._annotator.annotate(unit.document)
        await self._queue.put(
            WsOutputUnit(
                id=unit.id,
                annotator=self._annotator.name,
                annotations=annotations,
            )
        )

    async def __aiter__(self) -> AsyncIterator[WsOutputUnit]:
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

    into LIF annotations::

        [{"@type": "NamedEntity", "id": "ne0", "start": 0, "end": 5,
          "features": {"category": "PER", "word": "Alice"}}]
    """

    description: str = "HuggingFace Inference API NER model."
    supports_streaming = True
    base_url: str = "https://router.huggingface.co"
    model: str = "dslim/bert-base-NER"
    token: str = ""

    def __init__(self, model: str | None = None, token: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        if token is not None:
            self.token = token

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]:
        client = await self.get_client()
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        text = doc.text.value
        resp = await client.post(
            f"/models/{self.model}",
            json={"inputs": text},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        entities = resp.json()

        return [
            LIFAnnotation(
                id=f"ne{i}",
                type="NamedEntity",
                start=e["start"],
                end=e["end"],
                features={
                    "category": e["entity_group"],
                    "word": text[e["start"] : e["end"]],
                },
            )
            for i, e in enumerate(entities)
        ]

    @asynccontextmanager
    async def stream_session(self):
        session = _HfSession(self)
        try:
            yield session
        finally:
            await session._close()

    async def info(self) -> dict[str, Any]:
        return build_descriptor_node(self)
