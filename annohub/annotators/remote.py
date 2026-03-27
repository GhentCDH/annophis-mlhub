from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

import httpx
import websockets

from annohub.models import AnnotationResult, AnnotatorInfo, Contract, Document, WsInputUnit, WsOutputUnit


class _WsSession:
    """Thin wrapper around a websockets connection for streaming annotation."""

    def __init__(self, ws):
        self._ws = ws

    async def send(self, unit: WsInputUnit) -> None:
        await self._ws.send(unit.model_dump_json())

    async def __aiter__(self):
        async for message in self._ws:
            yield WsOutputUnit.model_validate_json(message)


class RemoteAnnotator(ABC):
    """Base for annotators backed by an external API."""

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    labels: list[str]
    base_url: str = ""
    contract: Contract

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        base_url: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
        requires: dict[str, bool] | None = None,
        produces: list[str] | None = None,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if base_url is not None:
            self.base_url = base_url
        if description is not None:
            self.description = description
        self.labels = labels if labels is not None else []
        self._client: httpx.AsyncClient | None = None
        self.contract = Contract(
            requires=requires if requires is not None else {"text": True},
            produces=produces if produces is not None else [self.annotation_type],
        )

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    @abstractmethod
    async def annotate(self, doc: Document) -> AnnotationResult: ...

    @abstractmethod
    async def info(self) -> AnnotatorInfo: ...

    def stream_session(self):
        raise NotImplementedError(f"{type(self).__name__} does not support stream_session()")

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class GenericRemoteAnnotator(RemoteAnnotator):
    """Remote annotator that POSTs to {base_url}/annotate.

    Expects the remote service to speak the standard annohub protocol:

        POST /annotate  {document JSON}
        -> {"annotator": "...", "annotation_type": "...", "spans": [...]}

    Configured entirely from TOML — no subclassing needed:

        [[annotator]]
        name = "remote-ner"
        annotation_type = "ner"
        class_path = "annohub.annotators.remote.GenericRemoteAnnotator"
        base_url = "http://localhost:8001"
    """

    def __init__(self, endpoint: str = "/annotate", timeout: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.timeout = timeout

    async def annotate(self, doc: Document) -> AnnotationResult:
        client = await self.get_client()
        resp = await client.post(
            self.endpoint,
            json=doc.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=data["spans"],
        )

    async def info(self) -> AnnotatorInfo:
        client = await self.get_client()
        resp = await client.get("/info")
        resp.raise_for_status()
        data = resp.json()
        return AnnotatorInfo(
            name=data["name"],
            annotation_type=data["annotation_type"],
            description=data["description"],
            labels=data["labels"],
            kind=data["kind"],
            contract=data.get("contract", {}),
        )

    @asynccontextmanager
    async def stream_session(self):
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{ws_url}/annotate") as ws:
            yield _WsSession(ws)
