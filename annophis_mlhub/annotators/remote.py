from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

import httpx
import websockets

from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument
from annophis_mlhub.models import WsInputUnit, WsOutputUnit


class _WsSession:
    """Thin wrapper around a websockets connection for streaming annotation."""

    def __init__(self, ws):
        self._ws = ws

    async def send(self, unit: WsInputUnit) -> None:
        await self._ws.send(unit.model_dump_json(by_alias=True))

    async def __aiter__(self):
        async for message in self._ws:
            yield WsOutputUnit.model_validate_json(message)


class RemoteAnnotator(ABC):
    """Base for annotators backed by an external API."""

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    base_url: str = ""
    lif_contract: LIFContract

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        base_url: str | None = None,
        description: str | None = None,
        requires_language: list[str] | None = None,
        requires_annotation: list[str] | None = None,
        requires_feature: list[str] | None = None,
        produces_annotation: list[str] | None = None,
        produces_feature: list[str] | None = None,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if base_url is not None:
            self.base_url = base_url
        if description is not None:
            self.description = description
        self._client: httpx.AsyncClient | None = None
        self.lif_contract = LIFContract(
            requires_language=requires_language or [],
            requires_annotation=requires_annotation or [],
            requires_feature=requires_feature or [],
            produces_annotation=produces_annotation or [],
            produces_feature=produces_feature or [],
        )

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    @abstractmethod
    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]: ...

    @abstractmethod
    async def info(self) -> dict[str, Any]: ...

    def stream_session(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not support stream_session()"
        )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class GenericRemoteAnnotator(RemoteAnnotator):
    """Remote annotator that POSTs to {base_url}/annotate.

    Expects the remote service to speak the LIF protocol:

        POST /annotate  {LIFDocument JSON}
        -> {"annotations": [LIFAnnotation, ...]}

    Configured entirely from TOML — no subclassing needed:

        [[annotator]]
        name = "remote-ner"
        annotation_type = "ner"
        class_path = "annophis_mlhub.annotators.remote.GenericRemoteAnnotator"
        base_url = "http://localhost:8001"
    """

    def __init__(self, endpoint: str = "/annotate", timeout: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.timeout = timeout

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]:
        client = await self.get_client()
        resp = await client.post(
            self.endpoint,
            json=doc.model_dump(by_alias=True),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [LIFAnnotation.model_validate(a) for a in data["annotations"]]

    async def info(self) -> dict[str, Any]:
        from annophis_mlhub.annotators.descriptors import build_descriptor_node

        try:
            client = await self.get_client()
            resp = await client.get("/info")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return build_descriptor_node(self)

    @asynccontextmanager
    async def stream_session(self):
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{ws_url}/annotate") as ws:
            yield _WsSession(ws)
