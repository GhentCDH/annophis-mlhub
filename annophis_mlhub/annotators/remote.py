import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import websockets

from annophis_mlhub.annotators.mixin import AnnotatorMixin
from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument
from annophis_mlhub.models import WsInputUnit, WsOutputUnit

logger = logging.getLogger(__name__)


class _WsSession:
    """Thin wrapper around a websockets connection for streaming annotation."""

    def __init__(self, ws):
        self._ws = ws

    async def send(self, unit: WsInputUnit) -> None:
        await self._ws.send(unit.model_dump_json(by_alias=True))

    async def __aiter__(self) -> AsyncIterator[WsOutputUnit]:
        async for message in self._ws:
            yield WsOutputUnit.model_validate_json(message)


class RemoteAnnotator(AnnotatorMixin, ABC):
    """Base for annotators backed by an external API."""

    base_url: str = ""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base_url is not None:
            self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

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

    supports_streaming = True

    def __init__(self, endpoint: str = "/annotate", timeout: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.timeout = timeout
        self._contract_synced = False

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]:
        client = await self.get_client()
        try:
            resp = await client.post(
                self.endpoint,
                json=doc.model_dump(by_alias=True),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Annotator {self.name!r} is not reachable at {self.base_url}"
            )
        except httpx.TimeoutException:
            raise RuntimeError(f"Annotator {self.name!r} timed out ({self.timeout}s)")
        data = resp.json()
        return [LIFAnnotation.model_validate(a) for a in data["annotations"]]

    async def _sync_contract(self, data: dict[str, Any]) -> None:
        """Merge remote contract fields into the local contract (once).

        Fields already set locally (e.g. from TOML) take precedence.
        """
        if self._contract_synced:
            return
        self._contract_synced = True

        _LIST_FIELDS = {
            "annophis_mlhub:requiresLanguage": "requires_language",
            "annophis_mlhub:requiresAnnotation": "requires_annotation",
            "annophis_mlhub:requiresFeature": "requires_feature",
            "annophis_mlhub:producesAnnotation": "produces_annotation",
            "annophis_mlhub:producesFeature": "produces_feature",
        }

        updates: dict[str, Any] = {}
        for json_key, field_name in _LIST_FIELDS.items():
            if json_key in data and not getattr(self.lif_contract, field_name):
                updates[field_name] = [
                    entry["@id"] if isinstance(entry, dict) else entry
                    for entry in data[json_key]
                ]

        if (
            "annophis_mlhub:inputGranularity" in data
            and not self.lif_contract.input_granularity
        ):
            updates["input_granularity"] = data["annophis_mlhub:inputGranularity"]

        if updates:
            self.lif_contract = LIFContract(
                **{
                    field: updates.get(field, getattr(self.lif_contract, field))
                    for field in LIFContract.model_fields
                }
            )
            logger.info(
                "Synced contract from %s: %s",
                self.base_url,
                list(updates.keys()),
            )

    async def info(self) -> dict[str, Any]:
        from annophis_mlhub.annotators.descriptors import build_descriptor_node

        try:
            client = await self.get_client()
            resp = await client.get("/info")
            resp.raise_for_status()
            data = resp.json()
            await self._sync_contract(data)
            return data
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch /info from %s: %s", self.base_url, exc)
            return build_descriptor_node(self)

    @asynccontextmanager
    async def stream_session(self):
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{ws_url}/annotate") as ws:
            yield _WsSession(ws)
