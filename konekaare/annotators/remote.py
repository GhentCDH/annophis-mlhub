from abc import ABC, abstractmethod

import httpx

from konekaare.models import AnnotationRequest, AnnotationResult, AnnotatorInfo


class RemoteAnnotator(ABC):
    """Base for annotators backed by an external API."""

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    labels: list[str] = []
    base_url: str = ""

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        base_url: str | None = None,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if base_url is not None:
            self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    @abstractmethod
    async def annotate(self, request: AnnotationRequest) -> AnnotationResult: ...

    @abstractmethod
    async def info(self) -> AnnotatorInfo: ...

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class GenericRemoteAnnotator(RemoteAnnotator):
    """Remote annotator that POSTs to {base_url}/annotate.

    Expects the remote service to speak the standard konekaare protocol:

        POST /annotate  {"text": "..."}
        -> {"annotator": "...", "annotation_type": "...", "spans": [...]}

    Configured entirely from TOML — no subclassing needed:

        [[annotator]]
        name = "remote-ner"
        annotation_type = "ner"
        class_path = "konekaare.annotators.remote.GenericRemoteAnnotator"
        base_url = "http://localhost:8001"
    """

    def __init__(
        self,
        endpoint: str = "/annotate",
        timeout: float = 30.0,
        labels: list[str] | None = None,
        description: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.timeout = timeout
        self.labels = [] if labels is None else labels
        self.description = [] if description is None else description

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        client = await self.get_client()
        resp = await client.post(
            self.endpoint,
            json={"text": request.text},
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
        )
