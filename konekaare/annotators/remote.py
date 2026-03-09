from abc import ABC, abstractmethod

import httpx

from konekaare.models import AnnotationRequest, AnnotationResult


class RemoteAnnotator(ABC):
    """Base for annotators backed by an external API."""

    name: str
    annotation_type: str
    base_url: str

    def __init__(self, name: str, annotation_type: str, base_url: str):
        self.name = name
        self.annotation_type = annotation_type
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    @abstractmethod
    async def annotate(self, request: AnnotationRequest) -> AnnotationResult: ...

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
