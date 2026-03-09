import asyncio
from abc import ABC, abstractmethod

from konekaare.models import AnnotationRequest, AnnotationResult


class LocalAnnotator(ABC):
    """Base for annotators that run blocking local models."""

    name: str
    annotation_type: str

    @abstractmethod
    def annotate_sync(self, request: AnnotationRequest) -> AnnotationResult:
        """Synchronous, blocking annotation. Runs in a thread."""
        ...

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        return await asyncio.to_thread(self.annotate_sync, request)
