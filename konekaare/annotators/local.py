import asyncio
from abc import ABC, abstractmethod

from konekaare.models import AnnotationRequest, AnnotationResult

# Default max concurrent inference threads across all local annotators.
_DEFAULT_MAX_CONCURRENCY = 1


class LocalAnnotator(ABC):
    """Base for annotators that run blocking local models.

    Uses a semaphore to bound concurrent inference threads.  This prevents
    multiple requests from hammering a GPU model in parallel.  The default
    concurrency is 1 (fully serialized); override via constructor kwarg.
    """

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    labels: list[str] = []

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    def annotate_sync(self, request: AnnotationRequest) -> AnnotationResult:
        """Synchronous, blocking annotation. Runs in a thread."""
        ...

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        async with self._semaphore:
            return await asyncio.to_thread(self.annotate_sync, request)
