import asyncio
from abc import ABC, abstractmethod
from typing import Any

from annophis_mlhub.annotators.descriptors import build_descriptor_node
from annophis_mlhub.annotators.mixin import AnnotatorMixin
from annophis_mlhub.lif import LIFAnnotation, LIFDocument

_DEFAULT_MAX_CONCURRENCY = 1


class LocalAnnotator(AnnotatorMixin, ABC):
    """Base for annotators that run blocking local models.

    Uses a semaphore to bound concurrent inference threads.  This prevents
    multiple requests from hammering a GPU model in parallel.  The default
    concurrency is 1 (fully serialized); override via constructor kwarg.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        """Synchronous, blocking annotation. Runs in a thread."""
        ...

    def info_sync(self) -> dict[str, Any]:
        """Return JSON-LD descriptor for this annotator. Override to customise."""
        return build_descriptor_node(self)

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]:
        async with self._semaphore:
            return await asyncio.to_thread(self.annotate_sync, doc)

    async def info(self) -> dict[str, Any]:
        return self.info_sync()
