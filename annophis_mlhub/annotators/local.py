import asyncio
from abc import ABC, abstractmethod

from annophis_mlhub.models import AnnotationResult, AnnotatorInfo, Contract, Document

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
    contract: Contract

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        description: str | None = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        requires: dict[str, bool] | None = None,
        produces: list[str] | None = None,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if description is not None:
            self.description = description
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.contract = Contract(
            requires=requires if requires is not None else {"text": True},  # ty:ignore[invalid-argument-type]
            produces=produces if produces is not None else [self.annotation_type],
        )

    @abstractmethod
    def annotate_sync(self, doc: Document) -> AnnotationResult:
        """Synchronous, blocking annotation. Runs in a thread."""
        ...

    def info_sync(self) -> AnnotatorInfo:
        """Return metadata about this annotator. Override to customise."""
        return AnnotatorInfo(
            name=self.name,
            annotation_type=self.annotation_type,
            kind="local",
            description=self.description,
            contract=self.contract,
        )

    async def annotate(self, doc: Document) -> AnnotationResult:
        async with self._semaphore:
            return await asyncio.to_thread(self.annotate_sync, doc)

    async def info(self) -> AnnotatorInfo:
        return self.info_sync()
