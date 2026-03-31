from typing import Protocol, runtime_checkable

from annohub.models import AnnotationResult, AnnotatorInfo, Contract, Document


@runtime_checkable
class Annotator(Protocol):
    """Protocol that all annotators must satisfy."""

    name: str
    annotation_type: str
    description: str
    contract: Contract

    async def annotate(self, doc: Document) -> AnnotationResult: ...

    async def info(self) -> AnnotatorInfo: ...
