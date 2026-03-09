from typing import Protocol, runtime_checkable

from konekaare.models import AnnotationRequest, AnnotationResult


@runtime_checkable
class Annotator(Protocol):
    """Protocol that all annotators must satisfy."""

    name: str
    annotation_type: str

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult: ...
