from typing import Protocol, runtime_checkable

from konekaare.models import AnnotationRequest, AnnotationResult


# The Annotator Protocol: duck typing; any class that has these properties is an 'Annotator',
#                         even without explicitely extending this class
@runtime_checkable
class Annotator(Protocol):
    """Protocol that all annotators must satisfy."""

    name: str
    annotation_type: str

    async def annotate(self, request: AnnotationRequest) -> AnnotationResult: ...
