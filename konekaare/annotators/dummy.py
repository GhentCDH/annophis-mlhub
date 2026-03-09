import re

from konekaare.annotators.local import LocalAnnotator
from konekaare.models import AnnotationRequest, AnnotationResult, Span


class DummyNerAnnotator(LocalAnnotator):
    """Dummy NER annotator that finds capitalized words."""

    def __init__(self, name: str, annotation_type: str):
        self.name = name
        self.annotation_type = annotation_type

    def annotate_sync(self, request: AnnotationRequest) -> AnnotationResult:
        spans = []
        for match in re.finditer(r"\b[A-Z][a-z]+\b", request.text):
            spans.append(
                Span(
                    start=match.start(),
                    end=match.end(),
                    label="ENTITY",
                    text=match.group(),
                )
            )
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=spans,
        )
