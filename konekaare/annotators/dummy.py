import re

from konekaare.annotators.local import LocalAnnotator
from konekaare.models import AnnotationRequest, AnnotationResult, AnnotatorInfo, Span


class DummyNerAnnotator(LocalAnnotator):
    """Dummy NER annotator that finds capitalized words."""

    description = "Regex-based NER that matches capitalized words."
    labels = ["ENTITY"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def info_sync(self) -> AnnotatorInfo:
        return AnnotatorInfo(
            name=self.name,
            annotation_type=self.annotation_type,
            kind="local",
            description=self.description,
            labels=self.labels,
        )
