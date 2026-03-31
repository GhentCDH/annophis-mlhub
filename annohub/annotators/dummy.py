import re

from annohub.annotators.local import LocalAnnotator
from annohub.models import AnnotationResult, AnnotatorInfo, Document, Span


class DummyNerAnnotator(LocalAnnotator):
    """Dummy NER annotator that finds capitalized words."""

    description = "Regex-based NER that matches capitalized words."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        spans = []
        for match in re.finditer(r"\b[A-Z][a-z]+\b", doc.text):
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
            contract=self.contract,
        )
