import re

from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument


class DummyNerAnnotator(LocalAnnotator):
    """Dummy NER annotator that finds capitalized words."""

    description = "Regex-based NER that matches capitalized words."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        annotations = []
        for i, match in enumerate(re.finditer(r"\b[A-Z][a-z]+\b", doc.text.value)):
            annotations.append(
                LIFAnnotation(
                    id=f"ne{i}",
                    type="NamedEntity",
                    start=match.start(),
                    end=match.end(),
                    features={"category": "ENTITY", "word": match.group()},
                )
            )
        return annotations
