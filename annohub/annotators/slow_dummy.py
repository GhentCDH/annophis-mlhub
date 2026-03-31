import re
import time

from annohub.annotators.local import LocalAnnotator
from annohub.models import AnnotationResult, Document, Span


class SlowDummyAnnotator(LocalAnnotator):
    """Dummy NER annotator with an artificial delay — useful for streaming demos.

    Finds capitalized words just like dummy-ner, but sleeps for `delay` seconds
    before returning so results visibly trickle in when streamed.
    """

    description = "Slow dummy NER (simulates a heavy model). Good for demo streaming."

    def __init__(self, delay: float = 1.0, **kwargs):
        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "name",
                    "annotation_type",
                    "max_concurrency",
                    "requires",
                    "produces",
                )
            }
        )
        self.delay = delay

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        time.sleep(self.delay)
        spans = [
            Span(start=m.start(), end=m.end(), label="ENTITY", text=m.group())
            for m in re.finditer(r"\b[A-Z][a-z]+\b", doc.text)
        ]
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=spans,
        )
