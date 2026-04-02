import re
import time

from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument


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
                    "requires_language",
                    "requires_annotation",
                    "requires_feature",
                    "produces_annotation",
                    "produces_feature",
                )
            }
        )
        self.delay = delay

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        time.sleep(self.delay)
        return [
            LIFAnnotation(
                id=f"ne{i}",
                type="NamedEntity",
                start=m.start(),
                end=m.end(),
                features={"category": "ENTITY", "word": m.group()},
            )
            for i, m in enumerate(re.finditer(r"\b[A-Z][a-z]+\b", doc.text.value))
        ]
