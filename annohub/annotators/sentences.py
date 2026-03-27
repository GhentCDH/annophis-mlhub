import re

from annohub.annotators.local import LocalAnnotator
from annohub.models import AnnotationResult, Document, Span


class SentenceAnnotator(LocalAnnotator):
    """Splits text into sentences using punctuation boundaries."""

    description = "Splits text into sentence spans."
    labels = ["SENTENCE"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        spans = []
        for match in re.finditer(r"[^.!?]*[.!?]", doc.text):
            text = match.group().strip()
            if text:
                spans.append(
                    Span(
                        start=match.start() + (match.end() - match.start() - len(match.group().lstrip())),
                        end=match.end(),
                        label="SENTENCE",
                        text=text,
                    )
                )
        # catch trailing text without terminal punctuation
        last_end = spans[-1].end if spans else 0
        tail = doc.text[last_end:].strip()
        if tail:
            start = doc.text.index(tail, last_end)
            spans.append(
                Span(start=start, end=start + len(tail), label="SENTENCE", text=tail)
            )
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=spans,
        )


class SentenceCountAnnotator(LocalAnnotator):
    """Counts the number of sentences in the document.

    Requires the 'sentences' key produced by SentenceAnnotator.
    Produces a single span whose text is the count.
    """

    description = "Counts sentences (requires sentences layer)."
    labels = ["COUNT"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        sentences = doc.model_dump().get("sentences", [])
        count = len(sentences)
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[
                Span(start=0, end=len(doc.text), label="COUNT", text=str(count))
            ],
        )
