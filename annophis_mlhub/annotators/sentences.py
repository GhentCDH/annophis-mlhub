import re

from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument


class SentenceAnnotator(LocalAnnotator):
    """Splits text into sentences using punctuation boundaries."""

    description = "Splits text into sentence spans."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        text = doc.text.value
        annotations = []
        idx = 0

        for match in re.finditer(r"[^.!?]*[.!?]", text):
            t = match.group().strip()
            if t:
                start = match.start() + (
                    match.end() - match.start() - len(match.group().lstrip())
                )
                annotations.append(
                    LIFAnnotation(
                        id=f"s{idx}",
                        type="Sentence",
                        start=start,
                        end=match.end(),
                    )
                )
                idx += 1

        # catch trailing text without terminal punctuation
        last_end = annotations[-1].end if annotations else 0
        tail = text[last_end:].strip()
        if tail:
            start = text.index(tail, last_end)
            annotations.append(
                LIFAnnotation(
                    id=f"s{idx}",
                    type="Sentence",
                    start=start,
                    end=start + len(tail),
                )
            )

        return annotations


class SentenceCountAnnotator(LocalAnnotator):
    """Counts the number of sentences in the document.

    Requires Sentence annotations in the view (produced by SentenceAnnotator).
    Produces a single annotation whose features contain the count.
    """

    description = "Counts sentences (requires Sentence annotations)."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        count = sum(1 for _ in doc.sentences())

        return [
            LIFAnnotation(
                id="sc0",
                type="SentenceCount",
                start=0,
                end=len(doc.text.value),
                features={"count": count},
            )
        ]
