"""Tests for content-addressed annotation caching."""

from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.cache import (
    CachePlan,
    build_filtered_document,
    compute_cache_plan,
    compute_input_hash,
    remove_stale_annotations,
    stamp_annotations,
)
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
    LIFContract,
    LIFDocument,
    LIFText,
    LIFView,
    ViewMetadata,
)

PRODUCER = "http://localhost:8000/vocab/test-tokenizer"


def _make_doc(text="a b c. d e f. g h i."):
    return LIFDocument(
        text=LIFText(value=text),
        views=[LIFView(id="v0", metadata=ViewMetadata(), annotations=[])],
    )


def _make_sentence_annotations(spans: list[tuple[int, int]]):
    return [
        LIFAnnotation(id=f"s{i}", type="Sentence", start=s, end=e)
        for i, (s, e) in enumerate(spans)
    ]


# ── compute_input_hash ──────────────────────────────────────────────────────


def test_hash_deterministic():
    assert compute_input_hash("hello") == compute_input_hash("hello")


def test_hash_differs_on_text():
    assert compute_input_hash("hello") != compute_input_hash("world")


def test_hash_differs_on_upstream():
    ann = LIFAnnotation(id="a1", type="Sentence", start=0, end=5)
    h1 = compute_input_hash("hello", [ann])
    h2 = compute_input_hash("hello", None)
    assert h1 != h2


# ── compute_cache_plan (document-level) ─────────────────────────────────────


def test_doc_level_all_miss():
    doc = _make_doc("hello")
    contract = LIFContract(produces_annotation=["Token"])
    plan = compute_cache_plan(doc, PRODUCER, contract)
    assert not plan.skip_entirely
    assert plan.miss_spans == [(0, 5)]
    assert plan.hits == []


def test_doc_level_cache_hit():
    doc = _make_doc("hello")
    contract = LIFContract(produces_annotation=["Token"])
    expected_hash = compute_input_hash("hello")

    # Add an existing annotation with matching hash
    doc.views[0].annotations.append(
        LIFAnnotation(
            id="t0",
            type="Token",
            start=0,
            end=5,
            metadata={"input_hash": expected_hash, "producer": PRODUCER},
        )
    )
    plan = compute_cache_plan(doc, PRODUCER, contract)
    assert plan.skip_entirely
    assert len(plan.hits) == 1


# ── compute_cache_plan (per-span) ──────────────────────────────────────────


def test_per_span_all_miss():
    doc = _make_doc("a b c. d e f.")
    sentences = _make_sentence_annotations([(0, 6), (7, 13)])
    doc.views[0].annotations = sentences
    doc.views[0].metadata.contains["Sentence"] = ContainsEntry(producer="splitter")

    contract = LIFContract(
        requires_annotation=["Sentence"],
        produces_annotation=["Token"],
        input_granularity="Sentence",
    )
    plan = compute_cache_plan(doc, PRODUCER, contract)
    assert not plan.skip_entirely
    assert len(plan.miss_spans) == 2


def test_per_span_partial_hit():
    doc = _make_doc("a b c. d e f.")
    sentences = _make_sentence_annotations([(0, 6), (7, 13)])
    doc.views[0].annotations = list(sentences)
    doc.views[0].metadata.contains["Sentence"] = ContainsEntry(producer="splitter")

    contract = LIFContract(
        requires_annotation=["Sentence"],
        produces_annotation=["Token"],
        input_granularity="Sentence",
    )

    # Compute the hash for the first sentence and add a cached token
    h = compute_input_hash("a b c.", [sentences[0]], strip_offsets=True)
    doc.views[0].annotations.append(
        LIFAnnotation(
            id="tok0",
            type="Token",
            start=0,
            end=1,
            metadata={
                "input_hash": h,
                "granularity_span": "0:6",
                "producer": PRODUCER,
            },
        )
    )

    plan = compute_cache_plan(doc, PRODUCER, contract)
    assert not plan.skip_entirely
    assert len(plan.hits) == 1
    assert plan.hits[0].id == "tok0"
    assert len(plan.miss_spans) == 1
    assert plan.miss_spans[0] == (7, 13)


def test_per_span_all_hit():
    doc = _make_doc("a b c. d e f.")
    sentences = _make_sentence_annotations([(0, 6), (7, 13)])
    doc.views[0].annotations = list(sentences)
    doc.views[0].metadata.contains["Sentence"] = ContainsEntry(producer="splitter")

    contract = LIFContract(
        requires_annotation=["Sentence"],
        produces_annotation=["Token"],
        input_granularity="Sentence",
    )

    for i, (s, e) in enumerate([(0, 6), (7, 13)]):
        h = compute_input_hash(doc.text.value[s:e], [sentences[i]], strip_offsets=True)
        doc.views[0].annotations.append(
            LIFAnnotation(
                id=f"tok{i}",
                type="Token",
                start=s,
                end=s + 1,
                metadata={
                    "input_hash": h,
                    "granularity_span": f"{s}:{e}",
                    "producer": PRODUCER,
                },
            )
        )

    plan = compute_cache_plan(doc, PRODUCER, contract)
    assert plan.skip_entirely
    assert len(plan.hits) == 2


# ── build_filtered_document ─────────────────────────────────────────────────


def test_filtered_doc_keeps_text():
    doc = _make_doc("a b c. d e f.")
    sentences = _make_sentence_annotations([(0, 6), (7, 13)])
    doc.views[0].annotations = sentences

    contract = LIFContract(requires_annotation=["Sentence"])
    filtered = build_filtered_document(doc, [(0, 6)], contract)

    assert filtered.text.value == "a b c. d e f."  # full text preserved
    assert len(filtered.views[0].annotations) == 1
    assert filtered.views[0].annotations[0].id == "s0"


# ── stamp_annotations ──────────────────────────────────────────────────────


def test_stamp_doc_level():
    doc = _make_doc("hello")
    contract = LIFContract(produces_annotation=["Token"])
    ann = LIFAnnotation(id="t0", type="Token", start=0, end=5)

    stamped = stamp_annotations([ann], PRODUCER, contract, doc)
    assert stamped[0].metadata["input_hash"] == compute_input_hash("hello")
    assert stamped[0].metadata["producer"] == PRODUCER
    assert "granularity_span" not in stamped[0].metadata


def test_stamp_per_span():
    doc = _make_doc("a b c. d e f.")
    sentences = _make_sentence_annotations([(0, 6), (7, 13)])
    doc.views[0].annotations = sentences
    doc.views[0].metadata.contains["Sentence"] = ContainsEntry(producer="splitter")

    contract = LIFContract(
        requires_annotation=["Sentence"],
        produces_annotation=["Token"],
        input_granularity="Sentence",
    )
    ann = LIFAnnotation(id="tok0", type="Token", start=2, end=3)
    stamped = stamp_annotations([ann], PRODUCER, contract, doc)
    assert stamped[0].metadata["granularity_span"] == "0:6"
    assert stamped[0].metadata["input_hash"] is not None


# ── remove_stale_annotations ───────────────────────────────────────────────


def test_remove_stale():
    doc = _make_doc("hello")
    hit = LIFAnnotation(
        id="keep",
        type="Token",
        start=0,
        end=3,
        metadata={"producer": PRODUCER},
    )
    stale = LIFAnnotation(
        id="remove",
        type="Token",
        start=3,
        end=5,
        metadata={"producer": PRODUCER},
    )
    other = LIFAnnotation(
        id="other",
        type="Sentence",
        start=0,
        end=5,
        metadata={"producer": "someone-else"},
    )
    doc.views[0].annotations = [hit, stale, other]

    plan = CachePlan(hits=[hit])
    result = remove_stale_annotations(doc, PRODUCER, plan)
    ids = [a.id for a in result.views[0].annotations]
    assert "keep" in ids
    assert "other" in ids
    assert "remove" not in ids


# ── Integration: pipeline with caching ──────────────────────────────────────


class SentenceSplitter(LocalAnnotator):
    def __init__(self):
        super().__init__(
            name="splitter",
            annotation_type="sentence",
            produces_annotation=["Sentence"],
        )

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        text = doc.text.value
        anns = []
        start = 0
        for i, ch in enumerate(text):
            if ch == ".":
                anns.append(
                    LIFAnnotation(
                        id=f"s{len(anns)}",
                        type="Sentence",
                        start=start,
                        end=i + 1,
                    )
                )
                start = i + 2  # skip space after period
        return anns


class Tokenizer(LocalAnnotator):
    call_count: int = 0

    def __init__(self):
        super().__init__(
            name="tokenizer",
            annotation_type="token",
            requires_annotation=["Sentence"],
            produces_annotation=["Token"],
            input_granularity="Sentence",
        )
        self.call_count = 0

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        self.call_count += 1
        anns = []
        for sent in doc.annotations("Sentence"):
            text = doc.text.value[sent.start : sent.end]
            offset = sent.start if sent.start is not None else 0
            for word in text.split():
                idx = text.index(word)
                anns.append(
                    LIFAnnotation(
                        id=f"t{len(anns)}",
                        type="Token",
                        start=offset + idx,
                        end=offset + idx + len(word),
                    )
                )
        return anns


def test_pipeline_caching_integration(client):
    splitter = SentenceSplitter()
    tokenizer = Tokenizer()
    annotators.register(splitter)
    annotators.register(tokenizer)

    doc_payload = {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": "a b. c d. e f."},
    }

    # First run: everything computed
    resp = client.post(
        "/annotate",
        json={
            "document": doc_payload,
            "annotators": ["splitter", "tokenizer"],
        },
    )
    assert resp.status_code == 200
    result1 = resp.json()
    assert tokenizer.call_count == 1

    # Second run with same document output: tokenizer should be skipped
    resp = client.post(
        "/annotate",
        json={
            "document": result1,
            "annotators": ["splitter", "tokenizer"],
        },
    )
    assert resp.status_code == 200
    assert tokenizer.call_count == 1  # not called again
