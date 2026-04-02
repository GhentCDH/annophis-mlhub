from annophis_mlhub import annotators
from annophis_mlhub.annotators.sentences import (
    SentenceAnnotator,
    SentenceCountAnnotator,
)
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFDocument,
    LIFText,
    LIFView,
    ViewMetadata,
)


def _lif_doc(text):
    return LIFDocument(text=LIFText(value=text))


def _lif_json(text):
    return {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": text},
    }


def test_sentence_split():
    ann = SentenceAnnotator(name="sent", annotation_type="sentences")
    doc = _lif_doc("Hello world. How are you? Fine.")
    annotations = ann.annotate_sync(doc)
    assert len(annotations) == 3
    assert all(a.type == "Sentence" for a in annotations)


def test_sentence_split_no_trailing_punctuation():
    ann = SentenceAnnotator(name="sent", annotation_type="sentences")
    doc = _lif_doc("First sentence. No punctuation")
    annotations = ann.annotate_sync(doc)
    assert len(annotations) == 2
    assert annotations[-1].end == len("First sentence. No punctuation")


def test_sentence_count():
    sent_ann = SentenceAnnotator(
        name="sent",
        annotation_type="sentences",
        produces_annotation=["lapps:Sentence"],
    )
    count_ann = SentenceCountAnnotator(
        name="count",
        annotation_type="sentence_count",
        requires_annotation=["lapps:Sentence"],
    )
    assert count_ann.lif_contract.requires_annotation == ["lapps:Sentence"]

    doc = _lif_doc("Hello. World.")
    sent_annotations = sent_ann.annotate_sync(doc)
    assert len(sent_annotations) == 2

    view = LIFView(
        id="v0",
        metadata=ViewMetadata(contains={"Sentence": ContainsEntry(producer="sent")}),
        annotations=sent_annotations,
    )
    doc_with_sents = doc.model_copy(update={"views": [view]})
    count_annotations = count_ann.annotate_sync(doc_with_sents)
    assert len(count_annotations) == 1
    assert count_annotations[0].features["count"] == 2


def test_sentence_count_empty():
    count_ann = SentenceCountAnnotator(name="count", annotation_type="sentence_count")
    doc = _lif_doc("Hello world")
    annotations = count_ann.annotate_sync(doc)
    assert annotations[0].features["count"] == 0


def test_pipeline_via_http(client):
    """Test sentence-split -> sentence-count pipeline via HTTP."""
    annotators.register(
        SentenceAnnotator(
            name="sent",
            annotation_type="sentences",
            produces_annotation=["lapps:Sentence"],
        )
    )
    annotators.register(
        SentenceCountAnnotator(
            name="count",
            annotation_type="sentence_count",
            requires_annotation=["lapps:Sentence"],
        )
    )
    resp = client.post(
        "/annotate",
        json={
            "document": _lif_json("Hello world. How are you?"),
            "annotators": ["sent", "count"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    view = data["views"][0]
    types = {a["@type"] for a in view["annotations"]}
    assert "Sentence" in types
    assert "SentenceCount" in types


def test_count_without_sentences_fails(client):
    """sentence-count without sentence-split should fail contract validation."""
    annotators.register(
        SentenceCountAnnotator(
            name="count",
            annotation_type="sentence_count",
            requires_annotation=["lapps:Sentence"],
        )
    )
    resp = client.post(
        "/annotate",
        json={
            "document": _lif_json("Hello world."),
            "annotators": ["count"],
        },
    )
    assert resp.status_code == 422
