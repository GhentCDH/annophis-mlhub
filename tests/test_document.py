"""Tests for LIF document models."""

from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
    LIFDocument,
    LIFText,
    LIFView,
    ViewMetadata,
)


def test_lif_document_creation():
    doc = LIFDocument(text=LIFText(value="hello world"))
    assert doc.text.value == "hello world"
    assert doc.views == []


def test_lif_document_with_language():
    doc = LIFDocument(text=LIFText(value="hello", language="lexvo:eng"))
    assert doc.text.language == "lexvo:eng"


def test_lif_document_with_view():
    doc = LIFDocument(
        text=LIFText(value="hello"),
        views=[
            LIFView(
                id="v0",
                metadata=ViewMetadata(
                    contains={"Token": ContainsEntry(producer="tokenizer")}
                ),
                annotations=[LIFAnnotation(id="t0", type="Token", start=0, end=5)],
            )
        ],
    )
    assert len(doc.views) == 1
    assert len(doc.views[0].annotations) == 1
    assert doc.views[0].annotations[0].type == "Token"


def test_lif_document_json_roundtrip():
    """Test that serialization with aliases produces valid JSON-LD."""
    doc = LIFDocument(
        text=LIFText(value="Fido barks.", language="lexvo:eng"),
        views=[
            LIFView(
                id="v0",
                annotations=[
                    LIFAnnotation(
                        id="t0",
                        type="Token",
                        start=0,
                        end=4,
                        features={"word": "Fido"},
                    )
                ],
            )
        ],
    )
    data = doc.model_dump(by_alias=True, exclude_none=True)
    assert data["text"]["@value"] == "Fido barks."
    assert data["text"]["@language"] == "lexvo:eng"
    assert data["views"][0]["annotations"][0]["@type"] == "Token"
    # Should round-trip back
    doc2 = LIFDocument.model_validate(data)
    assert doc2.text.value == doc.text.value
    assert doc2.views[0].annotations[0].type == "Token"


def test_lif_document_from_json_ld():
    """Parse a JSON-LD payload (as would come from a client)."""
    payload = {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": "Alice met Bob."},
        "views": [
            {
                "id": "v1",
                "metadata": {
                    "contains": {
                        "Sentence": {"producer": "splitter", "type": "splitter"}
                    }
                },
                "annotations": [
                    {"@type": "Sentence", "id": "s0", "start": 0, "end": 14}
                ],
            }
        ],
    }
    doc = LIFDocument.model_validate(payload)
    assert doc.text.value == "Alice met Bob."
    assert doc.views[0].id == "v1"
    assert doc.views[0].annotations[0].type == "Sentence"
    assert "Sentence" in doc.views[0].metadata.contains


def _doc_with_annotations():
    return LIFDocument(
        text=LIFText(value="Alice met Bob."),
        views=[
            LIFView(
                id="v0",
                annotations=[
                    LIFAnnotation(id="s0", type="Sentence", start=0, end=14),
                    LIFAnnotation(
                        id="t0",
                        type="Token",
                        start=0,
                        end=5,
                        features={"word": "Alice"},
                    ),
                    LIFAnnotation(
                        id="t1", type="Token", start=6, end=9, features={"word": "met"}
                    ),
                    LIFAnnotation(
                        id="t2",
                        type="Token",
                        start=10,
                        end=13,
                        features={"word": "Bob"},
                    ),
                    LIFAnnotation(
                        id="ne0",
                        type="NamedEntity",
                        start=0,
                        end=5,
                        features={"category": "PER"},
                    ),
                ],
            )
        ],
    )


def test_annotations_by_type():
    doc = _doc_with_annotations()
    tokens = list(doc.annotations("Token"))
    assert len(tokens) == 3
    assert all(a.type == "Token" for a in tokens)


def test_spans():
    doc = _doc_with_annotations()
    assert list(doc.spans("Token")) == [(0, 5), (6, 9), (10, 13)]


def test_span_texts():
    doc = _doc_with_annotations()
    assert list(doc.span_texts("Token")) == ["Alice", "met", "Bob"]


def test_sentences():
    doc = _doc_with_annotations()
    assert list(doc.sentences()) == [(0, 14)]


def test_tokens():
    doc = _doc_with_annotations()
    assert list(doc.tokens()) == [(0, 5), (6, 9), (10, 13)]


def test_annotations_empty_views():
    doc = LIFDocument(text=LIFText(value="hello"))
    assert list(doc.annotations("Token")) == []
    assert list(doc.sentences()) == []


def test_annotations_no_match():
    doc = _doc_with_annotations()
    assert list(doc.annotations("Dependency")) == []
