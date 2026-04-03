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
