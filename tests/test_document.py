"""Tests for Document model behavior."""

from annohub.models import AnnotationResult, Document, Span
from annohub.routes.annotate import merge_result


def test_document_creation():
    doc = Document(text="hello world")
    assert doc.text == "hello world"
    assert doc.meta == {}


def test_document_with_meta():
    doc = Document(text="hello", meta={"lang": "en"})
    assert doc.meta == {"lang": "en"}


def test_document_extra_fields():
    doc = Document(
        text="hello", ner=[{"start": 0, "end": 5, "label": "X", "text": "hello"}]
    )
    dump = doc.model_dump()
    assert "ner" in dump
    assert dump["ner"] == [{"start": 0, "end": 5, "label": "X", "text": "hello"}]


def test_merge_result():
    doc = Document(text="hello world")
    result = AnnotationResult(
        annotator="test",
        annotation_type="ner",
        spans=[Span(start=0, end=5, label="ENTITY", text="hello")],
    )
    merged = merge_result(doc, result)
    assert merged.text == "hello world"
    dump = merged.model_dump()
    assert "ner" in dump
    assert len(dump["ner"]) == 1
    assert dump["ner"][0]["label"] == "ENTITY"


def test_merge_preserves_existing_fields():
    doc = Document(text="hello", meta={"lang": "en"}, pos=[])
    result = AnnotationResult(
        annotator="test",
        annotation_type="ner",
        spans=[Span(start=0, end=5, label="X", text="hello")],
    )
    merged = merge_result(doc, result)
    dump = merged.model_dump()
    assert dump["meta"] == {"lang": "en"}
    assert dump["pos"] == []
    assert "ner" in dump
