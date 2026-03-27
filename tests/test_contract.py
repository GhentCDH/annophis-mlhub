"""Tests for contract validation logic."""

import pytest

from annohub import annotators
from annohub.annotators.local import LocalAnnotator
from annohub.models import AnnotationResult, Contract, Document, Span


class NerAnnotator(LocalAnnotator):
    """Annotator that requires 'text' and produces 'NER'."""

    name = "test-ner"
    annotation_type = "NER"

    def __init__(self, **kwargs):
        super().__init__(
            requires={"text": True},
            produces=["NER"],
            **kwargs,
        )

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[Span(start=0, end=3, label="ENTITY", text=doc.text[:3])],
        )


class ChainedAnnotator(LocalAnnotator):
    """Annotator that requires 'NER' key to already exist."""

    name = "chained"
    annotation_type = "sentiment"

    def __init__(self, **kwargs):
        super().__init__(
            requires={"text": True, "NER": True},
            produces=["sentiment"],
            **kwargs,
        )

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[],
        )


def test_contract_validation_passes(client):
    """Valid document passes contract validation."""
    annotators.register(NerAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello"}, "annotators": ["test-ner"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "NER" in data


def test_contract_validation_missing_key(client):
    """Missing required key returns 422."""
    annotators.register(ChainedAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello"}, "annotators": ["chained"]},
    )
    assert resp.status_code == 422
    assert "NER" in resp.json()["detail"]


def test_contract_pipeline_chaining(client):
    """First annotator produces key needed by second annotator."""
    annotators.register(NerAnnotator())
    annotators.register(ChainedAnnotator())
    resp = client.post(
        "/annotate",
        json={
            "document": {"text": "hello"},
            "annotators": ["test-ner", "chained"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "NER" in data
    assert "sentiment" in data


def test_contract_produces_keys_appear_in_output(client):
    """Contract.produces keys appear in the output document."""
    ann = NerAnnotator()
    annotators.register(ann)
    assert ann.contract.produces == ["NER"]

    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello"}, "annotators": ["test-ner"]},
    )
    data = resp.json()
    for key in ann.contract.produces:
        assert key in data
