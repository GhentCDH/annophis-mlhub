"""Tests for contract validation logic."""

from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.models import AnnotationResult, Contract, Document, Span
from annophis_mlhub.routes.annotate import validate_contract


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


# --- dot-path and value constraint tests ---


class TestDotPathValidation:
    """Tests for dot-separated path resolution and value constraints."""

    def test_dot_path_exists(self):
        doc = Document(text="hello", meta={"lang": "en"})
        contract = Contract(requires={"meta.lang": True})
        assert validate_contract(doc, contract) == []

    def test_dot_path_missing(self):
        doc = Document(text="hello", meta={})
        contract = Contract(requires={"meta.lang": True})
        violations = validate_contract(doc, contract)
        assert "meta.lang" in violations

    def test_dot_path_value_match(self):
        doc = Document(text="hello", meta={"lang": "en"})
        contract = Contract(requires={"meta.lang": "en"})
        assert validate_contract(doc, contract) == []

    def test_dot_path_value_mismatch(self):
        doc = Document(text="hello", meta={"lang": "fr"})
        contract = Contract(requires={"meta.lang": "en"})
        violations = validate_contract(doc, contract)
        assert len(violations) == 1
        assert "meta.lang" in violations[0]

    def test_dot_path_value_in_list(self):
        doc = Document(text="hello", meta={"lang": "de"})
        contract = Contract(requires={"meta.lang": ["en", "de"]})
        assert validate_contract(doc, contract) == []

    def test_dot_path_value_not_in_list(self):
        doc = Document(text="hello", meta={"lang": "ja"})
        contract = Contract(requires={"meta.lang": ["en", "de"]})
        violations = validate_contract(doc, contract)
        assert len(violations) == 1
        assert "meta.lang" in violations[0]

    def test_deep_dot_path(self):
        doc = Document(text="hello", meta={"source": {"type": "web"}})
        contract = Contract(requires={"meta.source.type": "web"})
        assert validate_contract(doc, contract) == []

    def test_flat_key_still_works(self):
        doc = Document(text="hello")
        contract = Contract(requires={"text": True})
        assert validate_contract(doc, contract) == []
