"""Tests for LIF contract validation logic."""

from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import (
    LIFAnnotation,
    LIFContract,
    LIFDocument,
    LIFText,
    LIFView,
    ContainsEntry,
    ViewMetadata,
    validate_lif_contract,
)


class NerAnnotator(LocalAnnotator):
    """Annotator that produces NamedEntity annotations."""

    name = "test-ner"
    annotation_type = "ner"

    def __init__(self, **kwargs):
        super().__init__(
            produces_annotation=["lapps:NamedEntity"],
            **kwargs,
        )

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        return [
            LIFAnnotation(
                id="ne0",
                type="NamedEntity",
                start=0,
                end=3,
                features={"category": "ENTITY", "word": doc.text.value[:3]},
            )
        ]


class ChainedAnnotator(LocalAnnotator):
    """Annotator that requires NamedEntity annotations to already exist."""

    name = "chained"
    annotation_type = "sentiment"

    def __init__(self, **kwargs):
        super().__init__(
            requires_annotation=["lapps:NamedEntity"],
            **kwargs,
        )

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        return []


def _lif_doc(text="hello"):
    return {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": text},
    }


def test_contract_validation_passes(client):
    """Annotator with no requirements passes validation."""
    annotators.register(NerAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": _lif_doc(), "annotators": ["test-ner"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["views"]) == 1
    assert len(data["views"][0]["annotations"]) == 1


def test_contract_validation_missing_annotation(client):
    """Missing required annotation type returns 422."""
    annotators.register(ChainedAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": _lif_doc(), "annotators": ["chained"]},
    )
    assert resp.status_code == 422
    assert "NamedEntity" in resp.json()["detail"]


def test_contract_pipeline_chaining(client):
    """First annotator produces annotations needed by second annotator."""
    annotators.register(NerAnnotator())
    annotators.register(ChainedAnnotator())
    resp = client.post(
        "/annotate",
        json={
            "document": _lif_doc(),
            "annotators": ["test-ner", "chained"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["views"]) == 1
    # Both annotators contributed to the view
    contains = data["views"][0]["metadata"]["contains"]
    assert "NamedEntity" in contains


# --- Direct contract validation tests ---


class TestLanguageValidation:
    def test_language_match(self):
        doc = LIFDocument(
            text=LIFText(value="hello", language="lexvo:grc"),
        )
        contract = LIFContract(requires_language="lexvo:grc")
        assert validate_lif_contract(doc, contract) == []

    def test_language_mismatch(self):
        doc = LIFDocument(
            text=LIFText(value="hello", language="lexvo:eng"),
        )
        contract = LIFContract(requires_language="lexvo:grc")
        violations = validate_lif_contract(doc, contract)
        assert len(violations) == 1
        assert "language" in violations[0]

    def test_language_missing(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
        )
        contract = LIFContract(requires_language="lexvo:grc")
        violations = validate_lif_contract(doc, contract)
        assert len(violations) == 1
        assert "language" in violations[0]


class TestAnnotationTypeValidation:
    def test_required_type_present(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
            views=[
                LIFView(
                    id="v0",
                    metadata=ViewMetadata(
                        contains={"Sentence": ContainsEntry(producer="test")}
                    ),
                )
            ],
        )
        contract = LIFContract(requires_annotation=["lapps:Sentence"])
        assert validate_lif_contract(doc, contract) == []

    def test_required_type_missing(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
            views=[LIFView(id="v0")],
        )
        contract = LIFContract(requires_annotation=["lapps:Sentence"])
        violations = validate_lif_contract(doc, contract)
        assert len(violations) == 1
        assert "Sentence" in violations[0]


class TestFeatureValidation:
    def test_required_feature_present(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
            views=[
                LIFView(
                    id="v0",
                    annotations=[
                        LIFAnnotation(
                            id="t0",
                            type="Token",
                            start=0,
                            end=5,
                            features={"pos": "NN"},
                        )
                    ],
                )
            ],
        )
        contract = LIFContract(requires_feature=["lapps:Token#pos"])
        assert validate_lif_contract(doc, contract) == []

    def test_required_feature_missing(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
            views=[
                LIFView(
                    id="v0",
                    annotations=[LIFAnnotation(id="t0", type="Token", start=0, end=5)],
                )
            ],
        )
        contract = LIFContract(requires_feature=["lapps:Token#pos"])
        violations = validate_lif_contract(doc, contract)
        assert len(violations) == 1
        assert "Token#pos" in violations[0]
