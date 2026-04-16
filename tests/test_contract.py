"""Tests for LIF contract validation logic."""

from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
    LIFContract,
    LIFDocument,
    LIFText,
    LIFView,
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


class TokenizerAnnotator(LocalAnnotator):
    """Produces Token annotations."""

    name = "tokenizer"
    annotation_type = "tokens"

    def __init__(self, **kwargs):
        super().__init__(produces_annotation=["lapps:Token"], **kwargs)

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        tokens = doc.text.value.split()
        result = []
        offset = 0
        for i, word in enumerate(tokens):
            start = doc.text.value.index(word, offset)
            result.append(
                LIFAnnotation(
                    id=f"tok{i}",
                    type="Token",
                    start=start,
                    end=start + len(word),
                    features={"word": word},
                )
            )
            offset = start + len(word)
        return result


class POSTaggerAnnotator(LocalAnnotator):
    """Enriches existing Token annotations with a pos feature."""

    name = "pos-tagger"
    annotation_type = "pos"

    def __init__(self, **kwargs):
        super().__init__(
            requires_annotation=["lapps:Token"],
            produces_feature=["lapps:Token#pos"],
            **kwargs,
        )

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        # Return annotations with the same ids as existing tokens, adding pos
        result = []
        for view in doc.views:
            for ann in view.annotations:
                if ann.type == "Token":
                    result.append(
                        LIFAnnotation(
                            id=ann.id,
                            type="Token",
                            features={"pos": "NN"},
                        )
                    )
        return result


class TrackingAnnotator(LocalAnnotator):
    """Annotator that records whether it was called."""

    name = "tracker"
    annotation_type = "test"
    called = False

    def __init__(self, **kwargs):
        super().__init__(produces_annotation=["lapps:NamedEntity"], **kwargs)
        TrackingAnnotator.called = False

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        TrackingAnnotator.called = True
        return [LIFAnnotation(id="ne0", type="NamedEntity", start=0, end=3)]


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


# Direct contract validation tests


class TestLanguageValidation:
    def test_language_match(self):
        doc = LIFDocument(
            text=LIFText(value="hello", language="lexvo:grc"),
        )
        contract = LIFContract(requires_language=["lexvo:grc"])
        assert validate_lif_contract(doc, contract) == []

    def test_language_mismatch(self):
        doc = LIFDocument(
            text=LIFText(value="hello", language="lexvo:eng"),
        )
        contract = LIFContract(requires_language=["lexvo:grc"])
        violations = validate_lif_contract(doc, contract)
        assert len(violations) == 1
        assert "language" in violations[0]

    def test_language_missing(self):
        doc = LIFDocument(
            text=LIFText(value="hello"),
        )
        contract = LIFContract(requires_language=["lexvo:grc"])
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


class TestFeatureMerging:
    """Test that returning annotations with existing ids merges features."""

    def test_pos_tagger_enriches_tokens(self, client):
        """A POS tagger adds features to tokens produced by a prior tokenizer."""
        annotators.register(TokenizerAnnotator())
        annotators.register(POSTaggerAnnotator())
        resp = client.post(
            "/annotate",
            json={
                "document": _lif_doc("hello world"),
                "annotators": ["tokenizer", "pos-tagger"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        view = data["views"][0]
        # Should still have exactly 2 token annotations, not 4
        assert len(view["annotations"]) == 2
        for ann in view["annotations"]:
            assert ann["@type"] == "Token"
            # Original word feature preserved, pos feature added
            assert "word" in ann["features"]
            assert ann["features"]["pos"] == "NN"
        # contains should record both Token and Token#pos
        contains = view["metadata"]["contains"]
        assert "Token" in contains
        assert "lapps:Token#pos" in contains


class TestStaticPreCheck:
    """The pipeline validates all contracts upfront before running annotators."""

    def test_later_failure_prevents_earlier_annotator_from_running(self, client):
        """If annotator B's contract is unsatisfiable, annotator A never runs."""
        annotators.register(TrackingAnnotator())
        annotators.register(ChainedAnnotator())  # requires NamedEntity
        annotators.register(POSTaggerAnnotator())  # requires Token

        # tracker produces NamedEntity, so chained is satisfied,
        # but pos-tagger requires Token which nobody produces → 422
        resp = client.post(
            "/annotate",
            json={
                "document": _lif_doc(),
                "annotators": ["tracker", "chained", "pos-tagger"],
            },
        )
        assert resp.status_code == 422
        assert "pos-tagger" in resp.json()["detail"]
        # tracker should NOT have been called
        assert not TrackingAnnotator.called

    def test_static_check_passes_valid_pipeline(self, client):
        """A valid pipeline passes the static check and runs all annotators."""
        annotators.register(TrackingAnnotator())
        annotators.register(ChainedAnnotator())  # requires NamedEntity

        resp = client.post(
            "/annotate",
            json={
                "document": _lif_doc(),
                "annotators": ["tracker", "chained"],
            },
        )
        assert resp.status_code == 200
        assert TrackingAnnotator.called
