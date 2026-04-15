from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument


class DummyAnnotator(LocalAnnotator):
    name = "dummy"
    annotation_type = "test"

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        return [
            LIFAnnotation(
                id="t0",
                type="Test",
                start=0,
                end=5,
                features={"word": doc.text.value[:5]},
            )
        ]


def _lif_doc(text="hello world"):
    return {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": text},
    }


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_annotators_empty(client):
    resp = client.get("/annotators")
    assert resp.status_code == 200
    data = resp.json()
    assert "@context" in data
    assert "rdfs:label" not in data
    assert "@graph" not in data


def test_list_annotators_with_dummy(client):
    annotators.register(DummyAnnotator())
    resp = client.get("/annotators")
    assert resp.status_code == 200
    data = resp.json()
    assert "@context" in data
    assert data["rdfs:label"] == "dummy"
    assert data["@type"] == "annophis_mlhub:Annotator"


def test_annotate_no_annotators(client):
    resp = client.post(
        "/annotate",
        json={"document": _lif_doc(), "annotators": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"]["@value"] == "hello world"


def test_annotate_with_dummy(client):
    annotators.register(DummyAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": _lif_doc(), "annotators": ["dummy"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"]["@value"] == "hello world"
    # Should have a view with annotations
    assert len(data["views"]) == 1
    view = data["views"][0]
    assert "Test" in view["metadata"]["contains"]
    assert len(view["annotations"]) == 1
    assert view["annotations"][0]["@type"] == "Test"


def test_annotate_unknown_annotator(client):
    resp = client.post(
        "/annotate",
        json={"document": _lif_doc(), "annotators": ["nonexistent"]},
    )
    assert resp.status_code == 404


def test_ws_annotate_local(client):
    annotators.register(DummyAnnotator())
    with client.websocket_connect("/annotate?annotators=dummy") as ws:
        ws.send_json({"id": "a", "document": _lif_doc()})
        ws.send_json({"id": "b", "document": _lif_doc("foo bar")})
        results = [ws.receive_json(), ws.receive_json()]
    ids = {r["id"] for r in results}
    assert ids == {"a", "b"}
    for r in results:
        assert r["annotator"] == "dummy"
        assert len(r["annotations"]) == 1


def test_ws_annotate_unknown(client):
    with client.websocket_connect("/annotate?annotators=nonexistent") as ws:
        msg = ws.receive_json()
    assert "error" in msg
    assert "nonexistent" in msg["error"]


def test_dummy_ner_from_config(_use_real_config, client):
    """Test the dummy-ner annotator loaded from mlhub.toml."""
    resp = client.get("/annotators")
    data = resp.json()
    names = [a.get("rdfs:label") for a in data["@graph"]]
    assert "dummy-ner" in names

    resp = client.post(
        "/annotate",
        json={
            "document": _lif_doc("Alice went to Paris."),
            "annotators": ["sentence-split", "dummy-ner"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    view = data["views"][0]
    ner_anns = [a for a in view["annotations"] if a["@type"] == "NamedEntity"]
    words = [a["features"]["word"] for a in ner_anns]
    assert "Alice" in words
    assert "Paris" in words
