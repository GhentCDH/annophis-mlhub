from annophis_mlhub import annotators
from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.models import AnnotationResult, Document, Span


class DummyAnnotator(LocalAnnotator):
    name = "dummy"
    annotation_type = "test"

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[Span(start=0, end=5, label="TEST", text=doc.text[:5])],
        )


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_annotators_empty(client):
    resp = client.get("/annotators")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_annotators_with_dummy(client):
    annotators.register(DummyAnnotator())
    resp = client.get("/annotators")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "dummy"
    assert data[0]["kind"] == "local"
    assert "contract" in data[0]


def test_annotate_no_annotators(client):
    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello world"}, "annotators": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"


def test_annotate_with_dummy(client):
    annotators.register(DummyAnnotator())
    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello world"}, "annotators": ["dummy"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert "test" in data
    assert data["test"][0]["label"] == "TEST"


def test_annotate_unknown_annotator(client):
    resp = client.post(
        "/annotate",
        json={"document": {"text": "hello"}, "annotators": ["nonexistent"]},
    )
    assert resp.status_code == 404


def test_ws_annotate_local(client):
    annotators.register(DummyAnnotator())
    with client.websocket_connect("/annotate?annotators=dummy") as ws:
        ws.send_json({"id": "a", "document": {"text": "hello world"}})
        ws.send_json({"id": "b", "document": {"text": "foo bar"}})
        results = [ws.receive_json(), ws.receive_json()]
    ids = {r["id"] for r in results}
    assert ids == {"a", "b"}
    for r in results:
        assert r["annotator"] == "dummy"
        assert r["annotation_type"] == "test"
        assert len(r["spans"]) == 1


def test_ws_annotate_unknown(client):
    with client.websocket_connect("/annotate?annotators=nonexistent") as ws:
        msg = ws.receive_json()
    assert "error" in msg
    assert "nonexistent" in msg["error"]


def test_dummy_ner_from_config(_use_real_config, client):
    """Test the dummy-ner annotator loaded from annophis_mlhub.toml."""
    resp = client.get("/annotators")
    data = resp.json()
    names = [a["name"] for a in data]
    assert "dummy-ner" in names

    resp = client.post(
        "/annotate",
        json={
            "document": {"text": "Alice went to Paris"},
            "annotators": ["dummy-ner"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ner" in data
    labels = [s["text"] for s in data["ner"]]
    assert "Alice" in labels
    assert "Paris" in labels
