import pytest

from konekaare import annotators
from konekaare.annotators.local import LocalAnnotator
from konekaare.models import AnnotationRequest, AnnotationResult, Span


class DummyAnnotator(LocalAnnotator):
    name = "dummy"
    annotation_type = "test"

    def annotate_sync(self, request: AnnotationRequest) -> AnnotationResult:
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[Span(start=0, end=5, label="TEST", text=request.text[:5])],
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


def test_annotate_no_annotators(client):
    resp = client.post("/annotate", json={"text": "hello world", "annotators": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert data["annotations"] == []


def test_annotate_with_dummy(client):
    annotators.register(DummyAnnotator())
    resp = client.post("/annotate", json={"text": "hello world", "annotators": ["dummy"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["annotator"] == "dummy"
    assert data["annotations"][0]["spans"][0]["label"] == "TEST"


def test_annotate_unknown_annotator(client):
    resp = client.post(
        "/annotate", json={"text": "hello", "annotators": ["nonexistent"]}
    )
    assert resp.status_code == 404


def test_ws_annotate_local(client):
    annotators.register(DummyAnnotator())
    with client.websocket_connect("/annotate?annotators=dummy") as ws:
        ws.send_json({"id": "a", "text": "hello world"})
        ws.send_json({"id": "b", "text": "foo bar"})
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
    """Test the dummy-ner annotator loaded from konekaare.toml."""
    resp = client.get("/annotators")
    data = resp.json()
    names = [a["name"] for a in data]
    assert "dummy-ner" in names

    resp = client.post(
        "/annotate", json={"text": "Alice went to Paris", "annotators": ["dummy-ner"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    ner = next(a for a in data["annotations"] if a["annotator"] == "dummy-ner")
    labels = [s["text"] for s in ner["spans"]]
    assert "Alice" in labels
    assert "Paris" in labels
