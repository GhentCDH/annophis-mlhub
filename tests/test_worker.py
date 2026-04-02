import pytest
from fastapi.testclient import TestClient

from annophis_mlhub.models import Span
from annophis_mlhub.worker import ModelWorker, create_worker_app


class UpperCaseWorker(ModelWorker):
    """Test worker that marks uppercase words."""

    name = "upper"
    annotation_type = "test"

    def load(self):
        self.loaded = True

    def predict(self, text: str) -> list[Span]:
        import re

        spans = []
        for m in re.finditer(r"\b[A-Z]{2,}\b", text):
            spans.append(
                Span(start=m.start(), end=m.end(), label="UPPER", text=m.group())
            )
        return spans


@pytest.fixture
def worker():
    return UpperCaseWorker()


@pytest.fixture
def client(worker):
    app = create_worker_app(worker)
    with TestClient(app) as c:
        yield c


def test_worker_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "upper"
    assert data["annotation_type"] == "test"
    assert data["status"] == "ok"
    assert "contract" in data


def test_worker_info(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "upper"
    assert data["contract"]["requires"] == {"text": True}
    assert data["contract"]["produces"] == ["test"]


def test_worker_annotate(client):
    resp = client.post("/annotate", json={"text": "hello WORLD and NASA"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["annotator"] == "upper"
    assert data["annotation_type"] == "test"
    labels = [s["text"] for s in data["spans"]]
    assert "WORLD" in labels
    assert "NASA" in labels


def test_worker_annotate_no_matches(client):
    resp = client.post("/annotate", json={"text": "all lowercase here"})
    assert resp.status_code == 200
    assert resp.json()["spans"] == []


def test_worker_ws_single(client):
    with client.websocket_connect("/annotate") as ws:
        ws.send_json({"id": "1", "document": {"text": "hello WORLD"}})
        result = ws.receive_json()
    assert result["id"] == "1"
    assert result["annotator"] == "upper"
    assert result["annotation_type"] == "test"
    assert any(s["text"] == "WORLD" for s in result["spans"])


def test_worker_ws_multiple(client):
    with client.websocket_connect("/annotate") as ws:
        ws.send_json({"id": "1", "document": {"text": "HELLO there"}})
        ws.send_json({"id": "2", "document": {"text": "lowercase only"}})
        ws.send_json({"id": "3", "document": {"text": "NASA and ESA"}})
        results = [ws.receive_json(), ws.receive_json(), ws.receive_json()]
    ids = {r["id"] for r in results}
    assert ids == {"1", "2", "3"}
    by_id = {r["id"]: r for r in results}
    assert any(s["text"] == "HELLO" for s in by_id["1"]["spans"])
    assert by_id["2"]["spans"] == []
    assert {s["text"] for s in by_id["3"]["spans"]} == {"NASA", "ESA"}
