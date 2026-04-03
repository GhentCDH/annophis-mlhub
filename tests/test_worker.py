import re

import pytest
from fastapi.testclient import TestClient

from annophis_mlhub.lif import LIFAnnotation
from annophis_mlhub.worker import ModelWorker, create_worker_app


class UpperCaseWorker(ModelWorker):
    """Test worker that marks uppercase words."""

    name = "upper"
    annotation_type = "test"

    def load(self):
        self.loaded = True

    def predict(self, text: str) -> list[LIFAnnotation]:
        return [
            LIFAnnotation(
                id=f"u{i}",
                type="UpperCase",
                start=m.start(),
                end=m.end(),
                features={"word": m.group()},
            )
            for i, m in enumerate(re.finditer(r"\b[A-Z]{2,}\b", text))
        ]


def _lif_doc(text):
    return {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": text},
    }


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
    assert data["rdfs:label"] == "upper"
    assert data["status"] == "ok"


def test_worker_info(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rdfs:label"] == "upper"
    assert data["@type"] == "annophis_mlhub:Annotator"


def test_worker_annotate(client):
    resp = client.post("/annotate", json=_lif_doc("hello WORLD and NASA"))
    assert resp.status_code == 200
    data = resp.json()
    words = [a["features"]["word"] for a in data["annotations"]]
    assert "WORLD" in words
    assert "NASA" in words


def test_worker_annotate_no_matches(client):
    resp = client.post("/annotate", json=_lif_doc("all lowercase here"))
    assert resp.status_code == 200
    assert resp.json()["annotations"] == []


def test_worker_ws_single(client):
    with client.websocket_connect("/annotate") as ws:
        ws.send_json({"id": "1", "document": _lif_doc("hello WORLD")})
        result = ws.receive_json()
    assert result["id"] == "1"
    assert result["annotator"] == "upper"
    assert any(a["features"]["word"] == "WORLD" for a in result["annotations"])


def test_worker_ws_multiple(client):
    with client.websocket_connect("/annotate") as ws:
        ws.send_json({"id": "1", "document": _lif_doc("HELLO there")})
        ws.send_json({"id": "2", "document": _lif_doc("lowercase only")})
        ws.send_json({"id": "3", "document": _lif_doc("NASA and ESA")})
        results = [ws.receive_json(), ws.receive_json(), ws.receive_json()]
    ids = {r["id"] for r in results}
    assert ids == {"1", "2", "3"}
    by_id = {r["id"]: r for r in results}
    assert any(a["features"]["word"] == "HELLO" for a in by_id["1"]["annotations"])
    assert by_id["2"]["annotations"] == []
    assert {a["features"]["word"] for a in by_id["3"]["annotations"]} == {"NASA", "ESA"}
