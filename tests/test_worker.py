import pytest
from fastapi.testclient import TestClient

from konekaare.models import Span
from konekaare.worker import ModelWorker, create_worker_app


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


def test_worker_info(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    assert resp.json()["name"] == "upper"


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
