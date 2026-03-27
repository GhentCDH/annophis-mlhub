import json

import httpx
import pytest
import websockets

from annohub.annotators.remote import GenericRemoteAnnotator
from annohub.models import Document, WsInputUnit


@pytest.fixture
def annotator():
    return GenericRemoteAnnotator(
        name="test-remote",
        annotation_type="ner",
        base_url="http://fake-host:9999",
    )


@pytest.mark.asyncio
async def test_generic_remote_annotator(annotator, monkeypatch):
    """GenericRemoteAnnotator posts full document and parses the response."""

    async def fake_post(self, url, *, json, timeout):
        assert url == "/annotate"
        assert json["text"] == "hello"

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "annotator": "test-remote",
                    "annotation_type": "ner",
                    "spans": [
                        {"start": 0, "end": 5, "label": "GREETING", "text": "hello"}
                    ],
                }

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    doc = Document(text="hello")
    result = await annotator.annotate(doc)

    assert result.annotator == "test-remote"
    assert result.annotation_type == "ner"
    assert len(result.spans) == 1
    assert result.spans[0].label == "GREETING"

    await annotator.close()


@pytest.mark.asyncio
async def test_stream_session(annotator, monkeypatch):
    """stream_session() opens a WS connection and yields a session that can send/receive."""

    outgoing = []
    incoming = [
        json.dumps(
            {"id": "42", "annotator": "test-remote", "annotation_type": "ner", "spans": []}
        )
    ]

    class FakeWs:
        async def send(self, data):
            outgoing.append(data)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if incoming:
                return incoming.pop(0)
            raise StopAsyncIteration

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(websockets, "connect", lambda url: FakeWs())

    unit = WsInputUnit(id="42", document=Document(text="hello"))
    async with annotator.stream_session() as session:
        await session.send(unit)
        results = [r async for r in session]

    assert len(outgoing) == 1
    sent = json.loads(outgoing[0])
    assert sent["id"] == "42"
    assert sent["document"]["text"] == "hello"
    assert len(results) == 1
    assert results[0].id == "42"
    assert results[0].annotator == "test-remote"


@pytest.mark.asyncio
async def test_custom_endpoint():
    """GenericRemoteAnnotator respects custom endpoint."""
    ann = GenericRemoteAnnotator(
        name="custom",
        annotation_type="pos",
        base_url="http://localhost:1234",
        endpoint="/custom/predict",
        timeout=5.0,
    )
    assert ann.endpoint == "/custom/predict"
    assert ann.timeout == 5.0
    assert ann.contract.requires == {"text": True}
    assert ann.contract.produces == ["pos"]
    await ann.close()
