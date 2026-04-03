import json

import httpx
import pytest
import websockets

from annophis_mlhub.annotators.remote import GenericRemoteAnnotator
from annophis_mlhub.lif import LIFDocument, LIFText
from annophis_mlhub.models import WsInputUnit


@pytest.fixture
def annotator():
    return GenericRemoteAnnotator(
        name="remote-test",
        annotation_type="ner",
        base_url="http://fake-host:9999",
    )


@pytest.mark.asyncio
async def test_generic_remote_annotator(annotator):
    """GenericRemoteAnnotator sends the LIF document and parses annotations."""
    doc = LIFDocument(text=LIFText(value="Hello World"))

    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "annotations": [
                    {
                        "id": "ne0",
                        "@type": "NamedEntity",
                        "start": 0,
                        "end": 5,
                        "features": {"word": "Hello"},
                    }
                ]
            },
        )
    )
    annotator._client = httpx.AsyncClient(
        transport=transport, base_url=annotator.base_url
    )

    annotations = await annotator.annotate(doc)
    assert len(annotations) == 1
    assert annotations[0].type == "NamedEntity"
    assert annotations[0].start == 0
    assert annotations[0].end == 5


class FakeWs:
    """Minimal fake websockets connection for testing stream_session."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._responses)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_stream_session(annotator, monkeypatch):
    """stream_session sends WsInputUnit and yields WsOutputUnit."""
    ws_response = json.dumps(
        {
            "id": "1",
            "annotator": "remote-test",
            "annotations": [
                {"id": "ne0", "@type": "NamedEntity", "start": 0, "end": 5}
            ],
        }
    )
    fake_ws = FakeWs([ws_response])
    monkeypatch.setattr(websockets, "connect", lambda url: fake_ws)

    async with annotator.stream_session() as session:
        unit = WsInputUnit(
            id="1",
            document=LIFDocument(text=LIFText(value="Hello")),
        )
        await session.send(unit)
        results = [r async for r in session]

    assert len(results) == 1
    assert results[0].id == "1"
    assert len(results[0].annotations) == 1


def test_custom_endpoint():
    ann = GenericRemoteAnnotator(
        name="custom",
        annotation_type="pos",
        base_url="http://localhost:1234",
        endpoint="/custom",
        timeout=10.0,
    )
    assert ann.endpoint == "/custom"
    assert ann.timeout == 10.0
    assert ann.lif_contract.produces_annotation == []
