import httpx
import pytest

from konekaare.annotators.remote import GenericRemoteAnnotator
from konekaare.models import AnnotationRequest


@pytest.fixture
def annotator():
    return GenericRemoteAnnotator(
        name="test-remote",
        annotation_type="ner",
        base_url="http://fake-host:9999",
    )


@pytest.mark.asyncio
async def test_generic_remote_annotator(annotator, monkeypatch):
    """GenericRemoteAnnotator posts to /annotate and parses the response."""

    async def fake_post(self, url, *, json, timeout):
        assert url == "/annotate"
        assert json == {"text": "hello"}

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

    req = AnnotationRequest(text="hello")
    result = await annotator.annotate(req)

    assert result.annotator == "test-remote"
    assert result.annotation_type == "ner"
    assert len(result.spans) == 1
    assert result.spans[0].label == "GREETING"

    await annotator.close()


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
    await ann.close()
