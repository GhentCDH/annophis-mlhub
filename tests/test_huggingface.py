import httpx
import pytest

from konekaare.annotators.huggingface import HuggingFaceAnnotator
from konekaare.models import AnnotationRequest


@pytest.fixture
def annotator():
    return HuggingFaceAnnotator(
        name="hf-ner",
        annotation_type="ner",
        model="dslim/bert-base-NER",
        token="hf_fake",
    )


@pytest.mark.asyncio
async def test_translates_hf_response(annotator, monkeypatch):
    """HF entities (entity_group, word) are translated to konekaare Spans (label, text)."""

    async def fake_post(self, url, *, json, headers, timeout):
        assert "/models/dslim/bert-base-NER" in url
        assert json == {"inputs": "Alice went to Paris"}
        assert headers["Authorization"] == "Bearer hf_fake"

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return [
                    {
                        "entity_group": "PER",
                        "score": 0.998,
                        "word": "Alice",
                        "start": 0,
                        "end": 5,
                    },
                    {
                        "entity_group": "LOC",
                        "score": 0.995,
                        "word": "Paris",
                        "start": 14,
                        "end": 19,
                    },
                ]

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await annotator.annotate(AnnotationRequest(text="Alice went to Paris"))

    assert result.annotator == "hf-ner"
    assert result.annotation_type == "ner"
    assert len(result.spans) == 2
    assert result.spans[0].label == "PER"
    assert result.spans[0].text == "Alice"
    assert result.spans[1].label == "LOC"
    assert result.spans[1].text == "Paris"

    await annotator.close()


@pytest.mark.asyncio
async def test_class_level_defaults():
    """Can set model/token as class attributes."""

    class MyHf(HuggingFaceAnnotator):
        name = "my-hf"
        annotation_type = "ner"
        model = "custom/model"
        token = "hf_custom"

    ann = MyHf()
    assert ann.name == "my-hf"
    assert ann.model == "custom/model"
    assert ann.token == "hf_custom"
    await ann.close()


@pytest.mark.asyncio
async def test_no_auth_header_without_token(monkeypatch):
    """No Authorization header is sent when token is empty."""

    ann = HuggingFaceAnnotator(name="hf-ner", annotation_type="ner")

    async def fake_post(self, url, *, json, headers, timeout):
        assert "Authorization" not in headers

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return []

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await ann.annotate(AnnotationRequest(text="hello"))
    assert result.spans == []

    await ann.close()
