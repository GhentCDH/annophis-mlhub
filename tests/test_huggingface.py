import httpx
import pytest

from annophis_mlhub.annotators.huggingface import HuggingFaceAnnotator
from annophis_mlhub.lif import LIFDocument, LIFText


@pytest.fixture
def annotator():
    return HuggingFaceAnnotator(
        name="hf-test",
        annotation_type="ner",
        model="dslim/bert-base-NER",
        token="hf_fake",
    )


@pytest.mark.asyncio
async def test_translates_hf_response(annotator):
    """HF entity_group/word/start/end → LIFAnnotation with features."""
    doc = LIFDocument(text=LIFText(value="Alice met Bob in Paris"))

    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json=[
                {
                    "entity_group": "PER",
                    "word": "Alice",
                    "start": 0,
                    "end": 5,
                    "score": 0.99,
                },
                {
                    "entity_group": "LOC",
                    "word": "Paris",
                    "start": 17,
                    "end": 22,
                    "score": 0.98,
                },
            ],
        )
    )
    annotator._client = httpx.AsyncClient(
        transport=transport, base_url=annotator.base_url
    )

    annotations = await annotator.annotate(doc)
    assert len(annotations) == 2
    assert annotations[0].type == "NamedEntity"
    assert annotations[0].features["category"] == "PER"
    assert annotations[0].features["word"] == "Alice"
    assert annotations[1].features["category"] == "LOC"


@pytest.mark.asyncio
async def test_class_level_defaults():
    class MyHf(HuggingFaceAnnotator):
        name = "my-hf"
        annotation_type = "ner"
        model = "custom/model"
        token = "hf_test"

    ann = MyHf()
    assert ann.model == "custom/model"
    assert ann.token == "hf_test"


@pytest.mark.asyncio
async def test_no_auth_header_without_token():
    ann = HuggingFaceAnnotator(name="no-token", annotation_type="ner", token="")
    doc = LIFDocument(text=LIFText(value="test"))

    captured_headers = {}

    def handler(req):
        captured_headers.update(dict(req.headers))
        return httpx.Response(200, json=[])

    ann._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=ann.base_url
    )
    await ann.annotate(doc)
    assert "authorization" not in captured_headers
