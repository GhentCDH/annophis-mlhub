import pytest

from annohub import annotators
from annohub.annotators.sentences import SentenceAnnotator, SentenceCountAnnotator
from annohub.models import Document


@pytest.mark.asyncio
async def test_sentence_split():
    ann = SentenceAnnotator(name="sent", annotation_type="sentences")
    doc = Document(text="Hello world. This is a test. And another one.")
    result = await ann.annotate(doc)
    assert result.annotation_type == "sentences"
    assert len(result.spans) == 3
    assert result.spans[0].text == "Hello world."
    assert result.spans[1].text == "This is a test."
    assert result.spans[2].text == "And another one."
    for s in result.spans:
        assert s.label == "SENTENCE"
        assert doc.text[s.start : s.end] == s.text


@pytest.mark.asyncio
async def test_sentence_split_no_trailing_punctuation():
    ann = SentenceAnnotator(name="sent", annotation_type="sentences")
    doc = Document(text="First sentence. No punctuation here")
    result = await ann.annotate(doc)
    assert len(result.spans) == 2
    assert result.spans[0].text == "First sentence."
    assert result.spans[1].text == "No punctuation here"


@pytest.mark.asyncio
async def test_sentence_count():
    ann = SentenceCountAnnotator(
        name="count",
        annotation_type="sentence_count",
        requires={"text": True, "sentences": True},
        produces=["sentence_count"],
    )
    doc = Document(
        text="One. Two. Three.",
        sentences=[
            {"start": 0, "end": 4, "label": "SENTENCE", "text": "One."},
            {"start": 5, "end": 9, "label": "SENTENCE", "text": "Two."},
            {"start": 10, "end": 16, "label": "SENTENCE", "text": "Three."},
        ],
    )
    result = await ann.annotate(doc)
    assert result.annotation_type == "sentence_count"
    assert len(result.spans) == 1
    assert result.spans[0].label == "COUNT"
    assert result.spans[0].text == "3"


@pytest.mark.asyncio
async def test_sentence_count_empty():
    ann = SentenceCountAnnotator(
        name="count",
        annotation_type="sentence_count",
        requires={"text": True, "sentences": True},
    )
    doc = Document(text="hello", sentences=[])
    result = await ann.annotate(doc)
    assert result.spans[0].text == "0"


def test_pipeline_via_http(client):
    """sentence-split → sentence-count through the HTTP pipeline."""
    annotators.register(SentenceAnnotator(name="sent", annotation_type="sentences"))
    annotators.register(
        SentenceCountAnnotator(
            name="count",
            annotation_type="sentence_count",
            requires={"text": True, "sentences": True},
            produces=["sentence_count"],
        )
    )
    resp = client.post(
        "/annotate",
        json={
            "document": {"text": "Alice runs. Bob walks."},
            "annotators": ["sent", "count"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "sentences" in data
    assert len(data["sentences"]) == 2
    assert "sentence_count" in data
    assert data["sentence_count"][0]["text"] == "2"


def test_count_without_sentences_fails(client):
    """sentence-count alone should 422 because 'sentences' key is missing."""
    annotators.register(
        SentenceCountAnnotator(
            name="count",
            annotation_type="sentence_count",
            requires={"text": True, "sentences": True},
            produces=["sentence_count"],
        )
    )
    resp = client.post(
        "/annotate",
        json={
            "document": {"text": "Hello world."},
            "annotators": ["count"],
        },
    )
    assert resp.status_code == 422
    assert "sentences" in resp.json()["detail"]
