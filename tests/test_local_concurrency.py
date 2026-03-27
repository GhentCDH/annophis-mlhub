import asyncio
import time

import pytest

from annohub.annotators.local import LocalAnnotator
from annohub.models import AnnotationResult, Document


class SlowAnnotator(LocalAnnotator):
    """Annotator that sleeps to simulate slow inference."""

    name = "slow"
    annotation_type = "test"

    def __init__(self, delay: float = 0.1, max_concurrency: int = 1):
        super().__init__(max_concurrency=max_concurrency)
        self.delay = delay

    def annotate_sync(self, doc: Document) -> AnnotationResult:
        time.sleep(self.delay)
        return AnnotationResult(
            annotator=self.name,
            annotation_type=self.annotation_type,
            spans=[],
        )


@pytest.mark.asyncio
async def test_semaphore_serializes_access():
    """With max_concurrency=1, requests should be serialized."""
    ann = SlowAnnotator(delay=0.05, max_concurrency=1)
    doc = Document(text="test")

    start = time.monotonic()
    await asyncio.gather(*(ann.annotate(doc) for _ in range(3)))
    elapsed = time.monotonic() - start

    # 3 serial calls of 0.05s each = ~0.15s minimum
    assert elapsed >= 0.12


@pytest.mark.asyncio
async def test_higher_concurrency_is_faster():
    """With max_concurrency=3, all requests can run in parallel."""
    ann = SlowAnnotator(delay=0.05, max_concurrency=3)
    doc = Document(text="test")

    start = time.monotonic()
    await asyncio.gather(*(ann.annotate(doc) for _ in range(3)))
    elapsed = time.monotonic() - start

    # 3 parallel calls of 0.05s = ~0.05s, definitely under 0.12s
    assert elapsed < 0.12
