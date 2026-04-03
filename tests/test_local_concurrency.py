import asyncio
import time

import pytest

from annophis_mlhub.annotators.local import LocalAnnotator
from annophis_mlhub.lif import LIFAnnotation, LIFDocument, LIFText


class SlowAnnotator(LocalAnnotator):
    name = "slow"
    annotation_type = "test"

    def __init__(self, delay: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        time.sleep(self.delay)
        return [LIFAnnotation(id="t0", type="Test", start=0, end=1)]


@pytest.mark.asyncio
async def test_semaphore_serializes_access():
    ann = SlowAnnotator(max_concurrency=1, delay=0.05)
    doc = LIFDocument(text=LIFText(value="hello"))

    start = time.monotonic()
    await asyncio.gather(*(ann.annotate(doc) for _ in range(3)))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.15 - 0.02  # 3 × 0.05s, serialized


@pytest.mark.asyncio
async def test_higher_concurrency_is_faster():
    ann = SlowAnnotator(max_concurrency=3, delay=0.05)
    doc = LIFDocument(text=LIFText(value="hello"))

    start = time.monotonic()
    await asyncio.gather(*(ann.annotate(doc) for _ in range(3)))
    elapsed = time.monotonic() - start

    assert elapsed < 0.15 - 0.02  # all 3 run in parallel
