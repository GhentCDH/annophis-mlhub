"""Tests that verify actual concurrent processing in the worker harness.

With max_workers=1 (default), N requests are serialized: total ~ N * delay.
With max_workers=N, requests run in parallel:       total ~ 1 * delay.
"""

import concurrent.futures
import logging
import time

from fastapi.testclient import TestClient

from annophis_mlhub.lif import LIFAnnotation, LIFDocument
from annophis_mlhub.worker import ModelWorker, create_worker_app

DELAY = 0.05
N = 5
logger = logging.getLogger(__name__)


class SlowWorker(ModelWorker):
    name = "slow"
    annotation_type = "test"

    def load(self):
        pass

    def predict(self, doc: LIFDocument) -> list[LIFAnnotation]:
        time.sleep(DELAY)
        return [LIFAnnotation(id="t0", type="Test", start=0, end=len(doc.text.value))]


def _lif_json(text="hello"):
    return {
        "@context": "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
        "text": {"@value": text},
    }


def _post(client_url_pair):
    client, url = client_url_pair
    return client.post(url, json=_lif_json())


def test_single_worker_serializes_requests():
    app = create_worker_app(SlowWorker(), max_workers=1)
    with TestClient(app) as client:
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_post, (client, "/annotate")) for _ in range(N)]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

    expected = N * DELAY * 0.8
    logger.info("single-worker elapsed=%.3fs (expected >= %.3fs)", elapsed, expected)
    assert all(r.status_code == 200 for r in results)
    assert elapsed >= expected


def test_multi_worker_processes_concurrently():
    app = create_worker_app(SlowWorker(), max_workers=N)
    with TestClient(app) as client:
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_post, (client, "/annotate")) for _ in range(N)]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

    expected = DELAY * 1.5
    logger.info("multi-worker elapsed=%.3fs (expected < %.3fs)", elapsed, expected)
    assert all(r.status_code == 200 for r in results)
    assert elapsed < expected
