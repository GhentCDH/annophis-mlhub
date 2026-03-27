"""Tests that verify actual concurrent processing in the worker harness.

With max_workers=1 (default), N requests are serialized: total ~ N * delay.
With max_workers=N, requests run in parallel:       total ~ 1 * delay.
"""

import concurrent.futures
import logging
import time

from fastapi.testclient import TestClient

from annohub.models import Span
from annohub.worker import ModelWorker, create_worker_app

DELAY = 0.10
N = 10

logger = logging.getLogger(__name__)


class SlowWorker(ModelWorker):
    """Worker that sleeps for a fixed duration to simulate heavy inference."""

    name = "slow-worker"
    annotation_type = "test"

    def __init__(self, delay: float = DELAY):
        super().__init__()
        self.delay = delay

    def load(self) -> None:
        pass

    def predict(self, text: str) -> list[Span]:
        time.sleep(self.delay)
        return []


def _post(client: TestClient, text: str):
    return client.post("/annotate", json={"text": text})


def test_single_worker_serializes_requests():
    """max_workers=1: N concurrent requests take >= N * delay (serialized)."""
    app = create_worker_app(SlowWorker(), max_workers=1)

    with TestClient(app) as client:
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(_post, client, f"text {i}") for i in range(N)]
            responses = [f.result() for f in futs]
        elapsed = time.monotonic() - start

    for r in responses:
        assert r.status_code == 200

    # Allow 20 % margin; still well above the parallel floor
    assert elapsed >= DELAY * N * 0.8, (
        f"Expected serialized time >= {DELAY * N * 0.8:.2f}s, got {elapsed:.2f}s"
    )


def test_multi_worker_processes_concurrently():
    """max_workers=N: N concurrent requests finish in ~ 1 * delay (parallel)."""
    app = create_worker_app(SlowWorker(), max_workers=N)

    with TestClient(app) as client:
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(_post, client, f"text {i}") for i in range(N)]
            responses = [f.result() for f in futs]
        elapsed = time.monotonic() - start

    for r in responses:
        assert r.status_code == 200

    # Parallel execution: should finish close to single delay time
    startup = 0.50
    assert elapsed < startup + DELAY * 1.3, (
        f"Expected parallel time < {startup + DELAY * 1.3:.2f}s, got {elapsed:.2f}s"
    )
