"""Worker app factory — turns a ModelWorker into a FastAPI service.

The app uses an internal asyncio.Queue to serialize inference requests
through a single background worker thread.  This is ideal for GPU models
that can't handle concurrent access.

Architecture::

    HTTP POST /annotate
         |
         v
    asyncio.Queue  -->  worker task (asyncio.to_thread)  -->  response Future
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from annophis_mlhub.models import (
    AnnotationResult,
    AnnotatorInfo,
    Document,
    WsInputUnit,
    WsOutputUnit,
)
from annophis_mlhub.worker.base import ModelWorker


class _WorkerInfo(AnnotatorInfo):
    status: str = "ok"


class _QueueItem:
    __slots__ = ("text", "future")

    def __init__(self, text: str, future: asyncio.Future[AnnotationResult]):
        self.text = text
        self.future = future


def create_worker_app(
    worker: ModelWorker,
    max_queue_size: int = 64,
    max_workers: int = 1,
) -> FastAPI:
    """Create a FastAPI app that serves a ModelWorker with an internal queue.

    ``max_workers`` controls how many worker threads run inference concurrently.
    The default of 1 is safe for GPU models; set higher for thread-safe CPU models
    that benefit from parallelism.
    """

    queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue(maxsize=max_queue_size)

    async def _process_queue() -> None:
        """Background task: pull items from queue, run inference in a thread."""
        while True:
            item = await queue.get()
            if item is None:
                break
            try:
                spans = await asyncio.to_thread(worker.predict, item.text)
                result = AnnotationResult(
                    annotator=worker.name,
                    annotation_type=worker.annotation_type,
                    spans=spans,
                )
                item.future.set_result(result)
            except Exception as exc:
                item.future.set_exception(exc)
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(worker.load)
        processors = [asyncio.create_task(_process_queue()) for _ in range(max_workers)]
        yield
        for _ in range(max_workers):
            await queue.put(None)
        await asyncio.gather(*processors)

    app = FastAPI(
        title=f"annophis_mlhub-worker: {worker.name}",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=_WorkerInfo)
    async def health():
        return _WorkerInfo(
            name=worker.name,
            annotation_type=worker.annotation_type,
            description=worker.description,
            kind="remote",
            contract=worker.contract,
        )

    @app.get("/info", response_model=_WorkerInfo)
    async def info():
        return _WorkerInfo(
            name=worker.name,
            annotation_type=worker.annotation_type,
            description=worker.description,
            kind="remote",
            contract=worker.contract,
        )

    @app.post("/annotate", response_model=AnnotationResult)
    async def annotate(doc: Document):
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AnnotationResult] = loop.create_future()
        await queue.put(_QueueItem(text=doc.text, future=future))
        return await future

    @app.websocket("/annotate")
    async def ws_annotate(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        result_queue: asyncio.Queue = asyncio.Queue()

        async def receive():
            try:
                while True:
                    data = await websocket.receive_json()
                    unit = WsInputUnit(**data)
                    future = loop.create_future()
                    future.add_done_callback(
                        lambda f, uid=unit.id: result_queue.put_nowait((uid, f))
                    )
                    await queue.put(_QueueItem(text=unit.document.text, future=future))
            except WebSocketDisconnect:
                pass
            finally:
                result_queue.put_nowait(None)

        async def send():
            while True:
                item = await result_queue.get()
                if item is None:
                    break
                uid, future = item
                try:
                    result = future.result()
                    await websocket.send_json(
                        WsOutputUnit(
                            id=uid,
                            annotator=result.annotator,
                            annotation_type=result.annotation_type,
                            spans=result.spans,
                        ).model_dump()
                    )
                except Exception as exc:
                    try:
                        await websocket.send_json({"id": uid, "error": str(exc)})
                    except Exception:
                        pass

        await asyncio.gather(receive(), send())

    return app
