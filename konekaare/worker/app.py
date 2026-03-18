"""Worker app factory — turns a ModelWorker into a FastAPI service.

The app uses an internal asyncio.Queue to serialize inference requests
through a single background worker thread.  This is ideal for GPU models
that can't handle concurrent access.

Architecture::

    HTTP POST /annotate
         |
         v
    asyncio.Queue  ──>  worker task (asyncio.to_thread)  ──>  response Future
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from konekaare.models import AnnotationResult, AnnotatorInfo, WsInputUnit, WsOutputUnit
from konekaare.worker.base import ModelWorker


class _WorkRequest(BaseModel):
    text: str


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
) -> FastAPI:
    """Create a FastAPI app that serves a ModelWorker with an internal queue."""

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
        # Load model
        await asyncio.to_thread(worker.load)
        # Start queue processor
        processor = asyncio.create_task(_process_queue())
        yield
        # Shutdown: send sentinel and wait
        await queue.put(None)
        await processor

    app = FastAPI(
        title=f"konekaare-worker: {worker.name}",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=_WorkerInfo)
    async def health():
        return _WorkerInfo(
            name=worker.name,
            annotation_type=worker.annotation_type,
            description=worker.description,
            labels=worker.labels,
            kind="remote",
        )

    @app.get("/info", response_model=_WorkerInfo)
    async def info():
        return _WorkerInfo(
            name=worker.name,
            annotation_type=worker.annotation_type,
            description=worker.description,
            labels=worker.labels,
            kind="remote",
        )

    @app.post("/annotate", response_model=AnnotationResult)
    async def annotate(req: _WorkRequest):
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AnnotationResult] = loop.create_future()
        await queue.put(_QueueItem(text=req.text, future=future))
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
                    await queue.put(_QueueItem(text=unit.text, future=future))
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
