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

from annophis_mlhub.annotators.descriptors import (
    build_descriptor_context,
    build_descriptor_node,
)
from annophis_mlhub.lif import LIFAnnotation, LIFDocument
from annophis_mlhub.models import WsInputUnit, WsOutputUnit
from annophis_mlhub.worker.base import ModelWorker


class _QueueItem:
    __slots__ = ("doc", "future")

    def __init__(self, doc: LIFDocument, future: asyncio.Future[list[LIFAnnotation]]):
        self.doc = doc
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
                annotations = await asyncio.to_thread(worker.predict, item.doc)
                item.future.set_result(annotations)
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

    @app.get("/health")
    async def health():
        desc = {**build_descriptor_node(worker), "@context": build_descriptor_context()}
        desc["status"] = "ok"
        return desc

    @app.get("/info")
    async def info():
        return {**build_descriptor_node(worker), "@context": build_descriptor_context()}

    @app.post("/annotate")
    async def annotate(doc: LIFDocument):
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[LIFAnnotation]] = loop.create_future()
        await queue.put(_QueueItem(doc=doc, future=future))
        annotations = await future
        return {"annotations": [a.model_dump(by_alias=True) for a in annotations]}

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
                    await queue.put(_QueueItem(doc=unit.document, future=future))
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
                    annotations = future.result()
                    await websocket.send_json(
                        WsOutputUnit(
                            id=uid,
                            annotator=worker.name,
                            annotations=annotations,
                        ).model_dump(by_alias=True)
                    )
                except Exception as exc:
                    try:
                        await websocket.send_json({"id": uid, "error": str(exc)})
                    except Exception:
                        pass

        await asyncio.gather(receive(), send())

    return app
