import asyncio
import contextlib
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from annohub import annotators
from annohub.annotators.base import Annotator
from annohub.annotators.remote import RemoteAnnotator
from annohub.models import (
    AnnotationResult,
    Contract,
    Document,
    WsInputUnit,
    WsOutputUnit,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class AnnotateRequest(BaseModel):
    document: Document
    annotators: list[str]


def validate_contract(doc: Document, contract: Contract) -> list[str]:
    """Check that all required keys exist in the document. Returns list of missing keys."""
    missing = []
    dump = doc.model_dump()
    for key in contract.requires:
        if key not in dump:
            missing.append(key)
    return missing


def merge_result(doc: Document, result: AnnotationResult) -> Document:
    """Create a new Document with the result's spans added under the annotation_type key."""
    data = doc.model_dump()
    data[result.annotation_type] = [s.model_dump() for s in result.spans]
    return Document(**data)


async def _is_available(a) -> bool:
    """Quick availability check — remote annotators are pinged, local ones always pass."""
    if not isinstance(a, RemoteAnnotator):
        return True
    try:
        await a.info()
        return True
    except Exception as e:
        logger.warning("Annotator %r is not available: %s", a.name, e)
        return False


@router.post("/annotate", response_model=Document)
async def annotate(request: AnnotateRequest) -> Document:
    """Annotate a document through a sequential pipeline of annotators."""
    all_annotators = annotators.all()
    doc = request.document

    for name in request.annotators:
        ann = all_annotators.get(name)
        if ann is None:
            raise HTTPException(404, f"Unknown annotator: {name}")
        if not await _is_available(ann):
            raise HTTPException(503, f"Annotator {name!r} is not available")

        missing = validate_contract(doc, ann.contract)
        if missing:
            raise HTTPException(
                422, f"Annotator {name!r} requires missing keys: {missing}"
            )

        result = await ann.annotate(doc)
        doc = merge_result(doc, result)

    return doc


@router.websocket("/annotate")
async def ws_annotate_hub(
    websocket: WebSocket,
    annotators_q: list[str] = Query(default=[], alias="annotators"),
):
    """Stream annotation results over WebSocket.

    Send WsInputUnit JSON messages; receive WsOutputUnit JSON messages as
    each annotator completes.  Pass ?annotators=name to limit to specific
    annotators; omit for all registered annotators.
    """
    all_anns = annotators.all()

    if annotators_q:
        targets = []
        for name in annotators_q:
            ann = all_anns.get(name)
            if ann is None:
                await websocket.accept()
                await websocket.send_json({"error": f"Unknown annotator: {name!r}"})
                await websocket.close()
                return
            targets.append(ann)
    else:
        targets = list(all_anns.values())

    remote_targets: list[RemoteAnnotator] = [
        a for a in targets if isinstance(a, RemoteAnnotator)
    ]
    local_targets: list[Annotator] = [
        a for a in targets if not isinstance(a, RemoteAnnotator)
    ]

    await websocket.accept()
    result_queue: asyncio.Queue[WsOutputUnit | None] = asyncio.Queue()
    local_tasks: set[asyncio.Task] = set()

    async def annotate_local(ann: Annotator, unit: WsInputUnit) -> None:
        try:
            result = await ann.annotate(unit.document)
            await result_queue.put(
                WsOutputUnit(
                    id=unit.id,
                    annotator=result.annotator,
                    annotation_type=result.annotation_type,
                    spans=result.spans,
                )
            )
        except Exception:
            pass

    async def run_reader(session) -> None:
        async for out in session:
            await result_queue.put(out)

    async def send() -> None:
        while True:
            item = await result_queue.get()
            if item is None:
                break
            try:
                await websocket.send_json(item.model_dump())
            except Exception:
                pass

    async with contextlib.AsyncExitStack() as stack:
        sessions = [
            await stack.enter_async_context(ann.stream_session())
            for ann in remote_targets
        ]
        reader_tasks = [asyncio.create_task(run_reader(s)) for s in sessions]
        send_task = asyncio.create_task(send())

        try:
            while True:
                data = await websocket.receive_json()
                unit = WsInputUnit(**data)
                for session in sessions:
                    await session.send(unit)
                for ann in local_targets:
                    task = asyncio.create_task(annotate_local(ann, unit))
                    local_tasks.add(task)
                    task.add_done_callback(local_tasks.discard)
        except WebSocketDisconnect:
            pass

    if reader_tasks:
        await asyncio.gather(*reader_tasks)
    if local_tasks:
        await asyncio.gather(*local_tasks)
    await result_queue.put(None)
    await send_task
