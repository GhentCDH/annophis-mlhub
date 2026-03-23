import asyncio
import contextlib
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from annohub import annotators
from annohub.annotators.base import Annotator
from annohub.annotators.remote import RemoteAnnotator
from annohub.models import (
    AnnotationLayer,
    AnnotationRequest,
    AnnotationResponse,
    WsInputUnit,
    WsOutputUnit,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest) -> AnnotationResponse:
    """Annotate a piece of text, with all or a subset of annotators"""
    all_annotators = annotators.all()

    targets: list[Annotator] = []
    for name in request.annotators:
        ann = all_annotators.get(name)
        if ann is None:
            raise HTTPException(404, f"Unknown annotator: {name}")
        if not await _is_available(ann):
            raise HTTPException(503, f"Annotator {name!r} is not available")
        targets.append(ann)

    results = await asyncio.gather(*(a.annotate(request) for a in targets))

    return AnnotationResponse(
        text=request.text,
        annotations=[
            AnnotationLayer(
                annotator=r.annotator,
                annotation_type=r.annotation_type,
                spans=r.spans,
            )
            for r in results
        ],
    )


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
            req = AnnotationRequest(text=unit.text, annotators=[ann.name])
            result = await ann.annotate(req)
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
                # await remote annotators' answers
                for session in sessions:
                    await session.send(unit)
                # run local annotators
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
