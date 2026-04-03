import asyncio
import contextlib
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from annophis_mlhub import annotators
from annophis_mlhub.annotators.base import Annotator
from annophis_mlhub.annotators.remote import RemoteAnnotator
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
    LIFDocument,
    LIFView,
    ViewMetadata,
    validate_lif_contract,
)
from annophis_mlhub.models import WsInputUnit, WsOutputUnit

router = APIRouter()
logger = logging.getLogger(__name__)


class AnnotateRequest(BaseModel):
    model_config = {"populate_by_name": True}

    document: LIFDocument
    annotators: list[str]


def _ensure_view(doc: LIFDocument) -> LIFDocument:
    """Ensure the document has at least one view."""
    if doc.views:
        return doc
    view = LIFView(id="v0", metadata=ViewMetadata(), annotations=[])
    return doc.model_copy(update={"views": [view]})


def _merge_annotations(
    doc: LIFDocument,
    annotations: list[LIFAnnotation],
    producer: str,
) -> LIFDocument:
    """Merge annotations into the document's single view."""
    view = doc.views[0]
    new_annotations = list(view.annotations) + annotations
    new_contains = dict(view.metadata.contains)
    # Add each distinct annotation @type to the contains metadata
    for ann in annotations:
        if ann.type not in new_contains:
            new_contains[ann.type] = ContainsEntry(producer=producer, type=ann.type)
    new_metadata = ViewMetadata(contains=new_contains)
    new_view = view.model_copy(
        update={"annotations": new_annotations, "metadata": new_metadata}
    )
    return doc.model_copy(update={"views": [new_view]})


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


@router.post("/annotate")
async def annotate(request: AnnotateRequest):
    """Annotate a LIF document through a sequential pipeline of annotators."""
    all_annotators = annotators.all()
    doc = _ensure_view(request.document)

    for name in request.annotators:
        ann = all_annotators.get(name)
        if ann is None:
            raise HTTPException(404, f"Unknown annotator: {name}")
        if not await _is_available(ann):
            raise HTTPException(503, f"Annotator {name!r} is not available")

        violations = validate_lif_contract(doc, ann.lif_contract)
        if violations:
            raise HTTPException(
                422, f"Annotator {name!r} contract violations: {violations}"
            )

        annotations = await ann.annotate(doc)
        doc = _merge_annotations(doc, annotations, ann.name)

    return doc.model_dump(by_alias=True, exclude_none=True)


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
            annotations = await ann.annotate(unit.document)
            await result_queue.put(
                WsOutputUnit(
                    id=unit.id,
                    annotator=ann.name,
                    annotations=annotations,
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
                await websocket.send_json(item.model_dump(by_alias=True))
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
