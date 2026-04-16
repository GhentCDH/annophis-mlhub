import asyncio
import contextlib
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from annophis_mlhub import annotators
from annophis_mlhub.annotators.base import Annotator, is_streamable
from annophis_mlhub.annotators.descriptors import annotator_uri
from annophis_mlhub.annotators.session import StreamSession
from annophis_mlhub.cache import CachePlan
from annophis_mlhub.lif import (
    LIFDocument,
    LIFView,
    ViewMetadata,
    apply_contract_to_metadata,
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


async def _is_available(a: Annotator) -> bool:
    """Quick availability check — remote annotators are pinged, local ones always pass."""
    if not getattr(a, "base_url", None):
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

    pipeline: list[Annotator] = []
    projected = doc
    for name in request.annotators:
        ann = all_annotators.get(name)
        if ann is None:
            raise HTTPException(404, f"Unknown annotator: {name}")

        # Sync contract from remote worker before validation
        if getattr(ann, "base_url", None):
            try:
                await ann.info()
            except Exception:
                pass  # availability is checked later

        violations = validate_lif_contract(projected, ann.lif_contract)
        if violations:
            raise HTTPException(
                422, f"Annotator {name!r} contract violations: {violations}"
            )
        projected = apply_contract_to_metadata(
            projected, ann.lif_contract, annotator_uri(ann.name)
        )
        pipeline.append(ann)

    for ann in pipeline:
        if not await _is_available(ann):
            raise HTTPException(503, f"Annotator {ann.name!r} is not available")

        producer = annotator_uri(ann.name)
        plan = CachePlan.compute(doc, producer, ann.lif_contract)
        if plan.skip_entirely:
            logger.debug("Cache hit for %s, skipping", ann.name)
            continue

        try:
            doc = await plan.execute(ann)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    return doc.jsonld()


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

    targets = targets

    streaming_targets = [a for a in targets if is_streamable(a)]
    non_streaming_targets = [a for a in targets if not is_streamable(a)]

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
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Local annotator %r failed on unit %s", ann.name, unit.id)

    async def run_reader(session: StreamSession) -> None:
        async for out in session:
            await result_queue.put(out)

    async def send() -> None:
        while True:
            item = await result_queue.get()
            if item is None:
                break
            try:
                await websocket.send_json(item.model_dump(by_alias=True))
            except (WebSocketDisconnect, RuntimeError):
                logger.debug("WebSocket disconnected during send, stopping")
                break

    async with contextlib.AsyncExitStack() as stack:
        sessions: list[StreamSession] = [
            await stack.enter_async_context(ann.stream_session())  # ty:ignore[unresolved-attribute]
            for ann in streaming_targets
        ]
        reader_tasks = [asyncio.create_task(run_reader(s)) for s in sessions]
        send_task = asyncio.create_task(send())

        try:
            while True:
                data = await websocket.receive_json()
                unit = WsInputUnit(**data)
                for session in sessions:
                    await session.send(unit)
                for ann in non_streaming_targets:
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
