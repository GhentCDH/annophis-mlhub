import asyncio
import contextlib
import logging
from typing import cast

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from annophis_mlhub import annotators
from annophis_mlhub.annotators.base import Annotator, is_streamable
from annophis_mlhub.annotators.descriptors import annotator_uri
from annophis_mlhub.annotators.session import StreamSession
from annophis_mlhub.cache import (
    build_filtered_document,
    compute_cache_plan,
    remove_stale_annotations,
    stamp_annotations,
)
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
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


def _merge_annotations(
    doc: LIFDocument,
    annotations: list[LIFAnnotation],
    producer: str,
    produces_feature: list[str] | None = None,
) -> LIFDocument:
    """Merge annotations into the document's single view.

    If an incoming annotation has the same ``id`` as an existing one,
    its features are merged into the existing annotation (new features
    are added, existing features are overwritten).  This allows an
    annotator to enrich tokens produced by a prior annotator (e.g. add
    a ``pos`` feature to existing Token annotations).

    Annotations with new ids are appended as usual.

    ``produces_feature`` entries (e.g. ``["lapps:Token#pos"]``) are
    recorded in ``metadata.contains`` so downstream annotators can
    check for them via contract validation.
    """
    view = doc.views[0]

    # Index existing annotations by id for fast lookup
    existing_by_id: dict[str, int] = {
        a.id: idx for idx, a in enumerate(view.annotations)
    }
    merged = list(view.annotations)

    for ann in annotations:
        if ann.id in existing_by_id:
            # Merge features into the existing annotation
            idx = existing_by_id[ann.id]
            old = merged[idx]
            merged_features = {**old.features, **ann.features}
            merged[idx] = old.model_copy(update={"features": merged_features})
        else:
            existing_by_id[ann.id] = len(merged)
            merged.append(ann)

    new_contains = dict(view.metadata.contains)
    for ann in annotations:
        if ann.type not in new_contains:
            new_contains[ann.type] = ContainsEntry(producer=producer, type=ann.type)
    for feat in produces_feature or []:
        if feat not in new_contains:
            new_contains[feat] = ContainsEntry(producer=producer, type=feat)
    new_metadata = ViewMetadata(contains=new_contains)
    new_view = view.model_copy(update={"annotations": merged, "metadata": new_metadata})
    return doc.model_copy(update={"views": [new_view]})


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

    # ── Static pre-check: validate the full pipeline before running anything ──
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

    # ── Execute: all contracts are satisfiable, run the pipeline ──────────
    for ann in pipeline:
        if not await _is_available(ann):
            raise HTTPException(503, f"Annotator {ann.name!r} is not available")

        producer = annotator_uri(ann.name)
        plan = compute_cache_plan(doc, producer, ann.lif_contract)
        print(plan)
        if plan.skip_entirely:
            logger.debug("Cache hit for %s, skipping", ann.name)
            continue

        doc = remove_stale_annotations(doc, producer, plan)

        if ann.lif_contract.input_granularity and plan.miss_spans:
            run_doc = build_filtered_document(doc, plan.miss_spans, ann.lif_contract)
        else:
            run_doc = doc

        annotations = await ann.annotate(run_doc)
        annotations = stamp_annotations(annotations, producer, ann.lif_contract, doc)
        doc = _merge_annotations(
            doc, annotations, producer, ann.lif_contract.produces_feature
        )

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

    targets = cast(list[Annotator], targets)

    streaming_targets = cast(list[Annotator], [a for a in targets if is_streamable(a)])
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
