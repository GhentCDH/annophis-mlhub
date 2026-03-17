import asyncio
import logging

from fastapi import APIRouter, HTTPException

from konekaare import annotators
from konekaare.annotators.remote import RemoteAnnotator
from konekaare.models import AnnotationLayer, AnnotationRequest, AnnotationResponse

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

    targets = []
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
