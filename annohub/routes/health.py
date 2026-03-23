import asyncio
import logging

from fastapi import APIRouter, HTTPException

from annohub import annotators
from annohub.annotators.base import Annotator
from annohub.models import AnnotatorInfo, Health

router = APIRouter()
logger = logging.getLogger(__name__)


async def _annotator_info(a: Annotator) -> AnnotatorInfo:
    try:
        return await a.info()
    except Exception as e:
        logger.warning("Annotator %r unavailable: %s", a.name, e)
        return AnnotatorInfo(
            name=a.name,
            annotation_type=a.annotation_type,
            kind="local" if hasattr(a, "annotate_sync") else "remote",
            description=getattr(a, "description", ""),
            labels=getattr(a, "labels", []),
            available=False,
        )


@router.get("/health", response_model=Health)
async def health():
    """Check the health of this server"""
    return {"status": "ok"}


@router.get("/annotators", response_model=list[AnnotatorInfo])
async def list_annotators():
    """List all available annotators"""
    infos = await asyncio.gather(*(_annotator_info(a) for a in annotators.all().values()))
    return [i for i in infos if i.available]


@router.get("/annotators/{name}", response_model=AnnotatorInfo)
async def get_annotator(name: str):
    """Get info for a specific annotator"""
    a = annotators.get(name)
    if a is None:
        raise HTTPException(404, f"Unknown annotator: {name}")
    return await _annotator_info(a)
