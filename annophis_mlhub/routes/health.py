import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from annophis_mlhub import annotators
from annophis_mlhub.annotators.base import Annotator
from annophis_mlhub.models import Health

router = APIRouter()
logger = logging.getLogger(__name__)


async def _annotator_info(a: Annotator) -> dict[str, Any] | None:
    """Get JSON-LD descriptor for an annotator, or None if unavailable."""
    try:
        return await a.info()
    except Exception as e:
        logger.warning("Annotator %r unavailable: %s", a.name, e)
        return None


@router.get("/health", response_model=Health)
async def health():
    """Check the health of this server"""
    return {"status": "ok"}


@router.get("/annotators")
async def list_annotators():
    """List all available annotators as JSON-LD descriptors"""
    infos = await asyncio.gather(
        *(_annotator_info(a) for a in annotators.all().values())
    )
    result = [i for i in infos if i is not None]
    return JSONResponse(content=result, media_type="application/ld+json")


@router.get("/annotators/{name}")
async def get_annotator(name: str):
    """Get JSON-LD descriptor for a specific annotator"""
    a = annotators.get(name)
    if a is None:
        raise HTTPException(404, f"Unknown annotator: {name}")
    info = await _annotator_info(a)
    if info is None:
        raise HTTPException(503, f"Annotator {name!r} is not available")
    return JSONResponse(content=info, media_type="application/ld+json")
