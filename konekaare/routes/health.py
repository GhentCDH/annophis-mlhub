from fastapi import APIRouter, HTTPException

from konekaare import annotators
from konekaare.annotators.base import Annotator
from konekaare.models import AnnotatorInfo, Health

router = APIRouter()


async def _annotator_info(a: Annotator) -> AnnotatorInfo:
    return await a.info()


@router.get("/health", response_model=Health)
async def health():
    """Check the health of this server"""
    return {"status": "ok"}


@router.get("/annotators", response_model=list[AnnotatorInfo])
async def list_annotators():
    """List annotators' info"""
    return [await _annotator_info(a) for a in annotators.all().values()]


@router.get("/annotators/{name}", response_model=AnnotatorInfo)
async def get_annotator(name: str):
    """Get annotator info for specific annotator"""
    a = annotators.get(name)
    if a is None:
        raise HTTPException(404, f"Unknown annotator: {name}")
    return await _annotator_info(a)
