from fastapi import APIRouter, HTTPException

from konekaare import annotators
from konekaare.annotators.local import LocalAnnotator
from konekaare.models import AnnotatorInfo, Health

router = APIRouter()


def _annotator_info(a) -> AnnotatorInfo:
    return AnnotatorInfo(
        name=a.name,
        annotation_type=a.annotation_type,
        kind="local" if isinstance(a, LocalAnnotator) else "remote",
        description=getattr(a, "description", ""),
        labels=getattr(a, "labels", []),
    )


@router.get("/health", response_model=Health)
async def health():
    return {"status": "ok"}


@router.get("/annotators", response_model=list[AnnotatorInfo])
async def list_annotators():
    return [_annotator_info(a) for a in annotators.all().values()]


@router.get("/annotators/{name}", response_model=AnnotatorInfo)
async def get_annotator(name: str):
    a = annotators.get(name)
    if a is None:
        raise HTTPException(404, f"Unknown annotator: {name}")
    return _annotator_info(a)
