from fastapi import APIRouter

from konekaare import annotators
from konekaare.annotators.local import LocalAnnotator
from konekaare.models import AnnotatorInfo, Health

router = APIRouter()


@router.get("/health", response_model=Health)
async def health():
    return {"status": "ok"}


@router.get("/annotators", response_model=list[AnnotatorInfo])
async def list_annotators():
    return [
        AnnotatorInfo(
            name=a.name,
            annotation_type=a.annotation_type,
            kind="local" if isinstance(a, LocalAnnotator) else "remote",
        )
        for a in annotators.all().values()
    ]
