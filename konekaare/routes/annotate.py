import asyncio

from fastapi import APIRouter, HTTPException

from konekaare import annotators
from konekaare.models import AnnotationLayer, AnnotationRequest, AnnotationResponse

router = APIRouter()


@router.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest) -> AnnotationResponse:
    available = annotators.all()
    names = request.annotators or list(available.keys())

    targets = []
    for name in names:
        ann = available.get(name)
        if ann is None:
            raise HTTPException(404, f"Unknown annotator: {name}")
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
