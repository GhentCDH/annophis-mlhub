from fastapi import APIRouter
from fastapi.responses import JSONResponse

from annohub.vocab import VOCABULARY

router = APIRouter()


@router.get("/vocab", summary="Annohub JSON-LD vocabulary")
async def get_vocabulary():
    """Return the annohub vocabulary as a JSON-LD document."""
    return JSONResponse(content=VOCABULARY, media_type="application/ld+json")
