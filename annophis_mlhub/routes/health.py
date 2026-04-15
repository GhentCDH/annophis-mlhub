import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pyld import jsonld

from annophis_mlhub import annotators
from annophis_mlhub.annotators.base import Annotator
from annophis_mlhub.annotators.descriptors import build_descriptor_context
from annophis_mlhub.models import Health

router = APIRouter()
logger = logging.getLogger(__name__)


async def _annotator_node(a: Annotator) -> dict[str, Any] | None:
    """Get JSON-LD graph node for an annotator, or None if unavailable.

    Strips ``@context`` from the node since the hub provides a shared
    top-level context in the ``@graph`` wrapper.
    """
    try:
        node = await a.info()
        node.pop("@context", None)
        return node
    except Exception as e:
        logger.warning("Annotator %r unavailable: %s", a.name, e)
        return None


@router.get("/health", response_model=Health)
async def health():
    """Check the health of this server"""
    return {"status": "ok"}


@router.get("/annotators")
async def list_annotators():
    """List all available annotators as a JSON-LD graph"""
    nodes = await asyncio.gather(
        *(_annotator_node(a) for a in annotators.all().values())
    )
    graph = [n for n in nodes if n is not None]
    doc = {
        "@context": build_descriptor_context(),
        "@graph": graph,
    }
    doc = jsonld.compact(doc, build_descriptor_context())
    return JSONResponse(content=doc, media_type="application/ld+json")


@router.get("/annotators/{name}")
async def get_annotator(name: str):
    """Get JSON-LD descriptor for a specific annotator"""
    a = annotators.get(name)
    if a is None:
        raise HTTPException(404, f"Unknown annotator: {name}")
    node = await _annotator_node(a)
    if node is None:
        raise HTTPException(503, f"Annotator {name!r} is not available")
    doc = jsonld.expand(
        {
            "@context": build_descriptor_context(),
            **node,
        }
    )[0]
    return JSONResponse(content=doc, media_type="application/ld+json")
