from contextlib import asynccontextmanager

from fastapi import FastAPI

from konekaare.docs import add_scalar_docs
from konekaare.routes import annotate, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    from konekaare import annotators
    from konekaare.config import load_annotators

    load_annotators()
    yield

    for ann in annotators.all().values():
        if hasattr(ann, "close"):
            await ann.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Konekaare",
        description="Text annotation API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.include_router(health.router, tags=["system"])
    app.include_router(annotate.router, tags=["annotation"])
    add_scalar_docs(app)
    return app


app = create_app()
