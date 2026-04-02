from contextlib import asynccontextmanager

from fastapi import FastAPI

from annophis_mlhub.docs import add_scalar_docs
from annophis_mlhub.routes import annotate, health, vocab


@asynccontextmanager
async def lifespan(app: FastAPI):
    from annophis_mlhub import annotators
    from annophis_mlhub.config import load_annotators

    load_annotators()
    yield

    for ann in annotators.all().values():
        if hasattr(ann, "close"):
            await ann.close()  # ty:ignore[call-non-callable]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Annohub",
        description="Text annotation API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.include_router(health.router, tags=["system"])
    app.include_router(annotate.router, tags=["annotation"])
    app.include_router(vocab.router, tags=["vocabulary"])
    add_scalar_docs(app)
    return app


app = create_app()
