# Annohub

Text annotation web API. Submit text, get annotations back (POS, NER, or anything).

## Stack

- **FastAPI** — API framework, OpenAPI compatible
- **Scalar** — interactive API docs at `/docs` with custom CSS
- **Pydantic / pydantic-settings** — validation and config
- **httpx** — async HTTP client for remote annotators
- **uvicorn** — ASGI server
- **Python 3.14**

## Project structure

```
annophis-mlhub/
├── app.py                     # app factory + lifespan
├── config.py                  # pydantic-settings + annotator loading from TOML
├── docs.py                    # Scalar docs route + custom CSS
├── models.py                  # Pydantic request/response schemas
├── annotators/
│   ├── __init__.py            # registry: register(), get(), all(), clear()
│   ├── base.py                # Annotator Protocol
│   ├── local.py               # LocalAnnotator ABC (semaphore-bounded concurrency)
│   ├── remote.py              # RemoteAnnotator ABC + GenericRemoteAnnotator
│   └── dummy.py               # example: regex-based NER annotator
├── worker/
│   ├── __init__.py            # public API: ModelWorker, Span, create_worker_app
│   ├── base.py                # ModelWorker ABC (implement load + predict)
│   ├── app.py                 # FastAPI app factory with internal queue
│   └── __main__.py            # CLI: python -m annophis-mlhub.worker serve
└── routes/
    ├── annotate.py            # POST /annotate
    └── health.py              # GET /health, GET /annotators
```

## Adding an annotator

### Local annotator

1. Subclass `LocalAnnotator`, implement `annotate_sync()`
2. Constructor must call `super().__init__(name, annotation_type)`
3. Optional `max_concurrency` kwarg controls parallel thread access (default: 1)
4. Add `[[annotator]]` entry in `annophis-mlhub.toml`

### Remote annotator (generic)

No subclassing needed — use `GenericRemoteAnnotator` directly in TOML:

```toml
[[annotator]]
name = "remote-ner"
annotation_type = "ner"
class_path = "annophis-mlhub.annotators.remote.GenericRemoteAnnotator"
base_url = "http://localhost:8001"
```

The remote service must speak the annophis-mlhub protocol: `POST /annotate` accepting `{"text": "..."}` and returning `{"annotator": "...", "annotation_type": "...", "spans": [...]}`.

### Worker harness (model service)

Wrap any ML model into a annophis-mlhub-compatible service:

```python
from annophis-mlhub.worker import ModelWorker, Span

class MyModel(ModelWorker):
    def load(self):
        self.model = ...  # load weights

    def predict(self, text: str) -> list[Span]:
        ...  # return list of Span
```

Run with:

```
python -m annophis-mlhub.worker serve my_module:MyModel \
    --name my-ner --annotation-type ner --port 8001
```

The harness provides `/annotate`, `/health`, `/info` endpoints and an internal queue that serializes inference through a single worker thread (safe for GPU models).

## Running

```
python main.py
```

Server settings via env vars: `ANNOHUB_HOST`, `ANNOHUB_PORT`, `ANNOHUB_DEBUG`.

## Testing

```
uv run pytest tests/ -v
```

Tests use a nonexistent config path by default so no annotators load. Use the `_use_real_config` fixture to test with `annophis-mlhub.toml`.

## Key design decisions

- All annotators are async at the interface. Local ones wrap blocking work with `asyncio.to_thread`.
- `LocalAnnotator` uses an `asyncio.Semaphore` to bound concurrent inference (default: 1, serialized).
- Multiple annotators run concurrently via `asyncio.gather`.
- Annotator registry is a plain dict, populated at startup from `annophis-mlhub.toml`.
- `class_path` in config enables dynamic import — new annotators need no core code changes.
- `GenericRemoteAnnotator` eliminates subclassing for standard-protocol remote services.
- Worker harness uses `asyncio.Queue` → single worker thread pattern for safe GPU inference.
