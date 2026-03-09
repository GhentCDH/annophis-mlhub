# Konekaare

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
konekaare/
├── app.py                     # app factory + lifespan
├── config.py                  # pydantic-settings + annotator loading from TOML
├── docs.py                    # Scalar docs route + custom CSS
├── models.py                  # Pydantic request/response schemas
├── annotators/
│   ├── __init__.py            # registry: register(), get(), all(), clear()
│   ├── base.py                # Annotator Protocol
│   ├── local.py               # LocalAnnotator ABC (asyncio.to_thread)
│   ├── remote.py              # RemoteAnnotator ABC (httpx)
│   └── dummy.py               # example: regex-based NER annotator
└── routes/
    ├── annotate.py            # POST /annotate
    └── health.py              # GET /health, GET /annotators
```

## Adding an annotator

1. Write a class extending `LocalAnnotator` (blocking) or `RemoteAnnotator` (API-backed)
2. Add an `[[annotator]]` entry in `konekaare.toml` with `name`, `annotation_type`, `class_path`
3. Any extra TOML fields are passed as kwargs to the constructor

## Running

```
python main.py
```

Server settings via env vars: `KONEKAARE_HOST`, `KONEKAARE_PORT`, `KONEKAARE_DEBUG`.

## Testing

```
uv run pytest tests/ -v
```

Tests use a nonexistent config path by default so no annotators load. Use the `_use_real_config` fixture to test with `konekaare.toml`.

## Key design decisions

- All annotators are async at the interface. Local ones wrap blocking work with `asyncio.to_thread`.
- Multiple annotators run concurrently via `asyncio.gather`.
- Annotator registry is a plain dict, populated at startup from `konekaare.toml`.
- `class_path` in config enables dynamic import — new annotators need no core code changes.
