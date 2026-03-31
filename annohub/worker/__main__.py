"""CLI entry point for the annohub worker harness.

Usage::

    python -m annohub.worker serve my_module:MyModel \\
        --name my-ner --annotation-type ner --port 8001
"""

import argparse
import importlib
import sys

import uvicorn

from annohub.worker.app import create_worker_app


def _import_worker(path: str):
    """Import 'module.path:ClassName' and return an instance."""
    module_path, class_name = path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


def main():
    parser = argparse.ArgumentParser(
        prog="annohub-worker",
        description="Run a annohub-compatible model service.",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start a worker service")
    serve.add_argument(
        "worker_class",
        help="Import path to ModelWorker subclass, e.g. 'my_module:MyModel'",
    )
    serve.add_argument("--name", default=None, help="Annotator name")
    serve.add_argument("--annotation-type", default=None, help="Annotation type")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve.add_argument("--port", type=int, default=8001, help="Bind port")
    serve.add_argument(
        "--max-queue-size", type=int, default=64, help="Max pending requests"
    )
    serve.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent inference threads (default: 1, safe for GPU models)",
    )

    args = parser.parse_args()
    if args.command != "serve":
        parser.print_help()
        sys.exit(1)

    cls = _import_worker(args.worker_class)
    worker = cls(name=args.name, annotation_type=args.annotation_type)

    app = create_worker_app(
        worker, max_queue_size=args.max_queue_size, max_workers=args.workers
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
