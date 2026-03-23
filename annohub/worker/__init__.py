"""Annohub worker harness — wrap any model into a annohub-compatible service.

Usage::

    from annohub.worker import ModelWorker, Span

    class MyModel(ModelWorker):
        def load(self):
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

        def predict(self, text: str) -> list[Span]:
            doc = self.nlp(text)
            return [
                Span(start=ent.start_char, end=ent.end_char,
                     label=ent.label_, text=ent.text)
                for ent in doc.ents
            ]

Then run::

    python -m annohub.worker serve my_module:MyModel --port 8001

Or run the FastAPI server in a script::
    from annohub.worker import create_worker_app
    from my_module import MyModel
    import uvicorn

    worker = MyModel()
    app = create_worker_app(worker)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)
"""

from annohub.models import Span
from annohub.worker.app import create_worker_app
from annohub.worker.base import ModelWorker

__all__ = ["ModelWorker", "Span", "create_worker_app"]
