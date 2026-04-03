"""Annohub worker harness — wrap any model into an annophis_mlhub-compatible service.

Usage::

    from annophis_mlhub.worker import ModelWorker, LIFAnnotation

    class MyModel(ModelWorker):
        def load(self):
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

        def predict(self, text: str) -> list[LIFAnnotation]:
            doc = self.nlp(text)
            return [
                LIFAnnotation(
                    id=f"ne{i}",
                    type=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    features={"word": ent.text},
                )
                for i, ent in enumerate(doc.ents)
            ]

Then run::

    python -m annophis_mlhub.worker serve my_module:MyModel --port 8001

Or run the FastAPI server in a script::
    from annophis_mlhub.worker import create_worker_app
    from my_module import MyModel
    import uvicorn

    worker = MyModel()
    app = create_worker_app(worker)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)
"""

from annophis_mlhub.lif import LIFAnnotation
from annophis_mlhub.worker.app import create_worker_app
from annophis_mlhub.worker.base import ModelWorker

__all__ = ["ModelWorker", "LIFAnnotation", "create_worker_app"]
