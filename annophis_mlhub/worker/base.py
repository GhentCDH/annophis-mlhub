from __future__ import annotations

from abc import ABC, abstractmethod

from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument


class ModelWorker(ABC):
    """Base class for wrapping an ML model into an annophis_mlhub service.

    Subclasses implement two methods:

    - ``load()``  — called once at startup to load model weights, etc.
    - ``predict(doc)`` — synchronous inference, returns a list of LIFAnnotations.

    The worker harness handles the rest: FastAPI app, health endpoint,
    internal queue, and background worker thread.
    """

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""

    def __init__(self, name: str | None = None, annotation_type: str | None = None):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type

    @property
    def lif_contract(self) -> LIFContract:
        return LIFContract(produces_annotation=[self.annotation_type])

    @abstractmethod
    def load(self) -> None:
        """Load model weights, resources, etc.  Called once at startup."""
        ...

    @abstractmethod
    def predict(self, doc: LIFDocument) -> list[LIFAnnotation]:
        """Run inference on a LIF document.  Called from a worker thread.

        The full document is provided so that implementations can access
        prior annotations in ``doc.views`` (e.g. sentence boundaries) as
        well as the raw text via ``doc.text.value``.
        """
        ...
