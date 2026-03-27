from abc import ABC, abstractmethod

from annohub.models import Contract, Span


class ModelWorker(ABC):
    """Base class for wrapping an ML model into a annohub service.

    Subclasses implement two methods:

    - ``load()``  — called once at startup to load model weights, etc.
    - ``predict(text)`` — synchronous inference, returns a list of Spans.

    The worker harness handles the rest: FastAPI app, health endpoint,
    internal queue, and background worker thread.
    """

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    labels: list[str] = []

    def __init__(self, name: str | None = None, annotation_type: str | None = None):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type

    @property
    def contract(self) -> Contract:
        return Contract(
            requires={"text": True},
            produces=[self.annotation_type],
        )

    @abstractmethod
    def load(self) -> None:
        """Load model weights, resources, etc.  Called once at startup."""
        ...

    @abstractmethod
    def predict(self, text: str) -> list[Span]:
        """Run inference on text.  Called from a worker thread."""
        ...
