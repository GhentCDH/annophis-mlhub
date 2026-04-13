from __future__ import annotations

from abc import ABC, abstractmethod

from annophis_mlhub.annotators.mixin import AnnotatorMixin
from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument


class ModelWorker(AnnotatorMixin, ABC):
    """Base class for wrapping an ML model into an annophis_mlhub service.

    Subclasses implement two methods:

    - ``load()``  — called once at startup to load model weights, etc.
    - ``predict(doc)`` — synchronous inference, returns a list of LIFAnnotations.

    The worker harness handles the rest: FastAPI app, health endpoint,
    internal queue, and background worker thread.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Backward compat: if no produces_annotation was given, infer from annotation_type
        if (
            not self.lif_contract.produces_annotation
            and self.annotation_type != "unknown"
        ):
            self.lif_contract = LIFContract(
                requires_language=self.lif_contract.requires_language,
                requires_annotation=self.lif_contract.requires_annotation,
                requires_feature=self.lif_contract.requires_feature,
                produces_annotation=[self.annotation_type],
                produces_feature=self.lif_contract.produces_feature,
                input_granularity=self.lif_contract.input_granularity,
            )

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
