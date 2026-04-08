import asyncio
from abc import ABC, abstractmethod
from typing import Any

from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument

_DEFAULT_MAX_CONCURRENCY = 1


class LocalAnnotator(ABC):
    """Base for annotators that run blocking local models.

    Uses a semaphore to bound concurrent inference threads.  This prevents
    multiple requests from hammering a GPU model in parallel.  The default
    concurrency is 1 (fully serialized); override via constructor kwarg.
    """

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    lif_contract: LIFContract

    def __init__(
        self,
        name: str | None = None,
        annotation_type: str | None = None,
        description: str | None = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        requires_language: list[str] | None = None,
        requires_annotation: list[str] | None = None,
        requires_feature: list[str] | None = None,
        produces_annotation: list[str] | None = None,
        produces_feature: list[str] | None = None,
    ):
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if description is not None:
            self.description = description
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.lif_contract = LIFContract(
            requires_language=requires_language or [],
            requires_annotation=requires_annotation or [],
            requires_feature=requires_feature or [],
            produces_annotation=produces_annotation or [],
            produces_feature=produces_feature or [],
        )

    @abstractmethod
    def annotate_sync(self, doc: LIFDocument) -> list[LIFAnnotation]:
        """Synchronous, blocking annotation. Runs in a thread."""
        ...

    def info_sync(self) -> dict[str, Any]:
        """Return JSON-LD descriptor for this annotator. Override to customise."""
        return build_descriptor_node(self)

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]:
        async with self._semaphore:
            return await asyncio.to_thread(self.annotate_sync, doc)

    async def info(self) -> dict[str, Any]:
        return self.info_sync()


def build_descriptor_context() -> list:
    """Return the shared ``@context`` for annotator descriptors."""
    from annophis_mlhub.config import settings

    vocab_ns = settings.vocab_base_url.rstrip("/") + "/"
    return [
        {
            "annophis_mlhub": vocab_ns,
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "lapps": "http://vocab.lappsgrid.org/",
            "dcterms": "http://purl.org/dc/terms/",
            "lexvo": "http://lexvo.org/id/iso639-3/",
        },
    ]


def build_descriptor_node(annotator: Any) -> dict[str, Any]:
    """Build a JSON-LD graph node for an annotator (no ``@context``)."""
    node: dict[str, Any] = {
        "@type": "annophis_mlhub:Annotator",
        "rdfs:label": annotator.name,
        "dcterms:description": annotator.description,
    }

    contract: LIFContract = annotator.lif_contract
    if contract.requires_language:
        node["annophis_mlhub:requiresLanguage"] = [
            {"@id": lang} for lang in contract.requires_language
        ]
    if contract.requires_annotation:
        node["annophis_mlhub:requiresAnnotation"] = [
            {"@id": t} for t in contract.requires_annotation
        ]
    if contract.requires_feature:
        node["annophis_mlhub:requiresFeature"] = [
            {"@id": f} for f in contract.requires_feature
        ]
    if contract.produces_annotation:
        node["annophis_mlhub:producesAnnotation"] = [
            {"@id": t} for t in contract.produces_annotation
        ]
    if contract.produces_feature:
        node["annophis_mlhub:producesFeature"] = [
            {"@id": f} for f in contract.produces_feature
        ]

    return node
