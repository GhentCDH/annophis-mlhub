"""JSON-LD descriptor utilities for annotators."""

from typing import Any

from annophis_mlhub.lif import LIFContract


def annotator_uri(name: str) -> str:
    """Return the canonical ``@id`` URI for an annotator."""
    from annophis_mlhub.config import settings

    return f"{settings.base_url.rstrip('/')}/annotators/{name}"


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
        "@id": annotator_uri(annotator.name),
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
    if contract.input_granularity:
        node["annophis_mlhub:inputGranularity"] = contract.input_granularity

    return node
