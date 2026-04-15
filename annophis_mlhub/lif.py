"""LAPPS Interchange Format (LIF) Pydantic models and contract validation.

Provides the data models for JSON-LD based annotation interchange,
plus contract validation using pyld for CURIE expansion.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pyld import jsonld

from annophis_mlhub.config import settings

LAPPS_CONTEXT: list[str | dict[str, str]] = [
    "http://vocab.lappsgrid.org/context-1.0.0.jsonld",
    # lexvo is not defined in the LAPPS context
    {"lexvo": "http://lexvo.org/id/iso639-3/"},
]


class LIFText(BaseModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    value: str = Field(validation_alias="@value", serialization_alias="@value")
    language: str | None = Field(
        default=None, validation_alias="@language", serialization_alias="@language"
    )


class LIFAnnotation(BaseModel):
    """A single annotation in a LIF view."""

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    id: str
    type: str = Field(validation_alias="@type", serialization_alias="@type")
    label: str | None = None
    start: int | None = None
    end: int | None = None
    features: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class ContainsEntry(BaseModel):
    """Metadata entry describing what a view contains."""

    producer: str = ""
    type: str = ""


class ViewMetadata(BaseModel):
    contains: dict[str, ContainsEntry] = {}


class LIFView(BaseModel):
    id: str
    metadata: ViewMetadata = ViewMetadata()
    annotations: list[LIFAnnotation] = []


class LIFDocument(BaseModel):
    """Top-level LAPPS Interchange Format document."""

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    context: Any = Field(
        default=LAPPS_CONTEXT,
        validation_alias="@context",
        serialization_alias="@context",
    )
    vocab: str | None = Field(
        default=None, validation_alias="@vocab", serialization_alias="@vocab"
    )
    text: LIFText
    metadata: dict[str, Any] = {}
    views: list[LIFView] = []

    # ── Convenience accessors ───────────────────────────────────────────

    def annotations(
        self, annotation_type: str, view: int = 0
    ) -> Iterator[LIFAnnotation]:
        """Yield all annotations of the given type from a view."""
        if view >= len(self.views):
            return
        for ann in self.views[view].annotations:
            if ann.type == annotation_type:
                yield ann

    def spans(self, annotation_type: str, view: int = 0) -> Iterator[tuple[int, int]]:
        """Yield ``(start, end)`` pairs for annotations of the given type."""
        for ann in self.annotations(annotation_type, view):
            if ann.start is not None and ann.end is not None:
                yield ann.start, ann.end

    def span_texts(self, annotation_type: str, view: int = 0) -> Iterator[str]:
        """Yield the text covered by each annotation of the given type."""
        text = self.text.value
        for start, end in self.spans(annotation_type, view):
            yield text[start:end]

    def sentences(self, view: int = 0) -> Iterator[tuple[int, int]]:
        """Yield ``(start, end)`` pairs for Sentence annotations."""
        return self.spans("Sentence", view)

    def tokens(self, view: int = 0) -> Iterator[tuple[int, int]]:
        """Yield ``(start, end)`` pairs for Token annotations."""
        return self.spans("Token", view)


# ── Contract ─────────────────────────────────────────────────────────────────


class LIFContract(BaseModel):
    """Declares an annotator's input requirements and output guarantees
    using LAPPS vocabulary URIs / CURIEs.
    """

    requires_language: list[str] = []  # lexvo URIs, e.g. ["lexvo:grc"]
    requires_annotation: list[str] = []  # e.g. ["lapps:Sentence"]
    requires_feature: list[str] = []  # e.g. ["lapps:Token#pos"]
    produces_annotation: list[str] = []  # e.g. ["lapps:Token"]
    produces_feature: list[str] = []  # e.g. ["lapps:Token#pos"]
    input_granularity: str | None = (
        None  # e.g. "Sentence", "Token"; None=document-level
    )


# ── CURIE expansion helpers ──────────────────────────────────────────────────

# Default context used for expanding CURIEs in contracts and annotations.
DEFAULT_CONTEXT: dict[str, str] = {
    "lapps": "http://vocab.lappsgrid.org/",
    "lexvo": "http://lexvo.org/id/iso639-3/",
    "dcterms": "http://purl.org/dc/terms/",
    "annophis_mlhub": settings.vocab_base_url.rstrip("/") + "/",
}


def _local_name(uri: str) -> str:
    """Extract the local name from a URI: everything after the last ``/`` or ``#``."""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def expand_curie(curie: str, context: dict[str, str] | None = None) -> str:
    """Expand a CURIE like ``lapps:Token`` to a full URI.

    Uses pyld for spec-compliant expansion.  Falls back to the input string
    if expansion produces no result (i.e. it was already a full URI or unknown prefix).
    """
    ctx = context or DEFAULT_CONTEXT
    # pyld expands @type correctly for CURIEs; use that instead of @id
    doc = {"@context": ctx, "@type": curie}
    expanded = jsonld.expand(doc)
    if expanded and "@type" in expanded[0]:
        types = expanded[0]["@type"]
        if types:
            return types[0]
    return curie


def _view_contains_types(
    view: LIFView, context: dict[str, str] | None = None
) -> set[str]:
    """Return annotation type identifiers from a view's metadata.

    Returns both the raw key (e.g. ``"Sentence"``) and the expanded URI
    (e.g. ``"http://vocab.lappsgrid.org/Sentence"``), so matching works
    regardless of whether the contract uses CURIEs or bare names.
    """
    ctx = context or DEFAULT_CONTEXT
    result = set()
    for type_key in view.metadata.contains:
        result.add(type_key)  # bare name
        result.add(expand_curie(type_key, ctx))  # expanded URI
    return result


# ── Contract validation ──────────────────────────────────────────────────────


def validate_lif_contract(
    doc: LIFDocument,
    contract: LIFContract,
) -> list[str]:
    """Check that a LIF document satisfies a contract's requirements.

    Returns a list of human-readable violation descriptions (empty = valid).
    """
    violations: list[str] = []

    # Build a context dict for CURIE expansion from the document's @context.
    ctx = dict(DEFAULT_CONTEXT)
    if isinstance(doc.context, dict):
        ctx.update(doc.context)
    elif isinstance(doc.context, list):
        for entry in doc.context:
            if isinstance(entry, dict):
                ctx.update(entry)

    # ── requires_language ────────────────────────────────────────────────
    if contract.requires_language:
        doc_lang = doc.text.language
        if doc_lang is None:
            violations.append(
                f"requires language {contract.requires_language} but document has no @language"
            )
        else:
            expanded_doc_lang = expand_curie(doc_lang, ctx)
            allowed = {expand_curie(lang, ctx) for lang in contract.requires_language}
            if expanded_doc_lang not in allowed:
                violations.append(
                    f"requires language {contract.requires_language} "
                    f"but document has {doc_lang}"
                )

    # ── requires_annotation ──────────────────────────────────────────────
    # Collect all annotation types declared in views' metadata.contains
    available_types: set[str] = set()
    for view in doc.views:
        available_types |= _view_contains_types(view, ctx)

    for req in contract.requires_annotation:
        expanded_req = expand_curie(req, ctx)
        local_name = _local_name(expanded_req)
        if expanded_req not in available_types and local_name not in available_types:
            violations.append(f"requires annotation type {req}")

    # ── requires_feature ─────────────────────────────────────────────────
    for req_feat in contract.requires_feature:
        # e.g. "lapps:Token#pos" → type URI + feature name
        expanded = expand_curie(req_feat, ctx)
        if "#" in expanded:
            type_uri, _, feat_name = expanded.rpartition("#")
        else:
            # No feature separator — treat as annotation type requirement
            violations.append(
                f"requires feature {req_feat} (malformed, expected Type#feature)"
            )
            continue

        # Check that at least one annotation of that type has the feature.
        # Match both bare name, expanded URI, and local name.
        type_local = _local_name(type_uri)
        found = False
        for view in doc.views:
            for ann in view.annotations:
                ann_type_expanded = expand_curie(ann.type, ctx)
                type_match = (
                    ann.type == type_uri
                    or ann_type_expanded == type_uri
                    or ann.type == type_local
                )
                if type_match and feat_name in ann.features:
                    found = True
                    break
            if found:
                break
        if not found:
            violations.append(f"requires feature {req_feat}")

    return violations


def apply_contract_to_metadata(
    doc: LIFDocument, contract: LIFContract, producer: str
) -> LIFDocument:
    """Project a contract's ``produces_*`` onto the document's metadata.

    Returns a copy of the document whose ``metadata.contains`` has been
    updated as if the annotator had run, without touching annotations.
    This enables a static dry-run: validate each contract in sequence
    against the projected state to catch pipeline errors before running
    any expensive annotators.
    """
    if not doc.views:
        view = LIFView(id="v0", metadata=ViewMetadata(), annotations=[])
        doc = doc.model_copy(update={"views": [view]})

    view = doc.views[0]
    new_contains = dict(view.metadata.contains)

    for ann_type in contract.produces_annotation:
        local = _local_name(expand_curie(ann_type))
        if local not in new_contains:
            new_contains[local] = ContainsEntry(producer=producer, type=local)

    for feat in contract.produces_feature:
        if feat not in new_contains:
            new_contains[feat] = ContainsEntry(producer=producer, type=feat)

    new_metadata = ViewMetadata(contains=new_contains)
    new_view = view.model_copy(update={"metadata": new_metadata})
    return doc.model_copy(update={"views": [new_view]})
