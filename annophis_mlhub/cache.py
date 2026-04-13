"""Content-addressed annotation caching.

Computes input hashes per work unit so the pipeline can skip recomputation
when upstream annotations haven't changed.  The LIFDocument itself acts as
the cache — no external storage needed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument

# ── Hashing ─────────────────────────────────────────────────────────────────


def compute_input_hash(
    text: str,
    upstream_annotations: list[LIFAnnotation] | None = None,
) -> str:
    """Deterministic hash of a work unit's effective input."""
    h = hashlib.sha256(text.encode())
    if upstream_annotations:
        for ann in sorted(upstream_annotations, key=lambda a: a.id):
            h.update(ann.model_dump_json(by_alias=True).encode())
    return h.hexdigest()[:16]


# ── Cache plan ──────────────────────────────────────────────────────────────


@dataclass
class CachePlan:
    hits: list[LIFAnnotation] = field(default_factory=list)
    miss_spans: list[tuple[int, int]] = field(default_factory=list)
    miss_hashes: dict[str, str] = field(default_factory=dict)  # "start:end" -> hash
    skip_entirely: bool = False

    def __repr__(self) -> str:
        result = ""
        result += "hits: \n"
        for hit in self.hits:
            result += f"\t-({hit.start}, {hit.end})\n"

        result += "misses: \n"
        for miss in self.miss_spans:
            result += f"\t-({miss[0]}, {miss[1]})\n"

        return result


def _span_key(start: int, end: int) -> str:
    return f"{start}:{end}"


def _annotations_in_range(
    annotations: list[LIFAnnotation],
    start: int,
    end: int,
) -> list[LIFAnnotation]:
    """Return annotations whose span falls within [start, end)."""
    return [
        a
        for a in annotations
        if a.start is not None
        and a.end is not None
        and a.start >= start
        and a.end <= end
    ]


def _annotations_by_producer(
    doc: LIFDocument,
    producer: str,
) -> list[LIFAnnotation]:
    """Return all annotations in the document produced by a given annotator."""
    if not doc.views:
        return []
    return [
        a for a in doc.views[0].annotations if a.metadata.get("producer") == producer
    ]


def _upstream_annotations(
    doc: LIFDocument,
    contract: LIFContract,
) -> list[LIFAnnotation]:
    """Return annotations matching the contract's requires_annotation types."""
    if not doc.views or not contract.requires_annotation:
        return []
    required = set(contract.requires_annotation)
    return [a for a in doc.views[0].annotations if a.type in required]


def compute_cache_plan(
    doc: LIFDocument,
    producer: str,
    contract: LIFContract,
) -> CachePlan:
    """Determine which work units need recomputation."""
    existing = _annotations_by_producer(doc, producer)
    upstream = _upstream_annotations(doc, contract)

    # ── Document-level (no granularity) ─────────────────────────────────
    if contract.input_granularity is None:
        doc_hash = compute_input_hash(doc.text.value, upstream or None)
        if existing and all(a.metadata.get("input_hash") == doc_hash for a in existing):
            return CachePlan(hits=existing, skip_entirely=True)
        return CachePlan(
            miss_spans=[(0, len(doc.text.value))],
            miss_hashes={_span_key(0, len(doc.text.value)): doc_hash},
        )

    # ── Per-span granularity ────────────────────────────────────────────
    granularity_spans: list[tuple[int, int]] = list(
        doc.spans(contract.input_granularity)
    )

    # Group existing annotations by their granularity_span metadata
    existing_by_span: dict[str, list[LIFAnnotation]] = {}
    for a in existing:
        key = a.metadata.get("granularity_span")
        if key is not None:
            existing_by_span.setdefault(key, []).append(a)

    plan = CachePlan()
    for start, end in granularity_spans:
        key = _span_key(start, end)
        text_slice = doc.text.value[start:end]
        upstream_in_range = _annotations_in_range(upstream, start, end)
        span_hash = compute_input_hash(text_slice, upstream_in_range or None)

        group = existing_by_span.get(key, [])
        if group and all(a.metadata.get("input_hash") == span_hash for a in group):
            plan.hits.extend(group)
        else:
            plan.miss_spans.append((start, end))
            plan.miss_hashes[key] = span_hash

    plan.skip_entirely = len(plan.miss_spans) == 0
    return plan


# ── Filtered document ───────────────────────────────────────────────────────


def build_filtered_document(
    doc: LIFDocument,
    miss_spans: list[tuple[int, int]],
    contract: LIFContract,
) -> LIFDocument:
    """Build a document containing only upstream annotations within miss spans.

    The original text is preserved so character offsets remain valid.
    """
    if not doc.views:
        return doc

    required_types = set(contract.requires_annotation)
    filtered: list[LIFAnnotation] = []
    for ann in doc.views[0].annotations:
        if ann.type not in required_types:
            continue
        for start, end in miss_spans:
            if (
                ann.start is not None
                and ann.end is not None
                and ann.start >= start
                and ann.end <= end
            ):
                filtered.append(ann)
                break

    new_view = doc.views[0].model_copy(update={"annotations": filtered})
    return doc.model_copy(update={"views": [new_view]})


# ── Stamp annotations ──────────────────────────────────────────────────────


def stamp_annotations(
    annotations: list[LIFAnnotation],
    producer: str,
    contract: LIFContract,
    doc: LIFDocument,
) -> list[LIFAnnotation]:
    """Stamp input_hash, granularity_span, and producer on returned annotations."""
    upstream = _upstream_annotations(doc, contract)

    if contract.input_granularity is None:
        doc_hash = compute_input_hash(doc.text.value, upstream or None)
        return [
            a.model_copy(
                update={
                    "metadata": {
                        **a.metadata,
                        "input_hash": doc_hash,
                        "producer": producer,
                    }
                }
            )
            for a in annotations
        ]

    # Build span lookup for granularity
    granularity_spans = list(doc.spans(contract.input_granularity))

    stamped = []
    for ann in annotations:
        # Find which granularity span contains this annotation
        span_key = None
        span_hash = None
        for start, end in granularity_spans:
            if (
                ann.start is not None
                and ann.start >= start
                and ann.end is not None
                and ann.end <= end
            ):
                key = _span_key(start, end)
                text_slice = doc.text.value[start:end]
                upstream_in_range = _annotations_in_range(upstream, start, end)
                span_hash = compute_input_hash(text_slice, upstream_in_range or None)
                span_key = key
                break

        stamped.append(
            ann.model_copy(
                update={
                    "metadata": {
                        **ann.metadata,
                        "input_hash": span_hash,
                        "granularity_span": span_key,
                        "producer": producer,
                    }
                }
            )
        )
    return stamped


# ── Remove stale annotations ───────────────────────────────────────────────


def remove_stale_annotations(
    doc: LIFDocument,
    producer: str,
    plan: CachePlan,
) -> LIFDocument:
    """Remove annotations from this producer that aren't cache hits."""
    if not doc.views:
        return doc

    hit_ids = {a.id for a in plan.hits}
    kept = [
        a
        for a in doc.views[0].annotations
        if a.metadata.get("producer") != producer or a.id in hit_ids
    ]
    new_view = doc.views[0].model_copy(update={"annotations": kept})
    return doc.model_copy(update={"views": [new_view]})
