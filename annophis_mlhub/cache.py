"""Content-addressed annotation caching.

Computes input hashes per work unit so the pipeline can skip recomputation
when upstream annotations haven't changed.
The LIFDocument itself acts as the cache: no external storage needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from annophis_mlhub.annotators.base import Annotator
from annophis_mlhub.lif import (
    ContainsEntry,
    LIFAnnotation,
    LIFContract,
    LIFDocument,
    ViewMetadata,
    _type_match_set,
)

logger = logging.getLogger(__name__)

_CACHE_META_KEYS = {"input_hash", "granularity_span", "producer"}


def _annotation_hash_data(ann: LIFAnnotation, strip_offsets: bool = False) -> str:
    updates: dict = {
        "metadata": {k: v for k, v in ann.metadata.items() if k not in _CACHE_META_KEYS}
    }
    if strip_offsets:
        updates["start"] = None
        updates["end"] = None
    clean = ann.model_copy(update=updates)
    return json.dumps(
        clean.model_dump(by_alias=True, exclude_none=True), sort_keys=True
    )


def _compute_input_hash(
    text: str,
    upstream_annotations: list[LIFAnnotation] | None = None,
    *,
    strip_offsets: bool = False,
) -> str:
    h = hashlib.sha256(text.encode())
    if upstream_annotations:
        for ann in sorted(upstream_annotations, key=lambda a: a.id):
            h.update(_annotation_hash_data(ann, strip_offsets=strip_offsets).encode())
    return h.hexdigest()[:16]


def _span_key(start: int, end: int) -> str:
    return f"{start}:{end}"


def _annotations_in_range(
    annotations: list[LIFAnnotation],
    start: int,
    end: int,
) -> list[LIFAnnotation]:
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
    if not doc.views:
        return []
    return [
        a for a in doc.views[0].annotations if a.metadata.get("producer") == producer
    ]


def _upstream_annotations(
    doc: LIFDocument,
    contract: LIFContract,
) -> list[LIFAnnotation]:
    if not doc.views or not contract.requires_annotation:
        return []
    required: set[str] = set()
    for t in contract.requires_annotation:
        required |= _type_match_set(t)
    return [a for a in doc.views[0].annotations if a.type in required]


@dataclass
class CachePlan:
    doc: LIFDocument
    producer: str
    contract: LIFContract
    hits: list[LIFAnnotation] = field(default_factory=list)
    miss_spans: list[tuple[int, int]] = field(default_factory=list)
    skip_entirely: bool = False

    @staticmethod
    def compute(
        doc: LIFDocument,
        producer: str,
        contract: LIFContract,
    ) -> CachePlan:
        """Determine which work units need recomputation."""
        existing = _annotations_by_producer(doc, producer)
        upstream = _upstream_annotations(doc, contract)

        logger.debug(
            "%s: computing cache plan (%d existing annotations)",
            producer,
            len(existing),
        )

        # No input granularity: use full document as single span
        if contract.input_granularity is None:
            doc_hash = _compute_input_hash(doc.text.value, upstream or None)
            if existing and all(
                a.metadata.get("input_hash") == doc_hash for a in existing
            ):
                logger.debug(
                    "%s: full cache hit (doc-level, %d annotations)",
                    producer,
                    len(existing),
                )
                return CachePlan(
                    doc, producer, contract, hits=existing, skip_entirely=True
                )
            logger.debug("%s: cache miss (doc-level)", producer)
            return CachePlan(
                doc, producer, contract, miss_spans=[(0, len(doc.text.value))]
            )

        granularity_spans: list[tuple[int, int]] = list(
            doc.spans(contract.input_granularity)
        )

        # No spans of this type in the document: fall back to document-level
        if not granularity_spans:
            logger.debug(
                "%s: no %s spans found, falling back to doc-level",
                producer,
                contract.input_granularity,
            )
            doc_hash = _compute_input_hash(doc.text.value, upstream or None)
            if existing and all(
                a.metadata.get("input_hash") == doc_hash for a in existing
            ):
                logger.debug("%s: full cache hit (doc-level fallback)", producer)
                return CachePlan(
                    doc, producer, contract, hits=existing, skip_entirely=True
                )
            return CachePlan(
                doc, producer, contract, miss_spans=[(0, len(doc.text.value))]
            )

        # Group existing annotations by span key and by input_hash
        existing_by_span: dict[str, list[LIFAnnotation]] = {}
        existing_by_hash: dict[str, list[tuple[str, LIFAnnotation]]] = {}
        for a in existing:
            key = a.metadata.get("granularity_span")
            ihash = a.metadata.get("input_hash")
            if key is not None:
                existing_by_span.setdefault(key, []).append(a)
            if ihash is not None and key is not None:
                existing_by_hash.setdefault(ihash, []).append((key, a))

        plan = CachePlan(doc, producer, contract)
        for start, end in granularity_spans:
            key = _span_key(start, end)
            text_slice = doc.text.value[start:end]
            upstream_in_range = _annotations_in_range(upstream, start, end)
            span_hash = _compute_input_hash(
                text_slice, upstream_in_range or None, strip_offsets=True
            )

            # Exact span key match
            group = existing_by_span.get(key, [])
            if group and all(a.metadata.get("input_hash") == span_hash for a in group):
                logger.debug(
                    "%s: span %s hit (exact match, %d annotations)",
                    producer,
                    key,
                    len(group),
                )
                plan.hits.extend(group)
                continue

            # Hash match: same content at different offsets
            hash_group = existing_by_hash.get(span_hash)
            if hash_group:
                old_key = hash_group[0][0]
                old_start = int(old_key.split(":")[0])
                offset_delta = start - old_start
                for _, a in hash_group:
                    plan.hits.append(
                        a.model_copy(
                            update={
                                "start": (a.start or 0) + offset_delta,
                                "end": (a.end or 0) + offset_delta,
                                "metadata": {
                                    **a.metadata,
                                    "granularity_span": key,
                                },
                            }
                        )
                    )
                del existing_by_hash[span_hash]
                logger.debug(
                    "%s: span %s hit (relocated from %s)", producer, key, old_key
                )
                continue

            plan.miss_spans.append((start, end))
            logger.debug("%s: span %s miss", producer, key)

        plan.skip_entirely = len(plan.miss_spans) == 0
        logger.debug(
            "%s: plan complete — %d hits, %d misses, skip=%s",
            producer,
            len(plan.hits),
            len(plan.miss_spans),
            plan.skip_entirely,
        )
        return plan

    async def execute(self, annotator: Annotator) -> LIFDocument:
        """Run the full cache-aware annotation cycle and return the updated document."""
        doc = self._remove_stale()
        run_doc = self._build_filtered_document(doc)

        logger.debug(
            "%s: running annotator on %d miss spans",
            self.producer,
            len(self.miss_spans),
        )
        annotations = await annotator.annotate(run_doc)
        logger.debug(
            "%s: annotator produced %d annotations", self.producer, len(annotations)
        )
        annotations = self._stamp(annotations, doc)
        return self._merge(doc, annotations)

    def _remove_stale(self) -> LIFDocument:
        """Remove annotations from this producer that aren't cache hits."""
        if not self.doc.views:
            return self.doc

        hit_ids = {a.id for a in self.hits}
        kept = [
            a
            for a in self.doc.views[0].annotations
            if a.metadata.get("producer") != self.producer or a.id in hit_ids
        ]
        new_view = self.doc.views[0].model_copy(update={"annotations": kept})
        return self.doc.model_copy(update={"views": [new_view]})

    def _build_filtered_document(self, doc: LIFDocument) -> LIFDocument:
        """Build a document containing only upstream annotations within miss spans.

        The original text is preserved so character offsets remain valid.
        Since e.g. Sentence annotations are filtered to only those that span missed
        parts of the document, a Sentence-level annotator won't process the hit sentences.
        """
        if not self.contract.input_granularity or not self.miss_spans or not doc.views:
            return doc

        required_types: set[str] = set()
        for t in self.contract.requires_annotation:
            required_types |= _type_match_set(t)
        filtered: list[LIFAnnotation] = []
        for ann in doc.views[0].annotations:
            if ann.type not in required_types:
                continue
            for start, end in self.miss_spans:
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

    def _stamp(
        self, annotations: list[LIFAnnotation], doc: LIFDocument
    ) -> list[LIFAnnotation]:
        """Stamp input_hash, granularity_span, and producer on returned annotations."""
        upstream = _upstream_annotations(doc, self.contract)

        granularity_spans = (
            list(doc.spans(self.contract.input_granularity))
            if self.contract.input_granularity
            else []
        )

        # Document-level: no granularity, or granularity type not present in doc
        if not granularity_spans:
            doc_hash = _compute_input_hash(doc.text.value, upstream or None)
            return [
                a.model_copy(
                    update={
                        "metadata": {
                            **a.metadata,
                            "input_hash": doc_hash,
                            "producer": self.producer,
                        }
                    }
                )
                for a in annotations
            ]

        stamped = []
        for ann in annotations:
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
                    span_hash = _compute_input_hash(
                        text_slice, upstream_in_range or None, strip_offsets=True
                    )
                    span_key = key
                    break

            stamped.append(
                ann.model_copy(
                    update={
                        "metadata": {
                            **ann.metadata,
                            "input_hash": span_hash,
                            "granularity_span": span_key,
                            "producer": self.producer,
                        }
                    }
                )
            )
        return stamped

    def _merge(self, doc: LIFDocument, annotations: list[LIFAnnotation]) -> LIFDocument:
        """Merge annotations into the document's single view.

        If an incoming annotation has the same ``id`` as an existing one,
        its features are merged into the existing annotation. Annotations
        with new ids are appended.

        ``produces_feature`` entries are recorded in ``metadata.contains``
        so downstream annotators can check for them via contract validation.
        """
        view = doc.views[0]

        existing_by_id: dict[str, int] = {
            a.id: idx for idx, a in enumerate(view.annotations)
        }
        merged = list(view.annotations)

        for ann in annotations:
            if ann.id in existing_by_id:
                idx = existing_by_id[ann.id]
                old = merged[idx]
                merged_features = {**old.features, **ann.features}
                merged[idx] = old.model_copy(update={"features": merged_features})
            else:
                existing_by_id[ann.id] = len(merged)
                merged.append(ann)

        new_contains = dict(view.metadata.contains)
        for ann in annotations:
            if ann.type not in new_contains:
                new_contains[ann.type] = ContainsEntry(
                    producer=self.producer, type=ann.type
                )
        for feat in self.contract.produces_feature or []:
            if feat not in new_contains:
                new_contains[feat] = ContainsEntry(producer=self.producer, type=feat)
        new_metadata = ViewMetadata(contains=new_contains)
        new_view = view.model_copy(
            update={"annotations": merged, "metadata": new_metadata}
        )
        return doc.model_copy(update={"views": [new_view]})
