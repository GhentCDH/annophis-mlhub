# Content-Addressed Annotation Caching

## The problem

Consider a pipeline: `text -> sentence_split -> tokenizer`

When the sentence splitter's output changes (e.g. a researcher corrects one boundary out of thousands), the tokenizer must re-run. But most sentences haven't changed — recomputing all of them is wasteful.

## The solution

Each annotation carries a hash of the input that produced it. The LIF document itself acts as the cache. When a pipeline re-runs, the framework compares hashes to determine which work units actually need recomputation.

No external cache layer. No extra storage. The data is already in the document.

## How it works

### Input hashing

Every annotation produced through the pipeline gets two metadata fields stamped by the framework:

- **`input_hash`** — SHA-256 (truncated to 16 hex chars) of the effective input that produced this annotation
- **`producer`** — URI of the annotator that created it

For annotators that operate per-span (e.g. per-sentence), a third field is added:

- **`granularity_span`** — `"start:end"` identifying which parent span this annotation belongs to

### What gets hashed

The hash captures everything the annotator saw to produce its output:

**Document-level annotators** (no `input_granularity`):
```
hash(full document text + all upstream annotations matching requires_annotation)
```

**Per-span annotators** (e.g. `input_granularity = "Sentence"`):
```
hash(text[start:end] of the sentence + upstream annotations within that span range)
```

The text is always scoped to the actual slice the annotator operates on — not the full document text.

When upstream annotations are included in the hash, cache-internal metadata (`input_hash`, `producer`, `granularity_span`) is stripped before serialization. This ensures the hash reflects only the semantic content of the annotation — its type, span, and features — not bookkeeping from previous pipeline runs. Without this, re-running an upstream annotator that produces identical output would still invalidate downstream caches because the metadata stamps differ between runs.

### The cache plan

Before calling an annotator, the framework computes a **cache plan** by inspecting existing annotations in the document:

1. Find all current granularity spans (e.g. all Sentence boundaries)
2. For each span, compute the input hash from the current state
3. Look up existing annotations from this producer, grouped by `granularity_span`
4. Compare hashes:
   - **Hit**: span exists, hash matches -> keep these annotations as-is
   - **Miss**: span is new, or hash changed -> needs recomputation
   - **Stale**: existing annotations whose `granularity_span` no longer matches any current span -> remove

### Pipeline execution with caching

```
for each annotator in pipeline:
    1. compute_cache_plan(document, annotator)
       -> hits, misses, stale

    2. if all hits: skip this annotator entirely

    3. remove stale annotations from document

    4. build a filtered document containing only
       upstream annotations within miss spans

    5. call annotator with filtered document

    6. stamp input_hash + granularity_span + producer
       on returned annotations

    7. merge new annotations into document
       (hits are already there)
```

### Example walkthrough

**Run 1** — fresh document, no cached annotations:

```
Text: "a b c. d e f g h. i j k."

sentence_split produces:
  s0: Sentence(0,6)   "a b c."
  s1: Sentence(7,17)  "d e f g h."
  s2: Sentence(18,24) "i j k."

tokenizer (input_granularity=Sentence):
  Cache plan: 3 misses (no existing annotations)
  Runs on all 3 sentences
  Stamps each token with:
    - input_hash = hash("a b c." + s0)  (for tokens in first sentence)
    - granularity_span = "0:6"
    - producer = ".../annotators/tokenizer"
```

**Run 2** — same document returned as input (e.g. client re-submits):

```
sentence_split:
  Cache plan: all annotations have matching input_hash -> skip entirely

tokenizer:
  Cache plan: all 3 spans exist with matching hashes -> skip entirely
```

**Run 3** — researcher edited sentence boundaries (split first sentence differently):

```
Text: "a b c. d e f g h. i j k."  (unchanged)

sentence_split produces different output:
  s0: Sentence(0,10)  "a b c. d e"    <- changed
  s1: Sentence(11,17) "f g h."        <- changed
  s2: Sentence(18,24) "i j k."        <- same

tokenizer:
  Cache plan:
    - "0:6" no longer exists   -> stale, remove old tokens
    - "7:17" no longer exists  -> stale, remove old tokens
    - "0:10" is new            -> miss, recompute
    - "11:17" is new           -> miss, recompute
    - "18:24" exists, hash matches -> HIT, keep tokens

  Only runs on sentences "a b c. d e" and "f g h."
  Tokens for "i j k." are preserved untouched.
```

## Configuring an annotator for caching

Add `input_granularity` to the contract — in TOML:

```toml
[[annotator]]
name = "tokenizer"
annotation_type = "token"
class_path = "my_module.Tokenizer"
requires_annotation = ["Sentence"]
produces_annotation = ["Token"]
input_granularity = "Sentence"
```

Or in Python:

```python
class MyTokenizer(LocalAnnotator):
    def __init__(self):
        super().__init__(
            name="tokenizer",
            annotation_type="token",
            requires_annotation=["Sentence"],
            produces_annotation=["Token"],
            input_granularity="Sentence",
        )
```

Annotators without `input_granularity` use document-level caching (all-or-nothing).

For remote annotators (`GenericRemoteAnnotator`), the contract is automatically synced from the worker's `/info` endpoint on first contact. If a `ModelWorker` declares `input_granularity` in its constructor, the hub picks it up without needing to duplicate it in `mlhub.toml`. Fields explicitly set in TOML take precedence over the remote contract.

## Design properties

- **Annotators are unaware of caching.** The framework handles hashing, plan computation, filtering, and stamping. Annotators see a normal `LIFDocument` and return annotations as usual.
- **The document is the cache.** No external state. Clients can persist documents and get caching for free on re-submission.
- **Offsets are preserved.** The filtered document keeps the original text — only view annotations are filtered. Character offsets remain valid.
- **Deterministic.** Same input always produces the same hash. Hashing is based on text content and upstream annotation data, not on timestamps or ordering artifacts.
- **Graceful on first run.** When no cached annotations exist, everything is a miss and the pipeline runs normally. The only overhead is hash computation.
