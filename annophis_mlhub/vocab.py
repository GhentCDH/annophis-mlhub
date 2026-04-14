"""Generates the annophis_mlhub JSON-LD vocabulary document dynamically.

The base URL is configured via ANNOHUB_VOCAB_BASE_URL (default:
http://vocab.annophis_mlhub.org/).  This allows self-hosted deployments to
mint their own vocabulary URIs.
"""

from annophis_mlhub.config import settings


def _build() -> dict:
    base = settings.vocab_base_url.rstrip("/") + "/"

    return {
        "@context": {
            "annophis_mlhub": base,
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "dcterms": "http://purl.org/dc/terms/",
        },
        "@graph": [
            {
                "@id": base,
                "@type": "owl:Ontology",
                "rdfs:label": "Annohub Vocabulary",
                "rdfs:comment": (
                    "Vocabulary for describing NLP annotation services, "
                    "their capabilities, and their input/output contracts."
                ),
            },
            # ── Classes ──────────────────────────────────────────────────────
            {
                "@id": "annophis_mlhub:Annotator",
                "@type": "owl:Class",
                "rdfs:label": "Annotator",
                "rdfs:comment": (
                    "An NLP annotation service that accepts a text document "
                    "and produces annotations."
                ),
            },
            # ── Object properties ─────────────────────────────────────────────
            {
                "@id": "annophis_mlhub:requiresAnnotation",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires annotation",
                "rdfs:comment": (
                    "An annotation type (identified by URI) that must already "
                    "be present in the input document. For example, a POS tagger "
                    "may require tokenisation."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "rdfs:Class"},
            },
            {
                "@id": "annophis_mlhub:requiresLanguage",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires language",
                "rdfs:comment": (
                    "A language (identified by URI, e.g. from lexvo.org) that "
                    "the input text must be written in."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "dcterms:LinguisticSystem"},
            },
            {
                "@id": "annophis_mlhub:requiresFeature",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires feature",
                "rdfs:comment": (
                    "A specific property or feature (identified by URI) that "
                    "must already be populated on input annotations. For example, "
                    "a lemmatiser may require the pos feature on Token annotations."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "rdf:Property"},
            },
            {
                "@id": "annophis_mlhub:producesAnnotation",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "produces annotation",
                "rdfs:comment": (
                    "An annotation type (identified by URI) that this annotator "
                    "adds to the output document."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "rdfs:Class"},
            },
            {
                "@id": "annophis_mlhub:inputGranularity",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "input granularity",
                "rdfs:comment": (
                    "The annotation type (identified by URI) that defines the "
                    "work-unit granularity for content-addressed caching. For "
                    "example, a tokenizer with inputGranularity lapps:Sentence "
                    "processes each sentence independently, enabling per-sentence "
                    "cache reuse when upstream annotations change."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "rdfs:Class"},
            },
            {
                "@id": "annophis_mlhub:producesFeature",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "produces feature",
                "rdfs:comment": (
                    "A specific property or feature (identified by URI) that "
                    "this annotator populates on its output annotations. For "
                    "example, a POS tagger produces the pos feature on Token "
                    "annotations."
                ),
                "rdfs:domain": {"@id": "annophis_mlhub:Annotator"},
                "rdfs:range": {"@id": "rdf:Property"},
            },
        ],
    }


VOCABULARY = _build()
