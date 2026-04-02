"""Generates the annohub JSON-LD vocabulary document dynamically.

The base URL is configured via ANNOHUB_VOCAB_BASE_URL (default:
http://vocab.annohub.org/).  This allows self-hosted deployments to
mint their own vocabulary URIs.
"""

from annohub.config import settings


def _build() -> dict:
    base = settings.vocab_base_url.rstrip("/") + "/"

    return {
        "@context": {
            "annohub": base,
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
                "@id": "annohub:Annotator",
                "@type": "owl:Class",
                "rdfs:label": "Annotator",
                "rdfs:comment": (
                    "An NLP annotation service that accepts a text document "
                    "and produces annotations."
                ),
            },
            # ── Object properties ─────────────────────────────────────────────
            {
                "@id": "annohub:requiresAnnotation",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires annotation",
                "rdfs:comment": (
                    "An annotation type (identified by URI) that must already "
                    "be present in the input document. For example, a POS tagger "
                    "may require tokenisation."
                ),
                "rdfs:domain": {"@id": "annohub:Annotator"},
                "rdfs:range": {"@id": "rdfs:Class"},
            },
            {
                "@id": "annohub:requiresLanguage",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires language",
                "rdfs:comment": (
                    "A language (identified by URI, e.g. from lexvo.org) that "
                    "the input text must be written in."
                ),
                "rdfs:domain": {"@id": "annohub:Annotator"},
                "rdfs:range": {"@id": "dcterms:LinguisticSystem"},
            },
            {
                "@id": "annohub:requiresFeature",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "requires feature",
                "rdfs:comment": (
                    "A specific property or feature (identified by URI) that "
                    "must already be populated on input annotations. For example, "
                    "a lemmatiser may require the pos feature on Token annotations."
                ),
                "rdfs:domain": {"@id": "annohub:Annotator"},
                "rdfs:range": {"@id": "rdf:Property"},
            },
            {
                "@id": "annohub:producesAnnotation",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "produces annotation",
                "rdfs:comment": (
                    "An annotation type (identified by URI) that this annotator "
                    "adds to the output document."
                ),
                "rdfs:domain": {"@id": "annohub:Annotator"},
                "rdfs:range": {"@id": "rdfs:Class"},
            },
            {
                "@id": "annohub:producesFeature",
                "@type": "owl:ObjectProperty",
                "rdfs:label": "produces feature",
                "rdfs:comment": (
                    "A specific property or feature (identified by URI) that "
                    "this annotator populates on its output annotations. For "
                    "example, a POS tagger produces the pos feature on Token "
                    "annotations."
                ),
                "rdfs:domain": {"@id": "annohub:Annotator"},
                "rdfs:range": {"@id": "rdf:Property"},
            },
        ],
    }


VOCABULARY = _build()
