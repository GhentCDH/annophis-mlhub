from pydantic import BaseModel


class Health(BaseModel):
    status: str = "ok"


class Span(BaseModel):
    start: int
    end: int
    label: str
    text: str


class AnnotationRequest(BaseModel):
    text: str
    annotators: list[str] | None = None


class AnnotationResult(BaseModel):
    """Result from a single annotator."""

    annotator: str
    annotation_type: str
    spans: list[Span]


class AnnotationLayer(BaseModel):
    annotator: str
    annotation_type: str
    spans: list[Span]


class AnnotationResponse(BaseModel):
    """Full response, potentially from multiple annotators."""

    text: str
    annotations: list[AnnotationLayer]


class AnnotatorInfo(BaseModel):
    name: str
    annotation_type: str
    kind: str  # "local" or "remote"
    description: str = ""
    labels: list[str] = []
    available: bool = True
