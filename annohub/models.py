from typing import Any

from pydantic import BaseModel


class Health(BaseModel):
    status: str = "ok"


class Span(BaseModel):
    start: int
    end: int
    label: str
    text: str


class Contract(BaseModel):
    """Declares an annotator's input requirements and output guarantees.

    ``requires`` maps dot-separated key paths to constraints:
    - ``True``  → key must exist (any value)
    - ``"en"``  → key must exist and equal ``"en"``
    - ``["en", "de"]`` → key must exist and be one of the listed values
    """

    requires: dict[str, bool | str | list[str]] = {}
    produces: list[str] = []


class Document(BaseModel):
    """Unified content container. Annotators extend it by adding keys."""

    model_config = {"extra": "allow"}
    meta: dict[str, Any] = {}
    text: str

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class AnnotationResult(BaseModel):
    """Result from a single annotator."""

    annotator: str
    annotation_type: str
    spans: list[Span]


class WsInputUnit(BaseModel):
    id: str
    document: Document


class WsOutputUnit(BaseModel):
    id: str
    annotator: str
    annotation_type: str
    spans: list[Span]


class AnnotatorInfo(BaseModel):
    name: str
    annotation_type: str
    kind: str  # "local" or "remote"
    description: str = ""
    labels: list[str] = []
    contract: Contract = Contract()
    available: bool = True
