from typing import Any, Protocol, runtime_checkable

from annophis_mlhub.lif import LIFAnnotation, LIFContract, LIFDocument


@runtime_checkable
class Annotator(Protocol):
    """Protocol that all annotators must satisfy."""

    name: str
    annotation_type: str
    description: str
    lif_contract: LIFContract

    async def annotate(self, doc: LIFDocument) -> list[LIFAnnotation]: ...

    async def info(self) -> dict[str, Any]: ...
