from pydantic import BaseModel

from annophis_mlhub.lif import LIFAnnotation, LIFDocument


class Health(BaseModel):
    status: str = "ok"


class WsInputUnit(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    document: LIFDocument


class WsOutputUnit(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    annotator: str
    annotations: list[LIFAnnotation]
