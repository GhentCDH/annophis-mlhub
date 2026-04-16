from pydantic import BaseModel, ConfigDict

from annophis_mlhub.lif import LIFAnnotation, LIFDocument


class Health(BaseModel):
    status: str = "ok"


class WsInputUnit(BaseModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    id: str
    document: LIFDocument


class WsOutputUnit(BaseModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    id: str
    annotator: str
    annotations: list[LIFAnnotation]
