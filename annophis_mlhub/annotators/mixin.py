from annophis_mlhub.lif import LIFContract


class AnnotatorMixin:
    """Shared attribute initialisation for all annotator-like classes.

    Sets name, annotation_type, description, and builds a LIFContract
    from requires_*/produces_* kwargs.
    """

    name: str = "unnamed"
    annotation_type: str = "unknown"
    description: str = ""
    lif_contract: LIFContract
    supports_streaming: bool = False

    def __init__(
        self,
        *,
        name: str | None = None,
        annotation_type: str | None = None,
        description: str | None = None,
        requires_language: list[str] | None = None,
        requires_annotation: list[str] | None = None,
        requires_feature: list[str] | None = None,
        produces_annotation: list[str] | None = None,
        produces_feature: list[str] | None = None,
        input_granularity: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if name is not None:
            self.name = name
        if annotation_type is not None:
            self.annotation_type = annotation_type
        if description is not None:
            self.description = description
        self.lif_contract = LIFContract(
            requires_language=requires_language or [],
            requires_annotation=requires_annotation or [],
            requires_feature=requires_feature or [],
            produces_annotation=produces_annotation or [],
            produces_feature=produces_feature or [],
            input_granularity=input_granularity,
        )
