import importlib
import tomllib
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings

CONFIG_PATH = Path("mlhub.toml")
DEFAULT_BASE_URL = "http://localhost:8000"


class AnnotatorConfig(BaseModel):
    name: str
    annotation_type: str
    class_path: str
    requires_language: str | None = None
    requires_annotation: list[str] = []
    requires_feature: list[str] = []
    produces_annotation: list[str] = []
    produces_feature: list[str] = []
    # arbitrary extra fields passed to the annotator constructor
    model_config = {"extra": "allow"}


class Settings(BaseSettings):
    model_config = {"env_prefix": "ANNOHUB_"}

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    config_path: Path = CONFIG_PATH
    base_url: str = DEFAULT_BASE_URL
    vocab_base_url: str = f"{DEFAULT_BASE_URL}/vocab"


settings = Settings()


_LIF_CONTRACT_FIELDS = {
    "requires_language",
    "requires_annotation",
    "requires_feature",
    "produces_annotation",
    "produces_feature",
}


def _load_config_file() -> dict:
    path = settings.config_path
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_annotators() -> None:
    from annophis_mlhub import annotators

    config = _load_config_file()

    for entry in config.get("annotator", []):
        cfg = AnnotatorConfig(**entry)
        mod_path, cls_name = cfg.class_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)

        # pass all fields except class_path as kwargs to the constructor;
        # omit empty contract fields so annotator defaults kick in
        exclude = {"class_path"}
        for field in _LIF_CONTRACT_FIELDS:
            value = getattr(cfg, field)
            if value is None or value == [] or value == "":
                exclude.add(field)
        kwargs = cfg.model_dump(exclude=exclude)
        annotators.register(cls(**kwargs))
