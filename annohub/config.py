import importlib
import tomllib
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings

CONFIG_PATH = Path("annohub.toml")


class AnnotatorConfig(BaseModel):
    name: str
    annotation_type: str
    class_path: str
    requires: dict[str, bool | str | list[str]] = {}
    produces: list[str] = []
    # arbitrary extra fields passed to the annotator constructor
    model_config = {"extra": "allow"}


class Settings(BaseSettings):
    model_config = {"env_prefix": "ANNOHUB_"}

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    config_path: Path = CONFIG_PATH


settings = Settings()


def _load_config_file() -> dict:
    path = settings.config_path
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_annotators() -> None:
    from annohub import annotators

    config = _load_config_file()

    for entry in config.get("annotator", []):
        cfg = AnnotatorConfig(**entry)
        mod_path, cls_name = cfg.class_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)

        # pass all fields except class_path as kwargs to the constructor;
        # omit empty requires/produces so annotator defaults kick in
        exclude = {"class_path"}
        if not cfg.requires:
            exclude.add("requires")
        if not cfg.produces:
            exclude.add("produces")
        kwargs = cfg.model_dump(exclude=exclude)
        annotators.register(cls(**kwargs))
