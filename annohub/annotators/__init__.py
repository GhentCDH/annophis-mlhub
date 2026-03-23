from annohub.annotators.base import Annotator

_registry: dict[str, Annotator] = {}


def register(annotator: Annotator) -> None:
    _registry[annotator.name] = annotator


def get(name: str) -> Annotator | None:
    return _registry.get(name)


def all() -> dict[str, Annotator]:
    return dict(_registry)


def clear() -> None:
    _registry.clear()
