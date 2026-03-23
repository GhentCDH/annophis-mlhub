import pytest
from fastapi.testclient import TestClient

from annohub import annotators
from annohub.app import create_app
from annohub.config import settings


@pytest.fixture(autouse=True)
def _empty_registry():
    """Clear registry before each test so config-loaded annotators don't leak."""
    annotators.clear()
    # point config at a nonexistent file so load_annotators() is a no-op
    original = settings.config_path
    settings.config_path = original.parent / "nonexistent.toml"
    yield
    annotators.clear()
    settings.config_path = original


@pytest.fixture
def _use_real_config():
    """Restore real config path so annotators load from annohub.toml."""
    from annohub.config import CONFIG_PATH

    settings.config_path = CONFIG_PATH


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
