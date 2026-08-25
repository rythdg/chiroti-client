import pytest

from chiroti import config


@pytest.fixture(autouse=True)
def reset_config(monkeypatch):
    config._overrides.clear()
    monkeypatch.delenv("CHIROTI_SERVER", raising=False)
    monkeypatch.delenv("CHIROTI_TOKEN", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", config.Path("/nonexistent/chiroti-config.json"))
    yield
    config._overrides.clear()
