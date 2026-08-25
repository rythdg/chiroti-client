import httpx
import pytest

from chiroti import client, config
from chiroti.exceptions import AuthenticationError, ChirotiConnectionError


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._body = body
        self.content = b"x"
        self.text = str(body)

    def json(self):
        return self._body


@pytest.fixture
def configured():
    config.configure(server="http://chiroti:8100", token="test-token")


def test_client_ask_success(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "hello", "model": "qwen-text"})

    monkeypatch.setattr(httpx, "request", fake_request)

    result = client.ask("hi")

    assert result == "hello"
    assert captured["url"] == "http://chiroti:8100/ask"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}
    assert captured["json"] == {"prompt": "hi"}


def test_client_ask_auth_error_raises_authentication_error(monkeypatch, configured):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeResponse(401, {"message": "Invalid token"}))

    with pytest.raises(AuthenticationError):
        client.ask("hi")


def test_client_connection_refused_raises_chiroti_connection_error(monkeypatch, configured):
    def raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", raise_connect_error)

    with pytest.raises(ChirotiConnectionError):
        client.ask("hi")


def test_client_config_precedence(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"server": "http://from-file", "token": "file-token"}')
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    assert config.get_server() == "http://from-file"

    monkeypatch.setenv("CHIROTI_SERVER", "http://from-env")
    assert config.get_server() == "http://from-env"

    config.configure(server="http://from-arg")
    assert config.get_server() == "http://from-arg"


def test_client_ask_passes_arbitrary_openai_kwargs_through_unmodified(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "qwen-text"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("hi", temperature=0.9, top_p=0.95, stop=["\n\n"], seed=42)

    assert captured["json"] == {
        "prompt": "hi",
        "temperature": 0.9,
        "top_p": 0.95,
        "stop": ["\n\n"],
        "seed": 42,
    }
