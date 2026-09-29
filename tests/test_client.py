import base64

import httpx
import pytest
from pydantic import BaseModel

from chiroti import client, config
from chiroti.exceptions import (
    AuthenticationError,
    ChirotiConnectionError,
    InferenceError,
    InvalidInputError,
    OutputValidationError,
    UnsupportedFeatureError,
)


class Experiment(BaseModel):
    organism: str
    temperature: float


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

    assert result.text == "hello"
    assert str(result) == "hello"
    assert captured["url"] == "http://chiroti:8100/ask"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}
    assert captured["json"] == {"prompt": "hi", "reasoning": True}


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


def test_client_timeout_raises_inference_error(monkeypatch, configured):
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("no response")

    monkeypatch.setattr(httpx, "request", raise_timeout)

    with pytest.raises(InferenceError):
        client.ask("hi")


def test_client_output_format_sends_json_schema_and_parses_response(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": '{"organism": "mouse", "temperature": 37.0}', "model": "qwen-text"})

    monkeypatch.setattr(httpx, "request", fake_request)

    result = client.ask("Extract conditions.", output_format=Experiment)

    assert captured["json"]["output_format"] == Experiment.model_json_schema()
    assert result.parsed == Experiment(organism="mouse", temperature=37.0)
    assert result.text == '{"organism": "mouse", "temperature": 37.0}'


def test_client_output_format_invalid_response_raises_output_validation_error_with_raw_text(monkeypatch, configured):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeResponse(200, {"text": "not json", "model": "m"}))

    with pytest.raises(OutputValidationError) as exc_info:
        client.ask("Extract conditions.", output_format=Experiment)
    assert exc_info.value.raw_text == "not json"


def test_client_output_format_on_unsupported_model_raises_unsupported_feature_error(monkeypatch, configured):
    monkeypatch.setattr(
        httpx, "request", lambda *a, **k: FakeResponse(422, {"message": "does not support structured output"})
    )

    with pytest.raises(UnsupportedFeatureError):
        client.ask("Extract conditions.", output_format=Experiment)


def test_client_ask_attachment_csv_appends_json_block_to_prompt(monkeypatch, configured, tmp_path):
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x,y\n1,2\n")
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("Summarize.", attachment=str(csv_path))

    assert captured["json"]["prompt"].startswith("Summarize.\n\n### Data\n```json")
    assert '"x": "1"' in captured["json"]["prompt"]
    assert "attachments" not in captured["json"]


def test_client_ask_attachment_none_sends_plain_prompt_payload(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("hi")

    assert captured["json"] == {"prompt": "hi", "reasoning": True}


def test_client_ask_attachment_image_sends_attachments_list_in_payload(monkeypatch, configured, tmp_path):
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("Describe this.", attachment=str(png_path))

    assert captured["json"]["prompt"] == "Describe this."
    assert len(captured["json"]["attachments"]) == 1
    upload = captured["json"]["attachments"][0]
    assert upload["filename"] == "figure.png"
    assert upload["content_type"] == "image/png"
    assert base64.b64decode(upload["data_base64"]) == png_path.read_bytes()


def test_client_ask_attachment_pdf_sends_attachments_list_in_payload(monkeypatch, configured, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fakepdfbytes")
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("Summarize this.", attachment=str(pdf_path))

    assert len(captured["json"]["attachments"]) == 1
    assert captured["json"]["attachments"][0]["content_type"] == "application/pdf"


def test_client_ask_attachment_mixed_image_and_csv_sends_both_prompt_text_and_attachments_list(
    monkeypatch, configured, tmp_path
):
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x,y\n1,2\n")
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("Analyze this.", attachment=[str(png_path), str(csv_path)])

    assert captured["json"]["prompt"].startswith("Analyze this.\n\n### Data\n```json")
    assert len(captured["json"]["attachments"]) == 1
    assert captured["json"]["attachments"][0]["content_type"] == "image/png"


def test_client_ask_attachment_unsupported_extension_raises_before_any_network_call(monkeypatch, configured, tmp_path):
    zip_path = tmp_path / "notes.zip"
    zip_path.write_text("hello")

    def fake_request(*a, **k):
        raise AssertionError("no network call should happen")

    monkeypatch.setattr(httpx, "request", fake_request)

    with pytest.raises(InvalidInputError):
        client.ask("Summarize.", attachment=str(zip_path))


def test_client_ask_user_kwarg_is_equivalent_to_positional_prompt(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask(user="hi")

    assert captured["json"] == {"prompt": "hi", "reasoning": True}


def test_client_ask_system_kwarg_included_in_payload(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("hi", system="You are terse.")

    assert captured["json"] == {"prompt": "hi", "reasoning": True, "system": "You are terse."}


def test_client_ask_system_omitted_from_payload_when_not_given(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("hi")

    assert "system" not in captured["json"]


def test_client_ask_both_prompt_and_user_raises_invalid_input_error(monkeypatch, configured):
    def fake_request(*a, **k):
        raise AssertionError("no network call should happen")

    monkeypatch.setattr(httpx, "request", fake_request)

    with pytest.raises(InvalidInputError):
        client.ask("hi", user="hi again")


def test_client_ask_neither_prompt_nor_user_raises_invalid_input_error():
    with pytest.raises(InvalidInputError):
        client.ask()


def test_client_ask_system_only_raises_invalid_input_error():
    with pytest.raises(InvalidInputError):
        client.ask(system="You are terse.")


def test_client_get_server_defaults_to_chiroti_host_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent.json")
    assert config.get_server() == config.DEFAULT_SERVER == "http://chiroti:8100"


def test_client_ask_reasoning_false_included_in_payload(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "model": "m"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.ask("hi", reasoning=False)

    assert captured["json"]["reasoning"] is False


def test_client_ask_response_carries_usage_and_timing_metadata(monkeypatch, configured):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeResponse(200, {
        "text": "42",
        "model": "m",
        "token_count": 20,
        "prompt_tokens": 3,
        "ttft": 0.05,
        "tps": 100.0,
        "reasoning": "thinking...",
    }))

    result = client.ask("hi")

    assert result.text == "42"
    assert result.token_count == 20
    assert result.prompt_tokens == 3
    assert result.ttft == 0.05
    assert result.tps == 100.0
    assert result.reasoning == "thinking..."
    assert result.parsed is None


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
        "reasoning": True,
        "temperature": 0.9,
        "top_p": 0.95,
        "stop": ["\n\n"],
        "seed": 42,
    }


def test_labnotes_empty_question_raises_invalid_input_error(configured):
    with pytest.raises(InvalidInputError):
        client.labnotes("")


def test_labnotes_sends_only_explicitly_passed_filters(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, timeout=None, **kwargs):
        captured["json"] = kwargs.get("json")
        captured["timeout"] = timeout
        return FakeResponse(200, {"text": "the answer", "sources": [], "attachments": []})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.labnotes("What did Tannishtha find?", author="tannishtha", type="bhalla_lab_note")

    assert captured["json"] == {
        "question": "What did Tannishtha find?",
        "reasoning": True,
        "author": "tannishtha",
        "type": "bhalla_lab_note",
    }
    assert captured["timeout"] == client._LABNOTES_TIMEOUT_SECONDS


def test_labnotes_reasoning_false_included_in_payload(monkeypatch, configured):
    captured = {}

    def fake_request(method, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse(200, {"text": "ok", "sources": [], "attachments": []})

    monkeypatch.setattr(httpx, "request", fake_request)

    client.labnotes("hi", reasoning=False)

    assert captured["json"]["reasoning"] is False


def test_labnotes_parses_answer_sources_attachments_and_usage(monkeypatch, configured):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeResponse(200, {
        "text": "CaMKII levels rose after stimulation.",
        "sources": [{"nid": 17, "title": "t", "author": "tannishtha", "created": "2020-01-01", "type": "bhalla_lab_note"}],
        "attachments": [{"content_type": "image/png", "size_bytes": 3, "data_base64": "abc"}],
        "usage": {"iterations": 3, "total_prompt_tokens": 500, "total_completion_tokens": 200, "total_time": 12.3},
    }))

    result = client.labnotes("What did Tannishtha find?")

    assert result.text == "CaMKII levels rose after stimulation."
    assert str(result) == result.text
    assert result.sources[0]["nid"] == 17
    assert result.attachments[0]["content_type"] == "image/png"
    assert result.usage["iterations"] == 3
    assert result.usage["total_time"] == 12.3


def test_labnotes_error_mapping_reuses_status_to_error(monkeypatch, configured):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeResponse(422, {"message": "tool calling not supported"}))

    with pytest.raises(UnsupportedFeatureError):
        client.labnotes("hi")
