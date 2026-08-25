"""HTTP calls to the Chiroti server."""

from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from chiroti.config import get_server, get_token
from chiroti.data import data_to_text
from chiroti.exceptions import (
    AuthenticationError,
    ChirotiConnectionError,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    OutputValidationError,
    UnsupportedFeatureError,
)

_STATUS_TO_ERROR = {
    400: InvalidInputError,
    401: AuthenticationError,
    404: ModelNotFoundError,
    422: UnsupportedFeatureError,
    502: InferenceError,
}

# httpx's default is 5s, far too short for LLM generation; give it minutes instead.
_DEFAULT_TIMEOUT_SECONDS = 300.0


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{get_server().rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    try:
        response = httpx.request(method, url, headers=headers, timeout=_DEFAULT_TIMEOUT_SECONDS, **kwargs)
    except httpx.ConnectError as e:
        raise ChirotiConnectionError(f"could not reach {url}: {e}")
    except httpx.TimeoutException as e:
        raise InferenceError(f"no response from {url} within {_DEFAULT_TIMEOUT_SECONDS:.0f}s: {e}")

    if response.is_success:
        return response.json()

    body = response.json() if response.content else {}
    error_cls = _STATUS_TO_ERROR.get(response.status_code, InferenceError)
    raise error_cls(body.get("message", response.text))


def ask(
    prompt: str,
    *,
    model: str | None = None,
    image=None,
    document=None,
    data: str | Path | list[str | Path] | None = None,
    max_tokens: int | None = None,
    reasoning: bool | None = None,
    output_format: type[BaseModel] | None = None,
    cache: bool | None = None,
    **openai_kwargs: Any,
) -> str | BaseModel:
    if not prompt.strip():
        raise InvalidInputError("prompt must not be empty")

    not_yet_implemented = {"image": image, "document": document, "reasoning": reasoning, "cache": cache}
    for name, value in not_yet_implemented.items():
        if value is not None:
            raise NotImplementedError(f"{name}= is not implemented yet")

    if data is not None:
        paths = [data] if isinstance(data, (str, Path)) else list(data)
        prompt = f"{prompt}\n\n{data_to_text(paths)}"

    payload = {"prompt": prompt, **openai_kwargs}
    if model is not None:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if output_format is not None:
        payload["output_format"] = output_format.model_json_schema()

    text = _request("POST", "/ask", json=payload)["text"]

    if output_format is None:
        return text
    try:
        return output_format.model_validate_json(text)
    except ValidationError as e:
        raise OutputValidationError(f"model output didn't match output_format: {e}", raw_text=text) from e


def models() -> list[str]:
    body = _request("GET", "/models")
    return [entry["name"] for entry in body]
