"""HTTP calls to the Chiroti server."""

from typing import Any

import httpx

from chiroti.config import get_server, get_token
from chiroti.exceptions import (
    AuthenticationError,
    ChirotiConnectionError,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
)

_STATUS_TO_ERROR = {
    400: InvalidInputError,
    401: AuthenticationError,
    404: ModelNotFoundError,
    502: InferenceError,
}


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{get_server().rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    try:
        response = httpx.request(method, url, headers=headers, **kwargs)
    except httpx.ConnectError as e:
        raise ChirotiConnectionError(f"could not reach {url}: {e}")

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
    max_tokens: int | None = None,
    reasoning: bool | None = None,
    output_format=None,
    cache: bool | None = None,
    **openai_kwargs: Any,
) -> str:
    if not prompt.strip():
        raise InvalidInputError("prompt must not be empty")

    not_yet_implemented = {
        "image": image,
        "document": document,
        "reasoning": reasoning,
        "output_format": output_format,
        "cache": cache,
    }
    for name, value in not_yet_implemented.items():
        if value is not None:
            raise NotImplementedError(f"{name}= is not implemented yet")

    payload = {"prompt": prompt, **openai_kwargs}
    if model is not None:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return _request("POST", "/ask", json=payload)["text"]


def models() -> list[str]:
    body = _request("GET", "/models")
    return [entry["name"] for entry in body]
