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
from chiroti.response import ChirotiResponse, LabnotesResponse

_STATUS_TO_ERROR = {
    400: InvalidInputError,
    401: AuthenticationError,
    404: ModelNotFoundError,
    422: UnsupportedFeatureError,
    502: InferenceError,
}

# httpx's default is 5s, far too short for LLM generation; give it minutes instead.
_DEFAULT_TIMEOUT_SECONDS = 300.0
# labnotes() may run several sequential model + Labnotes API round trips server-side.
_LABNOTES_TIMEOUT_SECONDS = 900.0


def _request(method: str, path: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS, **kwargs: Any) -> Any:
    url = f"{get_server().rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    try:
        response = httpx.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except httpx.ConnectError as e:
        raise ChirotiConnectionError(f"could not reach {url}: {e}")
    except httpx.TimeoutException as e:
        raise InferenceError(f"no response from {url} within {timeout:.0f}s: {e}")

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
    reasoning: bool = True,
    output_format: type[BaseModel] | None = None,
    cache: bool | None = None,
    **openai_kwargs: Any,
) -> ChirotiResponse:
    if not prompt.strip():
        raise InvalidInputError("prompt must not be empty")

    not_yet_implemented = {"image": image, "document": document, "cache": cache}
    for name, value in not_yet_implemented.items():
        if value is not None:
            raise NotImplementedError(f"{name}= is not implemented yet")

    if data is not None:
        paths = [data] if isinstance(data, (str, Path)) else list(data)
        prompt = f"{prompt}\n\n{data_to_text(paths)}"

    payload = {"prompt": prompt, "reasoning": reasoning, **openai_kwargs}
    if model is not None:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if output_format is not None:
        payload["output_format"] = output_format.model_json_schema()

    body = _request("POST", "/ask", json=payload)
    text = body["text"]

    parsed = None
    if output_format is not None:
        try:
            parsed = output_format.model_validate_json(text)
        except ValidationError as e:
            raise OutputValidationError(f"model output didn't match output_format: {e}", raw_text=text) from e

    return ChirotiResponse(
        text=text,
        token_count=body.get("token_count"),
        prompt_tokens=body.get("prompt_tokens"),
        ttft=body.get("ttft"),
        tps=body.get("tps"),
        reasoning=body.get("reasoning"),
        parsed=parsed,
    )


def models() -> list[str]:
    body = _request("GET", "/models")
    return [entry["name"] for entry in body]


def labnotes(
    question: str,
    *,
    type: str | None = None,
    author: str | None = None,
    title: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    keyword: str | None = None,
    text: str | None = None,
    limit: int | None = None,
) -> LabnotesResponse:
    """Answers a natural-language question over Bhalla Lab notes. Any filter
    passed here (author=, type=, ...) is a hard constraint the server enforces
    on every search it runs — the model can only search within it."""
    if not question.strip():
        raise InvalidInputError("question must not be empty")

    payload = {"question": question}
    for key, value in {
        "type": type, "author": author, "title": title,
        "created_from": created_from, "created_to": created_to,
        "keyword": keyword, "text": text, "limit": limit,
    }.items():
        if value is not None:
            payload[key] = value

    body = _request("POST", "/labnotes", json=payload, timeout=_LABNOTES_TIMEOUT_SECONDS)
    return LabnotesResponse(answer=body["answer"], sources=body.get("sources", []), attachments=body.get("attachments", []))
