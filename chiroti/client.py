"""HTTP calls to the Chiroti server."""

from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from chiroti.attachments import prepare_attachments
from chiroti.config import get_server, get_token
from chiroti.exceptions import (
    AuthenticationError,
    ChirotiConnectionError,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    OutputValidationError,
    UnsupportedFeatureError,
)
from chiroti.response import AskResponse, LabnotesResponse

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
_LABNOTES_TIMEOUT_SECONDS = 5000.0


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
    prompt: str | None = None,
    *,
    system: str | None = None,
    user: str | None = None,
    model: str | None = None,
    attachment: str | Path | list[str | Path] | None = None,
    max_tokens: int | None = None,
    reasoning: bool = True,
    output_format: type[BaseModel] | None = None,
    cache: bool | None = None,
    **openai_kwargs: Any,
) -> AskResponse:
    if prompt is not None and user is not None:
        raise InvalidInputError("pass the prompt either positionally or as user=, not both")
    prompt = prompt if prompt is not None else user

    if not prompt or not prompt.strip():
        raise InvalidInputError("prompt must not be empty")

    if cache is not None:
        raise NotImplementedError("cache= is not implemented yet")

    uploads = []
    if attachment is not None:
        paths = [attachment] if isinstance(attachment, (str, Path)) else list(attachment)
        prompt, uploads = prepare_attachments(prompt, paths)

    payload = {"prompt": prompt, "reasoning": reasoning, **openai_kwargs}
    if system is not None:
        payload["system"] = system
    if model is not None:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if output_format is not None:
        payload["output_format"] = output_format.model_json_schema()
    if uploads:
        payload["attachments"] = uploads

    body = _request("POST", "/ask", json=payload)
    text = body["text"]

    parsed = None
    if output_format is not None:
        try:
            parsed = output_format.model_validate_json(text)
        except ValidationError as e:
            raise OutputValidationError(f"model output didn't match output_format: {e}", raw_text=text) from e

    return AskResponse(
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
    reasoning: bool = True,
) -> LabnotesResponse:
    """Answers a natural-language question over Bhalla Lab notes. Any filter
    passed here (author=, type=, ...) is a hard constraint the server enforces
    on every search it runs — the model can only search within it."""
    if not question.strip():
        raise InvalidInputError("question must not be empty")

    payload = {"question": question, "reasoning": reasoning}
    for key, value in {
        "type": type, "author": author, "title": title,
        "created_from": created_from, "created_to": created_to,
        "keyword": keyword, "text": text, "limit": limit,
    }.items():
        if value is not None:
            payload[key] = value

    body = _request("POST", "/labnotes", json=payload, timeout=_LABNOTES_TIMEOUT_SECONDS)
    return LabnotesResponse(
        text=body["text"],
        sources=body.get("sources", []),
        attachments=body.get("attachments", []),
        usage=body.get("usage", {}),
    )
