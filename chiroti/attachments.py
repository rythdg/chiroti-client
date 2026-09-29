"""Dispatches ask()'s attachment= paths by extension: csv/npz go through
data.py's existing text-injection path, md/txt are read and injected into the
prompt as text, image/pdf get base64-encoded for upload to the server. File type is inferred from the extension — the caller
never says what kind of file each path is.
"""

import base64
from pathlib import Path

from chiroti.data import DATA_EXTENSIONS, data_to_text
from chiroti.exceptions import InvalidInputError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".md", ".txt"}
SUPPORTED_EXTENSIONS = DATA_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}


def _encode_upload(path: Path) -> dict:
    content_type = MIME_TYPES[path.suffix.lower()]
    data_base64 = base64.b64encode(path.read_bytes()).decode()
    return {"filename": path.name, "content_type": content_type, "data_base64": data_base64}


def _text_to_block(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise InvalidInputError(f"{path.name} is not valid UTF-8 text") from None
    return f"### {path.name}\n```\n{text}\n```"


def prepare_attachments(prompt: str, paths: list[str | Path]) -> tuple[str, list[dict]]:
    resolved = [Path(p) for p in paths]
    unknown = [p for p in resolved if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
    if unknown:
        raise InvalidInputError(
            f"unsupported attachment file type(s): {unknown} — supported extensions are "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    data_paths = [p for p in resolved if p.suffix.lower() in DATA_EXTENSIONS]
    upload_paths = [p for p in resolved if p.suffix.lower() in (IMAGE_EXTENSIONS | PDF_EXTENSIONS)]

    text_paths = [p for p in resolved if p.suffix.lower() in TEXT_EXTENSIONS]

    if data_paths:
        prompt = f"{prompt}\n\n{data_to_text([str(p) for p in data_paths])}"
    for p in text_paths:
        prompt = f"{prompt}\n\n{_text_to_block(p)}"

    return prompt, [_encode_upload(p) for p in upload_paths]
