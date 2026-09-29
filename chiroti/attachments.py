"""Dispatches ask()'s attachment= paths by extension: csv/npz go through
data.py's existing text-injection path, md/txt and source-code files are read
and injected into the prompt as text (.ipynb keeps cell sources only), image/pdf
get base64-encoded for upload to the server. File type is inferred from the
extension — the caller never says what kind of file each path is.
"""

import base64
import json
import re
from pathlib import Path

from chiroti.data import DATA_EXTENSIONS, data_to_text
from chiroti.exceptions import InvalidInputError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}
CODE_EXTENSIONS = {
    ".py", ".ipynb", ".c", ".h", ".cpp", ".cc", ".hpp", ".cu", ".m", ".jl", ".r",
    ".f", ".f90", ".java", ".js", ".ts", ".rs", ".go", ".sh", ".sql", ".tex",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
}
TEXT_EXTENSIONS = {".md", ".txt"} | CODE_EXTENSIONS
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


def _notebook_to_text(raw: str, name: str) -> str:
    """Cell sources only — outputs (often huge base64 images) are dropped."""
    try:
        cells = json.loads(raw)["cells"]
    except (ValueError, KeyError, TypeError):
        raise InvalidInputError(f"{name} is not a valid Jupyter notebook") from None
    parts = []
    for i, cell in enumerate(cells, 1):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        parts.append(f"# [{cell.get('cell_type', 'code')} cell {i}]\n{source}")
    return "\n\n".join(parts)


def _text_to_block(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise InvalidInputError(f"{path.name} is not valid UTF-8 text") from None
    if path.suffix.lower() == ".ipynb":
        text = _notebook_to_text(text, path.name)
    # fence must be longer than any backtick run inside (e.g. a .md with code blocks)
    fence = "`" * max([3, *(len(m) + 1 for m in re.findall(r"`+", text))])
    return f"### {path.name}\n{fence}\n{text}\n{fence}"


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
