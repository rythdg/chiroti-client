"""Converts local CSV/NPZ files into a JSON text block appended to the prompt.

This is pure client-side text preparation — no new wire format, no server
changes. Chiroti has no dedicated "tabular data" endpoint; data= is just
prompt augmentation before an ordinary /ask request. JSON (rather than raw
CSV lines) keeps each row/array's structure explicit for the model.
"""

import csv as csv_module
import json
from pathlib import Path

import numpy as np

from chiroti.exceptions import InvalidInputError


def _csv_file_to_json(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv_module.DictReader(f))
    if not rows:
        raise InvalidInputError(f"{path} is empty")
    return {"columns": list(rows[0].keys()), "rows": rows}


def _npz_file_to_json(path: Path) -> dict:
    archive = np.load(path)
    arrays = {}
    for name in archive.files:
        arr = archive[name]
        arrays[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "values": arr.tolist()}
    return arrays


def data_to_text(paths: list[str]) -> str:
    """Each file is described under its own filename key — independent of the others."""
    resolved = [Path(p) for p in paths]
    csv_paths = [p for p in resolved if p.suffix.lower() == ".csv"]
    npz_paths = [p for p in resolved if p.suffix.lower() == ".npz"]
    unknown = [p for p in resolved if p.suffix.lower() not in (".csv", ".npz")]
    if unknown:
        raise InvalidInputError(f"unsupported data file type(s): {unknown} — only .csv and .npz are supported")

    payload = {}
    if csv_paths:
        payload["csv_data"] = {p.name: _csv_file_to_json(p) for p in csv_paths}
    if npz_paths:
        payload["npz_data"] = {p.name: _npz_file_to_json(p) for p in npz_paths}

    return "### Data\n```json\n" + json.dumps(payload, indent=2) + "\n```"
