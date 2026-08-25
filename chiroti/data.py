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

MAX_CSV_ROWS = 2000
MAX_NPZ_INLINE_VALUES = 200


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv_module.DictReader(f))
    if not rows:
        raise InvalidInputError(f"{path} is empty")
    return rows


def _csv_files_to_json(paths: list[Path]) -> dict:
    columns = None
    all_rows = []
    for path in paths:
        rows = _read_csv_rows(path)
        if columns is None:
            columns = list(rows[0].keys())
        elif list(rows[0].keys()) != columns:
            raise InvalidInputError(
                f"{path} has different columns than {paths[0]}: {list(rows[0].keys())} vs {columns}"
            )
        all_rows.extend(rows)

    if len(all_rows) > MAX_CSV_ROWS:
        raise InvalidInputError(f"{len(all_rows)} rows across all csv files exceeds the {MAX_CSV_ROWS}-row limit")

    return {"source_files": [p.name for p in paths], "columns": columns, "rows": all_rows}


def _npz_file_to_json(path: Path) -> dict:
    archive = np.load(path)
    arrays = {}
    for name in archive.files:
        arr = archive[name]
        entry = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.size <= MAX_NPZ_INLINE_VALUES:
            entry["values"] = arr.tolist()
        else:
            flat = arr.reshape(-1)
            entry["stats"] = {
                "min": float(flat.min()), "max": float(flat.max()),
                "mean": float(flat.mean()), "std": float(flat.std()),
            }
        arrays[name] = entry
    return arrays


def data_to_text(paths: list[str]) -> str:
    """CSVs are appended into one table; each NPZ is described separately."""
    resolved = [Path(p) for p in paths]
    csv_paths = [p for p in resolved if p.suffix.lower() == ".csv"]
    npz_paths = [p for p in resolved if p.suffix.lower() == ".npz"]
    unknown = [p for p in resolved if p.suffix.lower() not in (".csv", ".npz")]
    if unknown:
        raise InvalidInputError(f"unsupported data file type(s): {unknown} — only .csv and .npz are supported")

    payload = {}
    if csv_paths:
        payload["csv_data"] = _csv_files_to_json(csv_paths)
    if npz_paths:
        payload["npz_data"] = {p.name: _npz_file_to_json(p) for p in npz_paths}

    return "### Data\n```json\n" + json.dumps(payload, indent=2) + "\n```"
