import json

import numpy as np
import pytest

from chiroti.data import data_to_text
from chiroti.exceptions import InvalidInputError


def _write_csv(path, header, rows):
    path.write_text("\n".join([",".join(header)] + [",".join(row) for row in rows]) + "\n")


def test_single_csv_produces_json_with_rows(tmp_path):
    path = tmp_path / "a.csv"
    _write_csv(path, ["x", "y"], [["1", "2"], ["3", "4"]])

    text = data_to_text([str(path)])
    payload = json.loads(text.split("```json\n")[1].split("\n```")[0])

    assert payload["csv_data"]["a.csv"]["columns"] == ["x", "y"]
    assert payload["csv_data"]["a.csv"]["rows"] == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]


def test_multiple_csvs_are_kept_independent_under_their_own_filename(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, ["x", "y"], [["1", "2"]])
    _write_csv(b, ["p", "q", "r"], [["3", "4", "5"]])

    text = data_to_text([str(a), str(b)])
    payload = json.loads(text.split("```json\n")[1].split("\n```")[0])

    assert payload["csv_data"]["a.csv"]["rows"] == [{"x": "1", "y": "2"}]
    assert payload["csv_data"]["b.csv"]["rows"] == [{"p": "3", "q": "4", "r": "5"}]


def test_large_csv_is_not_truncated(tmp_path):
    path = tmp_path / "a.csv"
    rows = [[str(i)] for i in range(5000)]
    _write_csv(path, ["x"], rows)

    text = data_to_text([str(path)])
    payload = json.loads(text.split("```json\n")[1].split("\n```")[0])

    assert len(payload["csv_data"]["a.csv"]["rows"]) == 5000


def test_npz_array_included_in_full_regardless_of_size(tmp_path):
    path = tmp_path / "a.npz"
    np.savez(path, values=np.arange(5000, dtype="float64"))

    text = data_to_text([str(path)])
    payload = json.loads(text.split("```json\n")[1].split("\n```")[0])

    assert payload["npz_data"]["a.npz"]["values"]["values"] == [float(i) for i in range(5000)]


def test_unsupported_file_type_rejected(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")

    with pytest.raises(InvalidInputError):
        data_to_text([str(path)])
