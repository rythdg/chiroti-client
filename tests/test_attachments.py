import base64
import json

import pytest

from chiroti.attachments import prepare_attachments
from chiroti.exceptions import InvalidInputError


def test_csv_and_npz_route_through_data_to_text(tmp_path):
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x,y\n1,2\n")

    prompt, uploads = prepare_attachments("Summarize.", [str(csv_path)])

    assert prompt.startswith("Summarize.\n\n### Data\n```json")
    assert uploads == []


def test_image_attachment_encoded_as_base64_upload(tmp_path):
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")

    prompt, uploads = prepare_attachments("Describe this.", [str(png_path)])

    assert prompt == "Describe this."
    assert len(uploads) == 1
    assert uploads[0]["filename"] == "figure.png"
    assert uploads[0]["content_type"] == "image/png"
    assert base64.b64decode(uploads[0]["data_base64"]) == png_path.read_bytes()


def test_pdf_attachment_encoded_as_base64_upload(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fakepdfbytes")

    prompt, uploads = prepare_attachments("Summarize this.", [str(pdf_path)])

    assert len(uploads) == 1
    assert uploads[0]["filename"] == "paper.pdf"
    assert uploads[0]["content_type"] == "application/pdf"


def test_mixed_image_and_csv_attachment_in_same_call_produces_both_prompt_text_and_uploads(tmp_path):
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x,y\n1,2\n")

    prompt, uploads = prepare_attachments("Analyze this.", [str(png_path), str(csv_path)])

    assert prompt.startswith("Analyze this.\n\n### Data\n```json")
    assert len(uploads) == 1
    assert uploads[0]["content_type"] == "image/png"


def test_unsupported_extension_raises_invalid_input_error_listing_all_supported_extensions(tmp_path):
    zip_path = tmp_path / "notes.zip"
    zip_path.write_text("hello")

    with pytest.raises(InvalidInputError) as exc_info:
        prepare_attachments("Summarize.", [str(zip_path)])

    message = str(exc_info.value)
    for ext in [".csv", ".npz", ".png", ".jpg", ".jpeg", ".pdf", ".md", ".txt"]:
        assert ext in message


@pytest.mark.parametrize("name", ["notes.txt", "notes.md", "NOTES.TXT"])
def test_text_attachment_injected_into_prompt_under_filename_header(tmp_path, name):
    path = tmp_path / name
    path.write_text("# hello\nworld")

    prompt, uploads = prepare_attachments("Summarize.", [str(path)])

    assert prompt == f"Summarize.\n\n### {name}\n```\n# hello\nworld\n```"
    assert uploads == []


def test_mixed_text_and_image_attachment_produces_both_prompt_text_and_upload(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("some notes")
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")

    prompt, uploads = prepare_attachments("Analyze.", [str(md_path), str(png_path)])

    assert "### notes.md" in prompt
    assert [u["content_type"] for u in uploads] == ["image/png"]


@pytest.mark.parametrize("name", ["a.py", "a.c", "a.m", "a.h", "a.json"])
def test_source_code_attachment_injected_as_text(tmp_path, name):
    path = tmp_path / name
    path.write_text("x = 1;")

    prompt, uploads = prepare_attachments("Explain.", [str(path)])

    assert prompt == f"Explain.\n\n### {name}\n```\nx = 1;\n```"
    assert uploads == []


def test_ipynb_keeps_cell_sources_and_drops_outputs(tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "text"]},
            {"cell_type": "code", "source": "print(1)", "outputs": [{"data": {"image/png": "HUGEBASE64"}}]},
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))

    prompt, _ = prepare_attachments("Explain.", [str(path)])

    assert "# [markdown cell 1]\n# Title\ntext" in prompt
    assert "# [code cell 2]\nprint(1)" in prompt
    assert "HUGEBASE64" not in prompt


def test_invalid_ipynb_raises_invalid_input_error(tmp_path):
    path = tmp_path / "nb.ipynb"
    path.write_text("not json")

    with pytest.raises(InvalidInputError, match="nb.ipynb"):
        prepare_attachments("Explain.", [str(path)])


def test_fence_grows_past_backticks_inside_the_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("```python\nx\n```")

    prompt, _ = prepare_attachments("Hi.", [str(path)])

    assert prompt == "Hi.\n\n### notes.md\n````\n```python\nx\n```\n````"


def test_non_utf8_text_attachment_raises_invalid_input_error(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(InvalidInputError, match="bad.txt"):
        prepare_attachments("Summarize.", [str(path)])


def test_content_type_derived_from_extension_not_sniffed(tmp_path):
    jpg_path = tmp_path / "a.jpg"
    jpg_path.write_bytes(b"notreallyajpeg")
    jpeg_path = tmp_path / "b.jpeg"
    jpeg_path.write_bytes(b"notreallyajpegeither")

    _, uploads = prepare_attachments("hi", [str(jpg_path), str(jpeg_path)])

    assert {u["content_type"] for u in uploads} == {"image/jpeg"}
