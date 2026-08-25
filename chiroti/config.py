"""Resolves server URL and token. Precedence: configure() call > env var > config file > default."""

import json
import os
from pathlib import Path

CONFIG_FILE = Path.home() / ".chiroti" / "config.json"

_overrides = {}


def configure(server: str | None = None, token: str | None = None) -> None:
    if server is not None:
        _overrides["server"] = server
    if token is not None:
        _overrides["token"] = token


def _from_file(key: str) -> str | None:
    if not CONFIG_FILE.exists():
        return None
    return json.loads(CONFIG_FILE.read_text()).get(key)


def get_server() -> str:
    value = _overrides.get("server") or os.environ.get("CHIROTI_SERVER") or _from_file("server")
    if not value:
        raise RuntimeError("No Chiroti server configured — set CHIROTI_SERVER or call chiroti.configure(server=...)")
    return value


def get_token() -> str:
    value = _overrides.get("token") or os.environ.get("CHIROTI_TOKEN") or _from_file("token")
    if not value:
        raise RuntimeError("No Chiroti token configured — set CHIROTI_TOKEN or call chiroti.configure(token=...)")
    return value
