from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=True)


def write_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, allow_nan=True)
