"""Sanitize machine-local paths before publishing research artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_REPO_ROOT = re.compile(r"/Users/[^/\s\"',]+/(?:Desktop|Deskto)/VISTEC_Paper/QSVT_paper")
_HOME = re.compile(r"/Users/[^/\s\"',]+")
_LOCAL_USERNAME = Path.home().name


def sanitize_public_text(value: str, *, strict: bool = False) -> str:
    """Replace machine-local paths with stable public labels.

    ``strict`` additionally removes standalone identity-bearing tokens and is
    reserved for the final package-copy pass. Keeping it off while an artifact
    is generated preserves dereferenceable temporary paths used by tests and
    local workflows; workspace and home-directory paths are sanitized in both
    modes.
    """

    text = value
    replacements = (
        (r"/submission_package_tqe_final(?=/|\b)", "<package_root>"),
        (r"/manuscript/", "/manuscript/"),
        (r"/outputs/", "/outputs/"),
        (r"/scripts/", "/scripts/"),
        (r"/configs/", "/configs/"),
        (r"/src/", "/<repo_root>/src/"),
    )
    for suffix, replacement in replacements:
        text = re.sub(_REPO_ROOT.pattern + suffix, replacement.lstrip("/"), text)
    text = _REPO_ROOT.sub("<repo_root>", text)
    text = _HOME.sub("<home>", text)
    if strict and _LOCAL_USERNAME:
        text = re.sub(
            rf"\b{re.escape(_LOCAL_USERNAME)}\b",
            "<user>",
            text,
        )
    return text


def public_path(path: str | Path) -> str:
    """Return a path suitable for a package-visible report."""

    return sanitize_public_text(str(path))


def sanitize_public_value(value: Any) -> Any:
    """Recursively sanitize strings in JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): sanitize_public_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, Path):
        return public_path(value)
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value
