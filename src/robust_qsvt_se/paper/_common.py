"""Shared helpers for the manuscript assembly package.

Every reader here fails soft: a missing or malformed artifact yields an empty
frame / ``False`` rather than an exception, so the audit can record absence
truthfully instead of fabricating values.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Hard claim-boundary wording shared across all manuscript-package phases.
ALLOWED_WORDING = (
    "controlled IEEE benchmark",
    "generated measurement rows",
    "AC-linearized weighted update",
    "single-step QSVT state-estimation update solver prototype",
    "criteria-selected IEEE-derived 4x4 subproblems",
    "gate-validated selected-subproblem evidence",
    "QSVT-compatible implementation pathway",
    "regularized spectral filter",
    "Ridge/Tikhonov reference",
    "paper-level artifact audit",
    "manuscript-ready evidence package",
)

DISALLOWED_WORDING = (
    "full IEEE-scale quantum state estimator",
    "QSVT beats Ridge",
    "quantum speedup",
    "quantum advantage",
    "real SCADA/PMU validation",
    "field data",
    "deployment-ready quantum solver",
    "full nonlinear AC QSVT loop",
)

# The two overclaims that are unconditionally unsupported regardless of artifacts.
HARD_UNSUPPORTED_CLAIMS = (
    "QSVT beats Ridge/Tikhonov",
    "Quantum speedup or quantum advantage",
)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame if it is absent or unreadable."""

    file_path = Path(path)
    if not file_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def csv_has_rows(path: str | Path) -> bool:
    frame = read_csv(path)
    return not frame.empty


def last_modified(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        stamp = file_path.stat().st_mtime
    except OSError:
        return None
    return (
        datetime.fromtimestamp(stamp, tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def artifact_type(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.is_dir():
        return "directory"
    suffix = file_path.suffix.lower()
    return {
        ".csv": "csv_table",
        ".md": "markdown",
        ".json": "manifest" if file_path.name == "manifest.json" else "json",
        ".png": "figure",
        ".pdf": "figure",
        ".yaml": "config",
        ".yml": "config",
    }.get(suffix, "other")


def write_table(frame: pd.DataFrame, path: str | Path, columns: list[str]) -> Path:
    """Write ``frame`` to CSV with exactly ``columns`` (missing ones filled blank)."""

    out = Path(path)
    ordered = pd.DataFrame(columns=columns)
    if not frame.empty:
        for column in columns:
            ordered[column] = frame[column] if column in frame.columns else ""
    ordered.to_csv(out, index=False)
    return out


def rows_to_table(rows: list[dict], path: str | Path, columns: list[str]) -> Path:
    return write_table(pd.DataFrame(rows), path, columns)


def first_existing(input_root: Path, *relative: str) -> Path | None:
    for rel in relative:
        candidate = input_root / rel
        if candidate.exists():
            return candidate
    return None


def resolved_alpha_map(config_path: str | Path) -> dict[str, Any]:
    """Map estimator name -> regularization alpha from a resolved config.

    Returns an empty map when the config is absent or malformed, so a missing
    alpha is recorded as not-traceable instead of fabricated.
    """

    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    estimators = _find_estimators(config)
    return {
        str(entry.get("name")): entry["alpha"]
        for entry in (estimators or [])
        if isinstance(entry, dict) and entry.get("alpha") is not None
    }


def _find_estimators(obj: Any) -> list[Any] | None:
    if isinstance(obj, dict):
        candidate = obj.get("estimators")
        if isinstance(candidate, list):
            return candidate
        for value in obj.values():
            found = _find_estimators(value)
            if found:
                return found
    return None
