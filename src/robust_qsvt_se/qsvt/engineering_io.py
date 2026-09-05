from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robust_qsvt_se import __version__
from robust_qsvt_se.utils.io import write_json
from robust_qsvt_se.utils.privacy import sanitize_public_text, sanitize_public_value

CLAIM_BOUNDARY = (
    "QSVT-compatible implementation pathway and resource-aware feasibility "
    "analysis only; no quantum speedup, quantum advantage, full IEEE-scale "
    "hardware execution, or PMU/SCADA field-data validation is claimed."
)


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_command() -> str:
    return sanitize_public_text(" ".join(sys.argv))


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def write_manifest(
    output_dir: str | Path,
    *,
    artifacts: dict[str, str],
    input_config: dict[str, Any],
    command: str | None = None,
    claim_boundary: str = CLAIM_BOUNDARY,
    seed_provenance: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    manifest = {
        "generated_at": utc_timestamp(),
        "command": sanitize_public_text(command) if command else current_command(),
        "input_config": _json_safe(input_config),
        "artifacts": sanitize_public_value(artifacts),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
        "claim_boundary": claim_boundary,
    }
    if seed_provenance is not None:
        manifest["seed_provenance"] = _json_safe(seed_provenance)
    path = output_path / "manifest.json"
    write_json(path, manifest)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return sanitize_public_text(str(value))
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value
