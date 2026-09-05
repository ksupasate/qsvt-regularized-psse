"""Shared scaffolding for the Phase 10 implementation-completion packages.

Phase 10 adds executable or fully auditable implementations for the remaining
QSVT workflow components (complete 8x8 sparse block-encoding wrapper, full
rectangular selected-output QSVT, explicit residual loading with repeat-cost
accounting, nonlinear AC QSVT-in-the-loop simulation, and an end-to-end
resource ledger).  Nothing here changes the estimator definitions: the target
stays the Ridge/Tikhonov spectral filter ``sigma / (sigma^2 + alpha)`` and the
QSVT path implements the bounded normalized version of the *same* filter at
the *same* alpha, with the single physical rescale factor ``C/beta``.

All generated prose passes the repository forbidden-phrase gates.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robust_qsvt_se.paper.selected_observable_qsvt_common import assert_safe, forbidden_in
from robust_qsvt_se.paper.tqe_revision_support_common import (
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

__all__ = [
    "PHASE10_CLAIM_BOUNDARY",
    "assert_safe",
    "forbidden_in",
    "json_ready",
    "sha256_file",
    "write_checksums",
    "write_command_log",
    "write_phase10_manifest",
]

PHASE10_CLAIM_BOUNDARY = (
    "Phase 10 implementation-completion artifacts for a controlled IEEE/PYPOWER benchmark "
    "study of QSVT-compatible implementation pathways for the same Ridge/Tikhonov "
    "regularized spectral filter at the same alpha. All executions are classical "
    "simulations (statevector or sampled counts); none is a quantum-hardware run. "
    "IEEE/PYPOWER cases provide benchmark network models with generated measurement rows, "
    "not field PMU/SCADA data. These artifacts do not claim speedup, do not claim "
    "QSVT-over-Ridge numerical superiority at matched alpha, do not claim IEEE-scale "
    "compiled sparse block encodings beyond the sizes actually validated, and do not "
    "claim practical competitiveness with classical solvers."
)


def json_ready(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``write_json`` can serialize."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: str | Path, artifacts: dict[str, Path]) -> Path:
    """Write ``checksums.sha256`` over the named artifacts (sorted by file name)."""

    directory = ensure_directory(output_dir)
    checksum_path = directory / "checksums.sha256"
    lines = [
        f"{sha256_file(path)}  {Path(path).name}\n"
        for _, path in sorted(artifacts.items(), key=lambda item: Path(item[1]).name)
        if Path(path).is_file()
    ]
    checksum_path.write_text("".join(lines), encoding="utf-8")
    return checksum_path


def write_command_log(output_dir: str | Path, command: str) -> Path:
    """Record the exact reproduction command line for this output package."""

    directory = ensure_directory(output_dir)
    log_path = directory / "command_log.txt"
    log_path.write_text(
        f"{now_iso()}  python={sys.version.split()[0]}\n{command}\n", encoding="utf-8"
    )
    return log_path


def write_phase10_manifest(
    *,
    output_dir: str | Path,
    experiment_id: str,
    script_name: str,
    command: str,
    description: str,
    artifacts: dict[str, Path],
    seeds: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the Phase 10 provenance manifest (command, checksums, versions)."""

    directory = ensure_directory(output_dir)
    command_log = write_command_log(directory, command)
    artifacts = dict(artifacts)
    artifacts.setdefault("command_log_txt", command_log)
    checksum_path = write_checksums(directory, artifacts)
    artifacts.setdefault("checksums_sha256", checksum_path)
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "script_name": script_name,
        "command": command,
        "description": description,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "seed_provenance": {
            "status": "recorded" if seeds else "not_applicable",
            "seeds": {name: int(value) for name, value in (seeds or {}).items()},
        },
        "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
        "checksums": {
            Path(path).name: sha256_file(path)
            for _, path in sorted(artifacts.items())
            if Path(path).is_file()
        },
        "key_package_versions": package_versions(
            ["numpy", "pandas", "scipy", "pennylane", "qiskit", "qiskit-aer", "pypower"]
        ),
        "claim_boundary": PHASE10_CLAIM_BOUNDARY,
        "changes_estimator_behavior": False,
        "fabricates_results": False,
    }
    if extra:
        manifest.update(json_ready(extra))
    write_json(manifest_path, json_ready(manifest))
    return manifest_path
