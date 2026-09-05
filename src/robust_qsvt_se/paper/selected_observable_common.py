"""Shared scaffolding for the QSVT selected-observable workload package.

This package adds a concrete *implementation pathway* layer on top of the
existing benchmark. It does **not** change any estimator. Ridge/Tikhonov stays
the matched classical reference, and the QSVT-compatible target implements the
same regularized spectral filter ``sigma / (sigma^2 + alpha)``. Every artifact
produced here is feasibility and boundary evidence, not a speedup result, not
full-vector readout, not field-data validation, and not a quantum-hardware run.

The module centralises:

* the single output root for all workload artifacts,
* the conservative claim-boundary wording,
* a *dual* forbidden-phrase self-check (the repository-wide list from
  :mod:`tqe_revision_support_common` plus the workload-specific banned
  overclaims spelled out in the task brief),
* a provenance-manifest writer with checksums.

Nothing here changes solver mathematics, alpha values, or existing outputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper.tqe_revision_support_common import (
    find_forbidden,
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

# All workload artifacts live under this root. Existing outputs are never touched.
WORKLOAD_DIR = Path("outputs/qsvt_selected_observable_workload")

# Conservative framing reused verbatim in every generated report header.
WORKLOAD_CLAIM_BOUNDARY = (
    "Controlled IEEE/PYPOWER benchmark and QSVT feasibility study for regularized "
    "spectral filtering in ill-conditioned power-system state estimation. IEEE/PYPOWER "
    "cases provide benchmark network models; measurement rows are generated from the "
    "network equations (not field PMU/SCADA data). The QSVT component is an "
    "implementation pathway for the same Ridge/Tikhonov regularized spectral filter, "
    "evaluated at the same alpha. This work does not claim speedup, QSVT-over-Ridge "
    "numerical superiority, full-vector quantum readout, or full IEEE-scale execution "
    "on quantum hardware."
)

# Exact overclaim strings the task brief bans (matched case-insensitively as
# substrings). These are checked *in addition* to the repository-wide list, so
# generated text must avoid both sets.
WORKLOAD_BANNED_OVERCLAIMS = (
    "quantum speedup demonstrated",
    "qsvt outperforms ridge",
    "qsvt beats ridge",
    "field pmu/scada validation",
    "pmu/scada validation",
    "full ieee-scale hardware execution",
    "full-vector readout solved",
    "full vector readout solved",
)

# Mandatory boundary statements for cost/readout reports, pre-phrased so they
# pass both forbidden-phrase checks (e.g. avoiding the bare "hardware execution"
# and "quantum advantage" substrings while still stating the boundary).
MANDATORY_BOUNDARY_STATEMENTS = (
    "This is selected-observable accounting over selected linear functionals.",
    "This is feasibility and boundary evidence, not a speedup result and not a claim of advantage.",
    "This is not full-vector readout; each estimate recovers one selected functional.",
    "This is not executed on quantum hardware; there is no quantum-hardware run.",
    "Sparse access and state preparation are modeled unless implemented and tested otherwise.",
)


def assert_safe(text: str) -> None:
    """Raise if ``text`` contains any repository-wide or workload-banned overclaim."""

    violations = forbidden_in(text)
    if violations:
        raise RuntimeError(f"generated text contains forbidden wording: {sorted(violations)}")


def forbidden_in(text: str) -> list[str]:
    """Return all forbidden phrases present in ``text`` (both banned sets)."""

    lowered = text.lower()
    repo_hits = find_forbidden(text)
    workload_hits = [phrase for phrase in WORKLOAD_BANNED_OVERCLAIMS if phrase in lowered]
    return sorted(set(repo_hits) | set(workload_hits))


def checksum(path: str | Path) -> str | None:
    """Return the SHA-256 hex digest of ``path`` or ``None`` if it is absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_workload_manifest(
    *,
    output_dir: str | Path,
    artifact_name: str,
    description: str,
    command: str,
    artifacts: dict[str, Path],
    input_files: list[str] | None = None,
    reran_long_experiments: bool = False,
    aggregated_from_existing: bool = False,
    extra: dict[str, Any] | None = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Write a provenance manifest with checksums, git hash, and command line."""

    directory = ensure_directory(output_dir)
    path = directory / manifest_name
    generated = {name: str(value) for name, value in sorted(artifacts.items())}
    generated["manifest"] = str(path)
    checksums = {name: checksum(value) for name, value in sorted(artifacts.items())}
    manifest: dict[str, Any] = {
        "artifact_name": artifact_name,
        "description": description,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "command_used": command,
        "input_files": sorted(input_files or []),
        "generated_file_paths": generated,
        "checksums": checksums,
        "reran_long_experiments": bool(reran_long_experiments),
        "aggregated_from_existing": bool(aggregated_from_existing),
        "key_package_versions": package_versions(["numpy", "pandas", "scipy", "matplotlib"]),
        "claim_boundary": WORKLOAD_CLAIM_BOUNDARY,
        "changes_estimator_behavior": False,
        "fabricates_results": False,
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest)
    return path
