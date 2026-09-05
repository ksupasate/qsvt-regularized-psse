"""Shared scaffolding for the TQE-revision support package.

These four artifacts (evidence audit, signed-readout diagnostic, cost
accounting, QSVT boundary diagnosis, claim traceability) are *audit-only*:
they read artifacts already on disk, model clearly-labelled diagnostics, and
never fabricate experiment values. This module centralises the output root,
the claim-boundary wording, a forbidden-wording self-check, and a provenance
manifest writer so every sub-artifact stays consistent and reviewer-safe.

Nothing here changes solver mathematics, alpha values, or existing outputs.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robust_qsvt_se import __version__
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.privacy import sanitize_public_text, sanitize_public_value

# All new artifacts live under this root; existing outputs are never overwritten.
REVISION_OUTPUT_ROOT = Path("outputs/tqe_revision_support")

REVISION_CLAIM_BOUNDARY = (
    "TQE-revision support artifacts for a controlled IEEE/PYPOWER benchmark study of "
    "QSVT-compatible implementation pathways for the same Ridge/Tikhonov regularized "
    "spectral filter. IEEE/PYPOWER cases provide benchmark network models and generated "
    "measurement rows, not real PMU/SCADA field data. These artifacts are an "
    "implementation-pathway, cost-accounting, and boundary analysis. They are not a "
    "quantum-speedup claim, not a QSVT-over-Ridge superiority claim, not full IEEE-scale "
    "hardware execution, not solved full-vector readout, and not field-data validation."
)

# Phrases that must never appear in a manuscript-safe field or generated summary.
# Each entry is matched case-insensitively as a substring.
FORBIDDEN_PHRASES = (
    "quantum speedup",
    "quantum advantage",
    "qsvt beats ridge",
    "qsvt outperforms ridge",
    "qsvt beats tikhonov",
    "beats ridge/tikhonov",
    "full-vector readout solved",
    "full vector readout solved",
    "readout solved",
    "field validated",
    "field-data validation",
    "pmu/scada validation",
    "hardware execution",
    "full ieee-scale quantum",
    "deployment-ready quantum",
    "production-ready quantum",
)


def now_iso() -> str:
    """UTC timestamp as ``YYYY-MM-DDTHH:MM:SSZ`` (no microseconds)."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit_hash() -> str | None:
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
    return completed.stdout.strip() or None


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def find_forbidden(text: str) -> list[str]:
    """Return the forbidden phrases present in ``text`` (case-insensitive)."""

    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def count_truthy(series: Any) -> int:
    """Count truthy entries of a traceability column whether it parsed as bool or string.

    ``pandas`` reads an all-``true``/``false`` CSV column as ``bool`` dtype, so a plain
    ``series == "true"`` comparison silently yields zero. A column mixing in ``partial``
    stays object-typed. Both representations are normalised here so the count is computed
    from the authoritative per-row evidence flags rather than depending on parse dtype.
    """

    if getattr(getattr(series, "dtype", None), "kind", None) == "b":
        return int(series.sum())
    return int(series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum())


def write_manifest(
    *,
    output_dir: str | Path,
    artifact_name: str,
    description: str,
    artifacts: dict[str, Path],
    extra: dict[str, Any] | None = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Write a provenance manifest with git/run/version metadata and file list."""

    directory = ensure_directory(output_dir)
    path = directory / manifest_name
    output_files = {
        name: sanitize_public_text(str(value)) for name, value in sorted(artifacts.items())
    }
    output_files["manifest"] = sanitize_public_text(str(path))
    manifest: dict[str, Any] = {
        "artifact_name": artifact_name,
        "description": description,
        "git_commit_hash": git_commit_hash(),
        "command_used": sanitize_public_text(" ".join(sys.argv)),
        "timestamp": now_iso(),
        "python_version": platform.python_version(),
        "package_version": __version__,
        "key_package_versions": package_versions(["numpy", "pandas", "scipy", "matplotlib"]),
        "output_files_generated": output_files,
        "claim_boundary": REVISION_CLAIM_BOUNDARY,
        "audit_only": True,
        "fabricates_results": False,
    }
    if extra:
        manifest.update(sanitize_public_value(extra))
    write_json(path, manifest)
    return path
