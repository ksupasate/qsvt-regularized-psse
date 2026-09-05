"""Phase 0: evidence freeze and reproducibility lock.

Snapshots the current manuscript-assembly package so the paper can cite a stable
artifact set: a content-addressed artifact manifest (sha256 of each indexed file),
a claim-matrix snapshot, a test-quality snapshot, the resolved git hash (or
``git_hash_unavailable``), the Python/dependency state, and the canonical
reproduction commands. It reads existing artifacts only and never fabricates
values; a missing input is recorded as absent rather than invented.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import git_commit, utc_timestamp, write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/freeze_manuscript_evidence.py"

ARTIFACT_MANIFEST_COLUMNS = [
    "artifact_path",
    "artifact_type",
    "size_bytes",
    "sha256",
    "status",
]
CLAIM_SNAPSHOT_COLUMNS = [
    "claim_id",
    "claim_text",
    "manuscript_section",
    "support_status",
    "limitation_note",
]
TEST_SUMMARY_COLUMNS = [
    "test_category",
    "count",
    "counts_as_scientific_validation",
]

# Core scientific dependencies whose versions are recorded for the freeze.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pyyaml",
    "networkx",
    "pypower",
    "qiskit",
    "pennylane",
)
_NON_VALIDATION_CATEGORIES = frozenset({"smoke_only", "unit_behavior"})

# Canonical verification commands recorded in the freeze (executed separately).
FULL_TEST_COMMAND = ".venv/bin/python -m pytest"
RUFF_CHECK_COMMAND = ".venv/bin/ruff check src scripts tests"
RUFF_FORMAT_COMMAND = ".venv/bin/ruff format --check src scripts tests"


def build_evidence_freeze(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "evidence_freeze"))
    # Verification status is recorded only when the caller actually ran the commands;
    # otherwise the freeze states the command was not executed in this process.
    test_status = str(config.get("test_status", "not_executed_in_freeze"))
    ruff_status = str(config.get("ruff_status", "not_executed_in_freeze"))

    artifact_rows = _artifact_manifest_rows(package_root)
    claim_rows = _claim_snapshot_rows(package_root)
    test_rows, test_meta = _test_snapshot(package_root)
    commit = git_commit()
    environment = _environment_snapshot()
    counts = _claim_category_counts(claim_rows)

    return _write_outputs(
        output_dir=output_dir,
        package_root=package_root,
        artifact_rows=artifact_rows,
        claim_rows=claim_rows,
        test_rows=test_rows,
        test_meta=test_meta,
        commit=commit,
        environment=environment,
        claim_counts=counts,
        test_status=test_status,
        ruff_status=ruff_status,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
            "test_status": test_status,
            "ruff_status": ruff_status,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest_rows(package_root: Path) -> list[dict[str, Any]]:
    index = read_csv(package_root / "manuscript_artifact_index.csv")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths: list[Path] = []
    if not index.empty and "artifact_path" in index.columns:
        paths.extend(Path(str(p)) for p in index["artifact_path"])
    # Always include the top-level package files so the freeze is self-contained.
    for top in sorted(package_root.glob("*")):
        if top.is_file() and top.suffix.lower() in {".csv", ".md", ".json"}:
            paths.append(top)
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            rows.append(
                {
                    "artifact_path": key,
                    "artifact_type": path.suffix.lower().lstrip(".") or "other",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "status": "present",
                }
            )
        else:
            rows.append(
                {
                    "artifact_path": key,
                    "artifact_type": path.suffix.lower().lstrip(".") or "other",
                    "size_bytes": "",
                    "sha256": "",
                    "status": "missing",
                }
            )
    return rows


def _claim_snapshot_rows(package_root: Path) -> list[dict[str, Any]]:
    matrix = read_csv(package_root / "claim_support_matrix_final.csv")
    if matrix.empty:
        return []
    rows = []
    for _, record in matrix.iterrows():
        rows.append(
            {
                "claim_id": record.get("claim_id", ""),
                "claim_text": record.get("claim_text", ""),
                "manuscript_section": record.get("manuscript_section", ""),
                "support_status": record.get("support_status", ""),
                "limitation_note": record.get("limitation_note", ""),
            }
        )
    return rows


def _test_snapshot(package_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = read_csv(package_root / "test_quality_audit" / "test_inventory.csv")
    if inventory.empty or "test_category" not in inventory.columns:
        return [], {"total": 0, "smoke_only": 0, "real_validation": 0, "relies_on_smoke": False}
    counts = inventory["test_category"].value_counts().to_dict()
    rows = [
        {
            "test_category": category,
            "count": int(count),
            "counts_as_scientific_validation": (
                "no" if category in _NON_VALIDATION_CATEGORIES else "yes"
            ),
        }
        for category, count in sorted(counts.items())
    ]
    real = int(counts.get("real_validation", 0))
    guard = int(counts.get("claim_boundary_guard", 0))
    meta = {
        "total": len(inventory),
        "smoke_only": int(counts.get("smoke_only", 0)),
        "real_validation": real,
        "relies_on_smoke": real == 0 and guard == 0,
    }
    return rows, meta


def _environment_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            snapshot[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            snapshot[package] = "not_installed"
    return snapshot


def _claim_category_counts(claim_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in claim_rows:
        status = str(row["support_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _evidence_freeze_markdown(
    *,
    package_root: Path,
    artifact_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    test_meta: dict[str, Any],
    commit: str | None,
    environment: dict[str, str],
    claim_counts: dict[str, int],
    test_status: str,
    ruff_status: str,
) -> str:
    import platform

    n_tables = _count_index(package_root / "manuscript_table_index.csv", "paper_ready", "status")
    n_figures = _count_index(package_root / "manuscript_figure_index.csv", "data_ready", "status")
    n_artifacts = len([r for r in artifact_rows if r["status"] == "present"])
    deps = "; ".join(f"{name}={version}" for name, version in environment.items())
    lines = [
        "# Evidence Freeze",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "## Freeze metadata",
        f"- Timestamp (UTC): {utc_timestamp()}",
        f"- Git commit: {commit if commit else 'git_hash_unavailable'}",
        f"- Python version: {platform.python_version()}",
        f"- Dependency snapshot: {deps}",
        f"- Final output package path: {package_root}",
        "",
        "## Indexed evidence",
        f"- Paper-ready tables indexed: {n_tables}.",
        f"- Figure datasets ready: {n_figures}.",
        f"- Artifacts frozen (content-addressed in frozen_artifact_manifest.csv): {n_artifacts}.",
        "",
        "## Verification commands",
        f"- Full test suite: `{FULL_TEST_COMMAND}` -> status: {test_status}.",
        f"- Ruff check: `{RUFF_CHECK_COMMAND}` -> status: {ruff_status}.",
        f"- Ruff format check: `{RUFF_FORMAT_COMMAND}` -> status: {ruff_status}.",
        "- Statuses marked `not_executed_in_freeze` are run separately; see reproducibility.md "
        "and final_reproduction_log.txt.",
        "",
        "## Claim-matrix category counts",
        *[f"- {status}: {count}" for status, count in sorted(claim_counts.items())],
        "",
        "## Test-quality snapshot",
        f"- Total test functions inventoried: {test_meta['total']}.",
        f"- Real-validation tests: {test_meta['real_validation']}; smoke-only: "
        f"{test_meta['smoke_only']} (smoke tests are NOT counted as scientific validation).",
        f"- Evidence package relies on smoke tests only: "
        f"{'yes' if test_meta['relies_on_smoke'] else 'no'}.",
        "",
        "## Unsupported-claim preservation",
        "- Unsupported claims remain unsupported in this freeze: QSVT-over-Ridge superiority, "
        "quantum speedup/advantage, and real PMU/SCADA validation are not promoted. Full-vector "
        "readout and full IEEE-scale gate-level execution remain assumption-only / out of scope.",
        "",
    ]
    return "\n".join(lines)


def _count_index(path: Path, ready_value: str, column: str) -> int:
    frame = read_csv(path)
    if frame.empty or column not in frame.columns:
        return 0
    return int((frame[column].astype(str) == ready_value).sum())


def _reproduction_log(commit: str | None, environment: dict[str, str], test_status: str) -> str:
    deps = "\n".join(f"  {name}={version}" for name, version in environment.items())
    return "\n".join(
        [
            "# Final Reproduction Log",
            "",
            f"git_commit: {commit if commit else 'git_hash_unavailable'}",
            f"frozen_at_utc: {utc_timestamp()}",
            "",
            "dependency_snapshot:",
            deps,
            "",
            "verification_commands:",
            f"  {FULL_TEST_COMMAND}  # status: {test_status}",
            f"  {RUFF_CHECK_COMMAND}",
            f"  {RUFF_FORMAT_COMMAND}",
            "",
            "rebuild: see outputs/final_manuscript_package/reproducibility.md and "
            "final_rebuild_commands.md",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    package_root: Path,
    artifact_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    test_meta: dict[str, Any],
    commit: str | None,
    environment: dict[str, str],
    claim_counts: dict[str, int],
    test_status: str,
    ruff_status: str,
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    artifact_path = rows_to_table(
        artifact_rows, output_dir / "frozen_artifact_manifest.csv", ARTIFACT_MANIFEST_COLUMNS
    )
    claim_path = rows_to_table(
        claim_rows, output_dir / "frozen_claim_matrix_snapshot.csv", CLAIM_SNAPSHOT_COLUMNS
    )
    test_path = rows_to_table(
        test_rows, output_dir / "frozen_test_summary.csv", TEST_SUMMARY_COLUMNS
    )
    freeze_md_path = output_dir / "EVIDENCE_FREEZE.md"
    commit_path = output_dir / "final_commit_hash.txt"
    log_path = output_dir / "final_reproduction_log.txt"

    freeze_md_path.write_text(
        _evidence_freeze_markdown(
            package_root=package_root,
            artifact_rows=artifact_rows,
            claim_rows=claim_rows,
            test_meta=test_meta,
            commit=commit,
            environment=environment,
            claim_counts=claim_counts,
            test_status=test_status,
            ruff_status=ruff_status,
        ),
        encoding="utf-8",
    )
    commit_path.write_text((commit if commit else "git_hash_unavailable") + "\n", encoding="utf-8")
    log_path.write_text(_reproduction_log(commit, environment, test_status), encoding="utf-8")

    artifacts = {
        "EVIDENCE_FREEZE": str(freeze_md_path),
        "frozen_artifact_manifest": str(artifact_path),
        "frozen_claim_matrix_snapshot": str(claim_path),
        "frozen_test_summary": str(test_path),
        "final_commit_hash": str(commit_path),
        "final_reproduction_log": str(log_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "artifact_rows": artifact_rows,
        "claim_rows": claim_rows,
        "test_rows": test_rows,
        "commit": commit if commit else "git_hash_unavailable",
        "claim_counts": claim_counts,
        "environment": environment,
    }
