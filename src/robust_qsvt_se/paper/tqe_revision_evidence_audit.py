"""Task A: lightweight repository evidence audit for the TQE revision.

Records which revision-relevant configs, scripts, source modules, output
folders, manuscript-support outputs, docs, and tests exist *before* (and after)
the new evidence artifacts are built. It performs read-only existence checks and
shallow inventories; it never modifies solver behaviour, experiment results, or
existing outputs, and it never fabricates values.

The curated ``EXPECTED_ARTIFACTS`` list is the reviewable backbone: each entry
states which downstream revision task depends on it, so a missing dependency is
surfaced in ``missing_expected_artifacts.csv`` instead of being silently assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper._common import artifact_type, last_modified
from robust_qsvt_se.paper.tqe_revision_support_common import (
    REVISION_OUTPUT_ROOT,
    write_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory

AUDIT_OUTPUT_DIR = REVISION_OUTPUT_ROOT / "evidence_audit"

MANIFEST_COLUMNS = [
    "category",
    "artifact_path",
    "artifact_type",
    "exists",
    "is_dir",
    "child_count",
    "size_bytes",
    "last_modified",
    "needed_by_task",
    "optional",
    "description",
]

MISSING_COLUMNS = [
    "category",
    "artifact_path",
    "needed_by_task",
    "optional",
    "description",
]


@dataclass(frozen=True, slots=True)
class ExpectedArtifact:
    category: str
    path: str
    description: str
    needed_by_task: str = ""
    optional: bool = False


# Curated, reviewer-facing inventory of artifacts that the revision tasks read or
# produce. Existing artifacts are inputs; the ``tqe_revision_support`` entries are
# the outputs the new tasks create (absent on a first audit, present afterwards).
EXPECTED_ARTIFACTS: tuple[ExpectedArtifact, ...] = (
    # --- Project configuration / measurement model -------------------------
    ExpectedArtifact("config", "pyproject.toml", "Project + pytest/ruff configuration."),
    ExpectedArtifact("config", "configs/real_ieee14.yaml", "IEEE14 benchmark config."),
    ExpectedArtifact(
        "config", "configs/qsvt_resource_full_ieee.yaml", "QSVT resource estimate config."
    ),
    ExpectedArtifact(
        "doc",
        "docs/EXPERIMENT_MEASUREMENT_MODEL.md",
        "Generated-measurement model + claim boundaries (claims 1-5).",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "doc", "docs/qsvt_implementation_scope.md", "QSVT implementation-pathway scope."
    ),
    # --- Source modules the new tasks build on -----------------------------
    ExpectedArtifact(
        "source_module",
        "src/robust_qsvt_se/qsvt/filters.py",
        "Ridge/Tikhonov + QSVT-target spectral filters.",
        needed_by_task="B,E",
    ),
    ExpectedArtifact(
        "source_module",
        "src/robust_qsvt_se/estimators/ridge.py",
        "Ridge estimator (matched-alpha reference).",
        needed_by_task="B,E",
    ),
    ExpectedArtifact(
        "source_module",
        "src/robust_qsvt_se/qsvt/engineering_utils.py",
        "Weighted-Jacobian engineering-system builder.",
        needed_by_task="B",
    ),
    ExpectedArtifact(
        "source_module",
        "src/robust_qsvt_se/experiments/tqe_revision_evidence.py",
        "Deterministic block selection + larger-matrix QSVT validation.",
        needed_by_task="B,D",
    ),
    # --- QSVT phase / resource / readout / boundary input outputs ----------
    ExpectedArtifact(
        "output_qsvt_phase",
        "outputs/qsvt_phase_validation_paper",
        "QSVT phase-synthesis validation outputs.",
    ),
    ExpectedArtifact(
        "output_qsvt_resource",
        "outputs/qsvt_oracle_model_resources/oracle_model_resource_summary.csv",
        "Oracle-model resource summary (degree, queries, qubits).",
        needed_by_task="C",
    ),
    ExpectedArtifact(
        "output_qsvt_resource",
        "outputs/hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv",
        "Hardware-aware total cost estimate (access/prep/amp/readout).",
        needed_by_task="C",
    ),
    ExpectedArtifact(
        "output_qsvt_resource",
        "outputs/full_qsvt_ieee_hardware_resources/qsvt_query_count_summary.csv",
        "QSVT query-count summary (2d+1 provenance).",
        needed_by_task="C",
    ),
    ExpectedArtifact(
        "output_qsvt_resource",
        "outputs/qsvt_resource_full_ieee/qsvt_resource_estimates.csv",
        "Full IEEE QSVT resource estimates.",
        needed_by_task="C",
    ),
    ExpectedArtifact(
        "output_qsvt_readout",
        "outputs/qsvt_gate_observable_readout",
        "Energy-style observable-first readout outputs.",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "output_qsvt_readout",
        "outputs/readout_limitation_formalization",
        "Readout-limitation formalization (no signed full-vector readout).",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "output_qsvt_boundary",
        "outputs/qsvt_multicase_approximation_diagnostics/multicase_approximation_summary.csv",
        "Multi-case polynomial approximation diagnostics.",
        needed_by_task="D",
    ),
    ExpectedArtifact(
        "output_qsvt_boundary",
        "outputs/tqe_qsvt_additional_experiments/degree_alpha_precision_sweep/"
        "degree_alpha_precision_sweep_results.csv",
        "Degree/alpha precision sweep (epsilon attainment).",
        needed_by_task="D",
    ),
    ExpectedArtifact(
        "output_qsvt_boundary",
        "outputs/tqe_qsvt_additional_experiments/full_gate_level_qsvt_coverage/"
        "full_gate_level_qsvt_coverage_results.csv",
        "Full gate-level QSVT coverage (phase/circuit status).",
        needed_by_task="D",
    ),
    # --- Manuscript-support outputs ----------------------------------------
    ExpectedArtifact(
        "output_manuscript_support",
        "outputs/final_manuscript_package",
        "Frozen manuscript assembly package.",
    ),
    ExpectedArtifact(
        "output_manuscript_support",
        "outputs/tqe_qsvt_additional_experiments/end_to_end_qsvt_vs_ridge",
        "End-to-end QSVT-vs-Ridge matched-alpha equivalence.",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "output_manuscript_support",
        "outputs/sparse_oracle_assumption_ledger",
        "Sparse-access oracle assumption ledger.",
        needed_by_task="E",
    ),
    # --- Tests that validate the cited evidence ----------------------------
    ExpectedArtifact(
        "test",
        "tests/test_tqe_revision_evidence.py",
        "Validates Ridge SVD update + block selection + QSVT matrix schema.",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "test",
        "tests/test_tqe_end_to_end_qsvt_vs_ridge.py",
        "Validates matched-alpha QSVT/Ridge equivalence.",
        needed_by_task="E",
    ),
    ExpectedArtifact(
        "test",
        "tests/test_filters.py",
        "Validates Ridge/Tikhonov spectral filter behaviour.",
        needed_by_task="E",
    ),
    # --- Outputs the revision tasks produce (absent before first build) ----
    ExpectedArtifact(
        "revision_artifact",
        "outputs/tqe_revision_support/signed_readout/signed_readout_summary.csv",
        "Task B: selected signed-observable readout diagnostic.",
        needed_by_task="B",
    ),
    ExpectedArtifact(
        "revision_artifact",
        "outputs/tqe_revision_support/cost_accounting/qsvt_cost_accounting.csv",
        "Task C: end-to-end cost accounting (included/excluded).",
        needed_by_task="C",
    ),
    ExpectedArtifact(
        "revision_artifact",
        "outputs/tqe_revision_support/qsvt_boundary/qsvt_boundary_diagnostics.csv",
        "Task D: QSVT boundary/failure diagnosis.",
        needed_by_task="D",
    ),
    ExpectedArtifact(
        "revision_artifact",
        "outputs/tqe_revision_support/claim_traceability/tqe_claim_traceability.csv",
        "Task E: claim-to-output traceability manifest.",
        needed_by_task="E",
    ),
)

# Directories inventoried (shallow) for a quick repository-state snapshot.
INVENTORY_DIRECTORIES = ("configs", "scripts", "tests", "outputs", "docs")


def run_evidence_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", AUDIT_OUTPUT_DIR)))
    repo_root = Path(resolved.get("repo_root", "."))

    manifest_rows = [_audit_row(repo_root, item) for item in EXPECTED_ARTIFACTS]
    manifest_frame = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    missing_frame = manifest_frame.loc[~manifest_frame["exists"], MISSING_COLUMNS].reset_index(
        drop=True
    )
    inventory = _directory_inventory(repo_root)

    manifest_csv = output_dir / "evidence_audit_manifest.csv"
    missing_csv = output_dir / "missing_expected_artifacts.csv"
    summary_md = output_dir / "evidence_audit_summary.md"
    manifest_frame.to_csv(manifest_csv, index=False)
    missing_frame.to_csv(missing_csv, index=False)
    summary_md.write_text(
        _summary_markdown(manifest_frame, missing_frame, inventory),
        encoding="utf-8",
    )

    artifacts = {
        "evidence_audit_manifest": manifest_csv,
        "missing_expected_artifacts": missing_csv,
        "evidence_audit_summary": summary_md,
    }
    manifest = write_manifest(
        output_dir=output_dir,
        artifact_name="tqe_revision_evidence_audit",
        description="Read-only existence audit of revision-relevant artifacts.",
        artifacts=artifacts,
        extra={
            "expected_artifact_count": len(manifest_frame),
            "present_count": int(manifest_frame["exists"].sum()),
            "missing_count": int((~manifest_frame["exists"]).sum()),
            "directory_inventory": inventory,
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "manifest_frame": manifest_frame,
        "missing_frame": missing_frame,
        "inventory": inventory,
        "artifacts": artifacts,
    }


def _audit_row(repo_root: Path, item: ExpectedArtifact) -> dict[str, Any]:
    path = repo_root / item.path
    exists = path.exists()
    is_dir = path.is_dir()
    child_count = _shallow_child_count(path) if is_dir else ""
    size_bytes = _safe_size(path) if exists and not is_dir else ""
    return {
        "category": item.category,
        "artifact_path": item.path,
        "artifact_type": artifact_type(path) if exists else "missing",
        "exists": bool(exists),
        "is_dir": bool(is_dir),
        "child_count": child_count,
        "size_bytes": size_bytes,
        "last_modified": last_modified(path) or "",
        "needed_by_task": item.needed_by_task,
        "optional": bool(item.optional),
        "description": item.description,
    }


def _directory_inventory(repo_root: Path) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for name in INVENTORY_DIRECTORIES:
        directory = repo_root / name
        inventory[name] = _shallow_child_count(directory) if directory.is_dir() else 0
    return inventory


def _shallow_child_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.iterdir())
    except OSError:
        return 0


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _summary_markdown(
    manifest: pd.DataFrame,
    missing: pd.DataFrame,
    inventory: dict[str, int],
) -> str:
    present = int(manifest["exists"].sum())
    total = len(manifest)
    lines = [
        "# TQE Revision Evidence Audit",
        "",
        "Read-only existence audit recorded **before/after** building the new revision "
        "artifacts. No solver behaviour, experiment result, or existing output is modified.",
        "",
        "## Repository Inventory (shallow child counts)",
        "",
        "| Directory | Entries |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}/` | {count} |" for name, count in inventory.items()]
    lines += [
        "",
        "## Expected-Artifact Coverage",
        "",
        f"- Expected artifacts tracked: **{total}**",
        f"- Present: **{present}**",
        f"- Missing: **{total - present}**",
        "",
        "### Present / missing by category",
        "",
        "| Category | Present | Missing |",
        "| --- | --- | --- |",
    ]
    for category, group in manifest.groupby("category"):
        present_n = int(group["exists"].sum())
        lines.append(f"| {category} | {present_n} | {len(group) - present_n} |")
    lines += ["", "## Missing Expected Artifacts", ""]
    if missing.empty:
        lines.append("- None. All tracked artifacts are present.")
    else:
        lines += ["| Path | Needed by task | Description |", "| --- | --- | --- |"]
        for _, row in missing.iterrows():
            task = row["needed_by_task"] or "-"
            lines.append(f"| `{row['artifact_path']}` | {task} | {row['description']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `revision_artifact` rows are produced by Tasks B-E; they are expected to be "
        "absent on the first audit and present after the build scripts run.",
        "- A missing non-revision artifact means a downstream task must regenerate or "
        "skip the corresponding evidence rather than invent values.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the TQE revision evidence audit")
    parser.add_argument("--output-dir", default=str(AUDIT_OUTPUT_DIR))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    run = run_evidence_audit({"output_dir": args.output_dir, "repo_root": args.repo_root})
    frame = run["manifest_frame"]
    print(
        f"Evidence audit complete: {int(frame['exists'].sum())}/{len(frame)} present -> "
        f"{run['artifacts']['evidence_audit_summary']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
