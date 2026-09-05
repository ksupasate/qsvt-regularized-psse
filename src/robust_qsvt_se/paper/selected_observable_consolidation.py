"""Task F: consolidate the selected-observable workload into paper-ready artifacts.

Runs the audit (A), sparse-access (B), selected-observable readout (C),
degree-aware alpha selection (D), and selected-observable cost accounting (E),
then writes the claim-boundary update, the consolidated paper-ready tables, and a
top-level manifest with checksums. Reuses existing artifacts where available;
never overwrites artifacts outside the workload directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper._common import read_csv
from robust_qsvt_se.paper.degree_aware_alpha import run_degree_aware_alpha_selection
from robust_qsvt_se.paper.selected_observable_audit import run_audit
from robust_qsvt_se.paper.selected_observable_common import (
    MANDATORY_BOUNDARY_STATEMENTS,
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
    write_workload_manifest,
)
from robust_qsvt_se.paper.selected_observable_cost import run_selected_observable_cost_accounting
from robust_qsvt_se.paper.selected_observable_workload import run_selected_observable_workload
from robust_qsvt_se.paper.sparse_access_workload import run_sparse_access_workload
from robust_qsvt_se.utils.io import ensure_directory

CONTRIBUTION_PARAGRAPH = (
    "The added implementation layer does not change the estimator. Ridge/Tikhonov remains "
    "the matched classical reference, and the QSVT-compatible target implements the same "
    "regularized spectral map at the same alpha. The new evidence makes the implementation "
    "pathway more concrete by specifying sparse-access lookups for generated weighted PSSE "
    "Jacobians, quantifying selected signed-observable readout, and reporting degree-aware "
    "alpha selection and selected-observable cost accounting. The results remain feasibility "
    "and boundary evidence: no speedup, no full-vector readout, and no run on quantum "
    "hardware."
)

# Disallowed/allowed wording carried through to the manuscript-facing claim update.
ALLOWED_CLAIMS = (
    "QSVT-compatible implementation pathway for the same Ridge/Tikhonov regularized filter",
    "controlled IEEE/PYPOWER benchmark with generated measurement rows",
    "validated classical sparse-access emulator (exact CSR lookup)",
    "selected signed-observable readout of the matched-alpha update",
    "degree-aware alpha selection under a QSVT degree budget",
    "selected-observable cost accounting with modeled/proxy/excluded terms",
)
DISALLOWED_CLAIMS = (
    "QSVT numerical superiority over Ridge/Tikhonov at the same alpha",
    "validation on real PMU/SCADA field measurements",
    "a demonstrated speedup or computational advantage",
    "full IEEE-scale runs on quantum hardware",
    "recovery of the full signed update vector (i.e., full-vector readout)",
    "a synthesized reversible sparse oracle circuit",
)


def run_consolidation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", WORKLOAD_DIR)))
    shared = {"output_dir": str(output_dir), "command": resolved.get("command", "consolidation")}

    sub_artifacts: dict[str, Path] = {}
    run_sparse_access_workload({**shared})
    run_selected_observable_workload({**shared, "trials": int(resolved.get("readout_trials", 200))})
    run_degree_aware_alpha_selection({**shared})
    run_selected_observable_cost_accounting(
        {**shared, "trials": int(resolved.get("cost_trials", 200))}
    )
    run_audit({"output_dir": str(output_dir)})

    claim_md = output_dir / "claim_boundary_update.md"
    tables_md = output_dir / "paper_ready_tables.md"
    claim_text = _claim_boundary_markdown()
    tables_text = _paper_ready_tables_markdown(output_dir)
    assert_safe(claim_text)
    assert_safe(tables_text)
    claim_md.write_text(claim_text, encoding="utf-8")
    tables_md.write_text(tables_text, encoding="utf-8")

    # Collect every workload artifact for the top-level manifest checksums.
    tracked = [
        "implementation_audit.md",
        "manuscript_integration_audit.md",
        "repo_integration_plan.json",
        "sparse_access_summary.csv",
        "sparse_access_summary.json",
        "sparse_access_validation.csv",
        "sparse_access_report.md",
        "selected_observables.csv",
        "readout_shot_sweep.csv",
        "readout_map.csv",
        "readout_summary.md",
        "observable_selection_policy.md",
        "degree_aware_alpha_grid.csv",
        "degree_aware_alpha_summary.csv",
        "degree_aware_alpha_report.md",
        "revised_degree_alpha_summary.csv",
        "revised_degree_alpha_report.md",
        "selected_observable_cost.csv",
        "selected_observable_cost_summary.md",
        "revised_cost_composition.csv",
        "revised_cost_composition.md",
        "integrated_small_qsvt_readout_demo.csv",
        "claim_boundary_update.md",
        "paper_ready_tables.md",
    ]
    artifacts = {name: output_dir / name for name in tracked if (output_dir / name).is_file()}
    for name, path in {
        "sparse_oracle_block_encoding_spec": Path("docs/SPARSE_ORACLE_BLOCK_ENCODING_SPEC.md"),
        "selected_signed_readout_doc": Path("docs/SELECTED_SIGNED_READOUT.md"),
        "sparse_access_model_doc": Path("docs/SPARSE_ACCESS_MODEL.md"),
        "degree_aware_alpha_doc": Path("docs/DEGREE_AWARE_ALPHA_SELECTION.md"),
        "manuscript_source": Path("manuscript/main.tex"),
        "manuscript_pdf": Path("manuscript/main.pdf"),
    }.items():
        if path.is_file():
            artifacts[name] = path
    sub_artifacts.update(artifacts)

    manifest = write_workload_manifest(
        output_dir=output_dir,
        artifact_name="selected_observable_workload_consolidation",
        description=(
            "Consolidated QSVT selected-observable workload: sparse access, selected "
            "signed-observable readout, degree-aware alpha selection, and selected-observable "
            "cost accounting. Feasibility and boundary evidence only."
        ),
        command=resolved.get("command", "scripts/run_selected_observable_workload.py"),
        artifacts=artifacts,
        input_files=[
            "build_engineering_system (pypower AC-linearized weighted Jacobians)",
            "outputs/full_alpha_sensitivity_classical/alpha_sweep_summary_by_case.csv",
            "outputs/hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv",
        ],
        reran_long_experiments=False,
        aggregated_from_existing=True,
        extra={
            "tasks": ["A_audit", "B_sparse_access", "C_readout", "D_degree_alpha", "E_cost"],
            "allowed_claims": list(ALLOWED_CLAIMS),
            "disallowed_claims": list(DISALLOWED_CLAIMS),
        },
        manifest_name="manifest.json",
    )
    sub_artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "artifacts": sub_artifacts}


def _claim_boundary_markdown() -> str:
    lines = [
        "# Claim-Boundary Update (Selected-Observable Workload)",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "## What this workload supports (allowed)",
        "",
        *[f"- {claim}" for claim in ALLOWED_CLAIMS],
        "",
        "## What this workload does not claim (disallowed)",
        "",
        *[f"- Not claimed: {claim}" for claim in DISALLOWED_CLAIMS],
        "",
        "## Mandatory boundary statements",
        "",
        *[f"- {statement}" for statement in MANDATORY_BOUNDARY_STATEMENTS],
        "",
        "## Estimator invariance",
        "",
        "- The estimator is unchanged. The QSVT-compatible target implements the same "
        "Ridge/Tikhonov regularized spectral filter `sigma/(sigma^2+alpha)` at the same "
        "alpha; no advantage over Ridge/Tikhonov is claimed.",
        "",
    ]
    return "\n".join(lines)


def _compact(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    present = [c for c in columns if c in frame.columns]
    out = frame[present] if present else frame
    return out.head(limit) if limit else out


def _frame_to_md(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_(table unavailable)_", ""]
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    lines.append("")
    return lines


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if value == 0.0:
            return "0"
        if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _paper_ready_tables_markdown(output_dir: Path) -> str:
    sparse = read_csv(output_dir / "sparse_access_summary.csv")
    observables = read_csv(output_dir / "selected_observables.csv")
    alpha_summary = read_csv(output_dir / "revised_degree_alpha_summary.csv")
    cost = read_csv(output_dir / "revised_cost_composition.csv")

    lines = [
        "# Paper-Ready Tables: QSVT Selected-Observable Workload",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "## Table 1: Sparse-Access Summary (generated weighted PSSE Jacobians)",
        "",
        *_frame_to_md(
            _compact(
                sparse,
                [
                    "case",
                    "shape_rows",
                    "shape_cols",
                    "nnz",
                    "density",
                    "max_row_nnz",
                    "index_qubits",
                    "value_register_qubits",
                    "access_status",
                    "reversible_oracle_synthesized",
                ],
            )
        ),
        "## Table 2: Selected-Observable Readout (exact matched values + classification)",
        "",
        *_frame_to_md(
            _compact(
                observables,
                [
                    "case",
                    "observable_id",
                    "selection_policy",
                    "selected_before_solving",
                    "depends_on_solution",
                    "support_size",
                    "exact_value",
                    "readout_model",
                    "status",
                ],
            )
        ),
        "## Table 3: Degree-Aware Alpha Selection (rule comparison)",
        "",
        *_frame_to_md(
            _compact(
                alpha_summary,
                [
                    "case",
                    "selection_rule",
                    "selected_alpha",
                    "rmse_at_selected",
                    "spectrum_point_degree",
                    "uniform_grid_degree",
                    "phase_synthesis_available",
                    "qsvt_query_count_realizable",
                ],
            )
        ),
        "## Table 4: Selected-Observable Repetition-Cost Composition",
        "",
        *_frame_to_md(
            _compact(
                cost,
                [
                    "case",
                    "observable_id",
                    "degree",
                    "shots",
                    "success_probability_proxy",
                    "unitary_queries_per_attempt",
                    "expected_attempts_no_aa",
                    "expected_unitary_queries_no_aa",
                    "readout_target_met",
                    "unitary_query_count_status",
                    "full_vector_recovery_included",
                ],
            )
        ),
        "## Contribution (conservative summary)",
        "",
        CONTRIBUTION_PARAGRAPH,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Consolidate the selected-observable workload")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    parser.add_argument("--readout-trials", type=int, default=200)
    parser.add_argument("--cost-trials", type=int, default=200)
    args = parser.parse_args(argv)
    run = run_consolidation(
        {
            "output_dir": args.output_dir,
            "readout_trials": args.readout_trials,
            "cost_trials": args.cost_trials,
            "command": "scripts/run_selected_observable_workload.py " + " ".join(argv or []),
        }
    )
    print(f"Consolidation complete: {run['artifacts']['manifest']}")


if __name__ == "__main__":  # pragma: no cover
    main()
