from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se import __version__
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_ROOT = Path("outputs/tqe_qsvt_additional_experiments")
DEGREE_SWEEP_DIR = "degree_alpha_precision_sweep"
BLOCK_ENCODING_DIR = "explicit_block_encoding_demo"
END_TO_END_DIR = "end_to_end_qsvt_vs_ridge"
CIRCUIT_BLOCK_ENCODING_DIR = "circuit_level_block_encoding"
INTEGRATED_QSVT_CIRCUIT_DIR = "integrated_small_qsvt_circuit"
OBSERVABLE_READOUT_DIR = "observable_first_readout"
SPARSE_ORACLE_DIR = "sparse_oracle_block_encoding_model"
NONLINEAR_FEASIBILITY_DIR = "nonlinear_ac_per_iteration_feasibility"
FINAL_ROBUSTNESS_AUDITS_DIR = "final_robustness_audits"
FULL_GATE_LEVEL_COVERAGE_DIR = "full_gate_level_qsvt_coverage"
PHASE_SYNTHESIS_8X8_RESCUE_DIR = "phase_synthesis_8x8_rescue"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
REPORTS_DIR = "reports"

CLAIM_BOUNDARY = (
    "QSVT-compatible selected-subproblem engineering evidence only; no quantum "
    "speedup, no QSVT-over-Ridge/Tikhonov numerical superiority, no full IEEE-scale "
    "quantum state-estimation claim, and no scalable dense-block-encoding claim."
)


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_command() -> str:
    return " ".join(sys.argv)


def ensure_tqe_output_tree(output_root: str | Path = OUTPUT_ROOT) -> dict[str, Path]:
    root = ensure_directory(output_root)
    paths = {
        "root": root,
        "degree": ensure_directory(root / DEGREE_SWEEP_DIR),
        "block": ensure_directory(root / BLOCK_ENCODING_DIR),
        "end_to_end": ensure_directory(root / END_TO_END_DIR),
        "circuit_block": ensure_directory(root / CIRCUIT_BLOCK_ENCODING_DIR),
        "integrated_qsvt": ensure_directory(root / INTEGRATED_QSVT_CIRCUIT_DIR),
        "observable_readout": ensure_directory(root / OBSERVABLE_READOUT_DIR),
        "sparse_oracle": ensure_directory(root / SPARSE_ORACLE_DIR),
        "nonlinear_feasibility": ensure_directory(root / NONLINEAR_FEASIBILITY_DIR),
        "final_robustness": ensure_directory(root / FINAL_ROBUSTNESS_AUDITS_DIR),
        "full_gate_level_coverage": ensure_directory(root / FULL_GATE_LEVEL_COVERAGE_DIR),
        "figures": ensure_directory(root / FIGURES_DIR),
        "tables": ensure_directory(root / TABLES_DIR),
        "reports": ensure_directory(root / REPORTS_DIR),
    }
    return paths


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


def git_status_short() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def package_versions(package_names: list[str] | None = None) -> dict[str, str | None]:
    names = package_names or [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "qiskit",
        "pennylane",
        "pyqsp",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def reproducibility_metadata(
    *,
    config: dict[str, Any],
    started_at: str,
    ended_at: str,
    status: str,
    command: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _json_safe(
        {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "command": command or current_command(),
            "config": config,
            "random_seeds": _collect_seeds(config),
            "git_commit": git_commit(),
            "git_status_short": git_status_short(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "package_version": __version__,
            "dependency_versions": package_versions(),
            "artifacts": artifacts or {},
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )


def write_top_level_manifest_and_report(
    output_root: str | Path = OUTPUT_ROOT,
    *,
    extra_artifacts: dict[str, str] | None = None,
) -> dict[str, Path]:
    paths = ensure_tqe_output_tree(output_root)
    root = paths["root"]
    artifacts = _discover_artifacts(root)
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    manifest = {
        "generated_at": utc_timestamp(),
        "command": current_command(),
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
        "dependency_versions": package_versions(),
        "artifacts": artifacts,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    report_path = paths["reports"] / "final_additional_experiments_report.md"
    report_path.write_text(_final_report_markdown(root), encoding="utf-8")
    return {"manifest": manifest_path, "final_report": report_path}


def _discover_artifacts(root: Path) -> dict[str, str]:
    expected = {
        "degree_results_csv": root / DEGREE_SWEEP_DIR / "degree_alpha_precision_sweep_results.csv",
        "degree_metadata_json": root
        / DEGREE_SWEEP_DIR
        / "degree_alpha_precision_sweep_metadata.json",
        "degree_summary_table": root / TABLES_DIR / "table_degree_alpha_precision_summary.csv",
        "degree_figure": root / FIGURES_DIR / "figure_degree_alpha_precision_tradeoff.png",
        "degree_report": root / REPORTS_DIR / "degree_alpha_precision_sweep_report.md",
        "block_results_csv": root / BLOCK_ENCODING_DIR / "block_encoding_demo_results.csv",
        "block_metadata_json": root / BLOCK_ENCODING_DIR / "block_encoding_demo_metadata.json",
        "block_summary_table": root / TABLES_DIR / "table_block_encoding_resource_summary.csv",
        "block_figure": root / FIGURES_DIR / "figure_block_encoding_errors.png",
        "block_report": root / REPORTS_DIR / "explicit_block_encoding_demo_report.md",
        "end_to_end_results_csv": root / END_TO_END_DIR / "end_to_end_qsvt_vs_ridge_results.csv",
        "end_to_end_metadata_json": root
        / END_TO_END_DIR
        / "end_to_end_qsvt_vs_ridge_metadata.json",
        "end_to_end_summary_table": root
        / TABLES_DIR
        / "table_end_to_end_qsvt_vs_ridge_summary.csv",
        "end_to_end_relative_error_figure": root
        / FIGURES_DIR
        / "figure_end_to_end_relative_update_error.png",
        "end_to_end_residual_ratio_figure": root
        / FIGURES_DIR
        / "figure_end_to_end_residual_ratio_comparison.png",
        "end_to_end_update_scatter_figure": root
        / FIGURES_DIR
        / "figure_qsvt_vs_ridge_update_scatter.png",
        "end_to_end_report": root / REPORTS_DIR / "end_to_end_qsvt_vs_ridge_report.md",
        "circuit_block_results_csv": root
        / CIRCUIT_BLOCK_ENCODING_DIR
        / "circuit_level_block_encoding_results.csv",
        "circuit_block_statevector_details_csv": root
        / CIRCUIT_BLOCK_ENCODING_DIR
        / "statevector_action_verification_details.csv",
        "circuit_block_metadata_json": root
        / CIRCUIT_BLOCK_ENCODING_DIR
        / "circuit_level_block_encoding_metadata.json",
        "circuit_block_summary_table": root
        / TABLES_DIR
        / "table_circuit_level_block_encoding_summary.csv",
        "circuit_block_action_errors_figure": root
        / FIGURES_DIR
        / "figure_circuit_block_action_errors.png",
        "circuit_block_depth_figure": root
        / FIGURES_DIR
        / "figure_circuit_block_encoding_depth.png",
        "circuit_block_cx_counts_figure": root
        / FIGURES_DIR
        / "figure_circuit_block_encoding_cx_counts.png",
        "circuit_block_report": root / REPORTS_DIR / "circuit_level_block_encoding_report.md",
        "integrated_qsvt_results_csv": root
        / INTEGRATED_QSVT_CIRCUIT_DIR
        / "integrated_small_qsvt_circuit_results.csv",
        "integrated_qsvt_sanity_csv": root
        / INTEGRATED_QSVT_CIRCUIT_DIR
        / "sanity_check_results.csv",
        "integrated_qsvt_statevector_details_csv": root
        / INTEGRATED_QSVT_CIRCUIT_DIR
        / "statevector_probe_details.csv",
        "integrated_qsvt_metadata_json": root
        / INTEGRATED_QSVT_CIRCUIT_DIR
        / "integrated_small_qsvt_circuit_metadata.json",
        "integrated_qsvt_summary_table": root
        / TABLES_DIR
        / "table_integrated_small_qsvt_circuit_summary.csv",
        "integrated_qsvt_update_error_figure": root
        / FIGURES_DIR
        / "figure_integrated_qsvt_circuit_update_error.png",
        "integrated_qsvt_transform_error_figure": root
        / FIGURES_DIR
        / "figure_integrated_qsvt_circuit_transform_error.png",
        "integrated_qsvt_resource_counts_figure": root
        / FIGURES_DIR
        / "figure_integrated_qsvt_circuit_resource_counts.png",
        "integrated_qsvt_report": root / REPORTS_DIR / "integrated_small_qsvt_circuit_report.md",
        "observable_readout_results_csv": root
        / OBSERVABLE_READOUT_DIR
        / "observable_first_readout_results.csv",
        "observable_readout_counts_json": root
        / OBSERVABLE_READOUT_DIR
        / "observable_first_readout_counts.json",
        "observable_readout_metadata_json": root
        / OBSERVABLE_READOUT_DIR
        / "observable_first_readout_metadata.json",
        "observable_readout_summary_table": root
        / TABLES_DIR
        / "table_observable_first_readout_summary.csv",
        "observable_readout_error_figure": root
        / FIGURES_DIR
        / "figure_observable_readout_error_vs_shots.png",
        "observable_readout_ci_figure": root
        / FIGURES_DIR
        / "figure_observable_readout_ci_width.png",
        "observable_readout_ridge_qsvt_figure": root
        / FIGURES_DIR
        / "figure_observable_readout_ridge_vs_qsvt.png",
        "observable_readout_report": root / REPORTS_DIR / "observable_first_readout_report.md",
        "sparse_oracle_sparsity_csv": root / SPARSE_ORACLE_DIR / "sparsity_audit_results.csv",
        "sparse_oracle_verification_csv": root
        / SPARSE_ORACLE_DIR
        / "oracle_reconstruction_verification.csv",
        "sparse_oracle_resource_csv": root
        / SPARSE_ORACLE_DIR
        / "sparse_oracle_resource_estimates.csv",
        "sparse_oracle_samples_json": root / SPARSE_ORACLE_DIR / "oracle_samples.json",
        "sparse_oracle_metadata_json": root
        / SPARSE_ORACLE_DIR
        / "sparse_oracle_block_encoding_model_metadata.json",
        "sparse_oracle_summary_table": root
        / TABLES_DIR
        / "table_sparse_oracle_block_encoding_summary.csv",
        "sparse_oracle_density_figure": root / FIGURES_DIR / "figure_sparse_jacobian_density.png",
        "sparse_oracle_row_nnz_figure": root
        / FIGURES_DIR
        / "figure_sparse_row_nnz_distribution.png",
        "sparse_oracle_normalization_figure": root
        / FIGURES_DIR
        / "figure_sparse_oracle_normalization_overhead.png",
        "sparse_oracle_qubit_figure": root
        / FIGURES_DIR
        / "figure_sparse_oracle_qubit_estimates.png",
        "sparse_oracle_report": root / REPORTS_DIR / "sparse_oracle_block_encoding_model_report.md",
        "nonlinear_feasibility_diagnostics_csv": root
        / NONLINEAR_FEASIBILITY_DIR
        / "nonlinear_iteration_diagnostics.csv",
        "nonlinear_feasibility_run_summary_csv": root
        / NONLINEAR_FEASIBILITY_DIR
        / "nonlinear_run_summary.csv",
        "nonlinear_feasibility_metadata_json": root
        / NONLINEAR_FEASIBILITY_DIR
        / "nonlinear_ac_per_iteration_feasibility_metadata.json",
        "nonlinear_feasibility_summary_table": root
        / TABLES_DIR
        / "table_nonlinear_ac_qsvt_feasibility_summary.csv",
        "nonlinear_feasibility_condition_figure": root
        / FIGURES_DIR
        / "figure_nonlinear_condition_number_by_iteration.png",
        "nonlinear_feasibility_degree_figure": root
        / FIGURES_DIR
        / "figure_nonlinear_required_degree_by_iteration.png",
        "nonlinear_feasibility_residual_figure": root
        / FIGURES_DIR
        / "figure_nonlinear_rmse_residual_by_iteration.png",
        "nonlinear_feasibility_sparse_overhead_figure": root
        / FIGURES_DIR
        / "figure_nonlinear_sparse_oracle_overhead_by_iteration.png",
        "nonlinear_feasibility_report": root
        / REPORTS_DIR
        / "nonlinear_ac_per_iteration_feasibility_report.md",
        "final_robustness_phase_audit_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "phase_synthesis_hard_case_audit.csv",
        "final_robustness_signed_readout_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "signed_phase_aware_readout_results.csv",
        "final_robustness_noise_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "noise_sensitivity_integrated_qsvt.csv",
        "final_robustness_pq_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "reactive_pq_row_composition_ablation.csv",
        "final_robustness_tiny_oracle_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "tiny_reversible_sparse_oracle_lookup.csv",
        "final_robustness_repeat_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "integrated_qsvt_repeat_case.csv",
        "final_robustness_alpha_csv": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "alpha_selection_diagnostic.csv",
        "final_robustness_manifest": root
        / FINAL_ROBUSTNESS_AUDITS_DIR
        / "final_robustness_audit_manifest.json",
        "final_robustness_summary_table": root
        / TABLES_DIR
        / "table_final_robustness_audit_summary.csv",
        "final_robustness_report": root / REPORTS_DIR / "final_robustness_audits_report.md",
        "full_gate_level_coverage_results_csv": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / "full_gate_level_qsvt_coverage_results.csv",
        "full_gate_level_coverage_metadata_json": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / "full_gate_level_qsvt_coverage_metadata.json",
        "full_gate_level_coverage_summary_table": root
        / TABLES_DIR
        / "table_full_gate_level_qsvt_coverage_summary.csv",
        "full_gate_level_coverage_errors_figure": root
        / FIGURES_DIR
        / "figure_full_gate_level_qsvt_errors.png",
        "full_gate_level_coverage_depth_cx_figure": root
        / FIGURES_DIR
        / "figure_full_gate_level_qsvt_depth_cx.png",
        "full_gate_level_coverage_success_probability_figure": root
        / FIGURES_DIR
        / "figure_full_gate_level_qsvt_success_probability.png",
        "full_gate_level_coverage_report": root
        / REPORTS_DIR
        / "full_gate_level_qsvt_coverage_report.md",
        "full_gate_level_degree_remediation_csv": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / "full_gate_level_qsvt_degree_remediation.csv",
        "full_gate_level_degree_remediation_report": root
        / REPORTS_DIR
        / "full_gate_level_qsvt_degree_remediation_report.md",
        "phase_synthesis_8x8_rescue_attempts_csv": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / PHASE_SYNTHESIS_8X8_RESCUE_DIR
        / "phase_synthesis_8x8_rescue_attempts.csv",
        "phase_synthesis_8x8_rescue_summary_csv": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / PHASE_SYNTHESIS_8X8_RESCUE_DIR
        / "phase_synthesis_8x8_rescue_summary.csv",
        "phase_synthesis_8x8_rescue_metadata_json": root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / PHASE_SYNTHESIS_8X8_RESCUE_DIR
        / "phase_synthesis_8x8_rescue_metadata.json",
        "phase_synthesis_8x8_rescue_error_figure": root
        / FIGURES_DIR
        / "figure_8x8_rescue_error_vs_degree.png",
        "phase_synthesis_8x8_rescue_admissibility_figure": root
        / FIGURES_DIR
        / "figure_8x8_rescue_admissibility_margin.png",
        "phase_synthesis_8x8_rescue_success_probability_figure": root
        / FIGURES_DIR
        / "figure_8x8_rescue_success_probability.png",
        "phase_synthesis_8x8_rescue_report": root
        / REPORTS_DIR
        / "phase_synthesis_8x8_rescue_report.md",
    }
    return {key: str(path) for key, path in expected.items() if path.exists()}


def _final_report_markdown(root: Path) -> str:
    degree_summary = root / TABLES_DIR / "table_degree_alpha_precision_summary.csv"
    block_summary = root / TABLES_DIR / "table_block_encoding_resource_summary.csv"
    end_to_end_summary = root / TABLES_DIR / "table_end_to_end_qsvt_vs_ridge_summary.csv"
    circuit_summary = root / TABLES_DIR / "table_circuit_level_block_encoding_summary.csv"
    integrated_summary = root / TABLES_DIR / "table_integrated_small_qsvt_circuit_summary.csv"
    observable_summary = root / TABLES_DIR / "table_observable_first_readout_summary.csv"
    sparse_summary = root / TABLES_DIR / "table_sparse_oracle_block_encoding_summary.csv"
    nonlinear_summary = root / TABLES_DIR / "table_nonlinear_ac_qsvt_feasibility_summary.csv"
    final_robustness_summary = root / TABLES_DIR / "table_final_robustness_audit_summary.csv"
    full_gate_summary = root / TABLES_DIR / "table_full_gate_level_qsvt_coverage_summary.csv"
    full_gate_remediation = (
        root / FULL_GATE_LEVEL_COVERAGE_DIR / "full_gate_level_qsvt_degree_remediation.csv"
    )
    phase_rescue_summary = (
        root
        / FULL_GATE_LEVEL_COVERAGE_DIR
        / PHASE_SYNTHESIS_8X8_RESCUE_DIR
        / "phase_synthesis_8x8_rescue_summary.csv"
    )
    degree_lines = _degree_findings(degree_summary)
    block_lines = _block_findings(block_summary)
    end_to_end_lines = _end_to_end_findings(end_to_end_summary)
    circuit_lines = _circuit_block_findings(circuit_summary)
    integrated_lines = _integrated_qsvt_findings(integrated_summary)
    observable_lines = _observable_readout_findings(observable_summary)
    sparse_lines = _sparse_oracle_findings(sparse_summary)
    nonlinear_lines = _nonlinear_feasibility_findings(nonlinear_summary)
    final_robustness_lines = _final_robustness_findings(final_robustness_summary)
    full_gate_lines = _full_gate_level_coverage_findings(full_gate_summary)
    full_gate_remediation_lines = _full_gate_level_remediation_findings(full_gate_remediation)
    phase_rescue_lines = _phase_synthesis_8x8_rescue_findings(phase_rescue_summary)
    artifacts = _discover_artifacts(root)
    artifact_lines = [f"- `{path}`" for path in artifacts.values()]
    if not artifact_lines:
        artifact_lines = ["- No experiment artifacts have been generated yet."]
    return "\n".join(
        [
            "# Final Additional TQE Experiments Report",
            "",
            "## 1. What was implemented",
            "",
            "- Degree-alpha-precision sweep for the bounded QSVT-compatible Ridge/Tikhonov target.",
            "- Explicit dense block-encoding demo for selected IEEE-derived "
            "weighted-Jacobian blocks.",
            "- End-to-end QSVT-compatible polynomial update consistency check against "
            "matched Ridge/Tikhonov.",
            "- Circuit-level dense block-encoding verification for selected "
            "IEEE-derived weighted-Jacobian blocks.",
            "- Integrated small QSVT circuit sequence for a selected IEEE-derived "
            "weighted-Jacobian block.",
            "- Observable-first readout with computational-basis shot simulation for "
            "selected update observables.",
            "- Sparse-oracle access-model audit for generated IEEE/PYPOWER weighted Jacobians.",
            "- Nonlinear AC per-iteration QSVT-compatible feasibility diagnostic.",
            "- Final robustness audits covering phase synthesis, signed readout, noise "
            "sensitivity, P/Q row composition, tiny sparse-oracle lookup, repeat-case "
            "integrated QSVT, and alpha selection.",
            "- Fuller gate-level QSVT coverage audit across selected small "
            "IEEE-derived subproblems under a fixed dense-circuit budget.",
            "- Targeted 8x8 phase-synthesis rescue for unresolved integrated gate-level QSVT rows.",
            "- Reproducibility metadata, manifests, summary tables, figures, and "
            "claim-safe reports.",
            "",
            "## 2. Commands run",
            "",
            "- `.venv/bin/python scripts/run_tqe_qsvt_degree_alpha_precision_sweep.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_explicit_block_encoding_demo.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_end_to_end_qsvt_vs_ridge.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_circuit_level_block_encoding.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_integrated_small_qsvt_circuit.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_observable_first_readout.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_sparse_oracle_block_encoding_model.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_nonlinear_ac_per_iteration_feasibility.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_final_robustness_audits.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_full_gate_level_qsvt_coverage.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_full_gate_level_degree_remediation.py`",
            "- `.venv/bin/python scripts/run_tqe_qsvt_phase_synthesis_8x8_rescue.py`",
            "",
            "## 3. Files created",
            "",
            *artifact_lines,
            "",
            "## 4. Cases succeeded, failed, or skipped",
            "",
            *degree_lines["status"],
            *block_lines["status"],
            *end_to_end_lines["status"],
            *circuit_lines["status"],
            *integrated_lines["status"],
            *observable_lines["status"],
            *sparse_lines["status"],
            *nonlinear_lines["status"],
            *final_robustness_lines["status"],
            *full_gate_lines["status"],
            *full_gate_remediation_lines["status"],
            *phase_rescue_lines["status"],
            "",
            "## 5. Key numerical results",
            "",
            *degree_lines["findings"],
            *block_lines["findings"],
            *end_to_end_lines["findings"],
            *circuit_lines["findings"],
            *integrated_lines["findings"],
            *observable_lines["findings"],
            *sparse_lines["findings"],
            *nonlinear_lines["findings"],
            *final_robustness_lines["findings"],
            *full_gate_lines["findings"],
            *full_gate_remediation_lines["findings"],
            *phase_rescue_lines["findings"],
            "",
            "## 6. Manuscript figure/table support",
            "",
            "- The degree sweep supports a TQE-facing degree-cost figure/table for "
            "bounded Ridge/Tikhonov QSVT targets.",
            "- The block-encoding demo supports a resource-boundary table for selected "
            "dense subproblem embeddings.",
            "- The end-to-end solver check supports the implementation-pathway "
            "consistency claim for selected weighted-Jacobian subproblems.",
            "- The circuit-level block-encoding verification supports selected-subproblem "
            "circuit-object evidence for the dense block-encoding pathway.",
            "- The integrated small QSVT circuit sequence supports circuit-level "
            "transform consistency evidence for a selected small block.",
            "- The observable-first readout diagnostic supports selected-observable "
            "sampling evidence while preserving full-vector readout limitations.",
            "- The sparse-oracle model supports a claim-safe route from dense "
            "selected-subproblem evidence toward full-matrix sparse-access assumptions.",
            "- The nonlinear per-iteration diagnostic supports a future-work boundary "
            "for QSVT-compatible updates inside nonlinear AC workflows.",
            "- The final robustness audits support reviewer-facing confidence checks "
            "and should mostly be placed in the supplement or reproducibility package.",
            "- The full gate-level coverage audit supports a supplement-level "
            "account of which selected small dense circuits can be validated under "
            "the configured budget.",
            "- The full gate-level degree-remediation audit documents whether "
            "flagged update-level mismatches clear under higher candidate degrees.",
            "- The 8x8 phase-synthesis rescue documents whether unresolved 8x8 "
            "rows can be recovered by admissible target contraction and phase "
            "synthesis retries.",
            "",
            "## 7. Recommended manuscript wording",
            "",
            "These results quantify the degree cost of implementing the bounded "
            "Ridge/Tikhonov spectral filter as a QSVT-compatible target. They do not "
            "imply QSVT numerically outperforms Ridge/Tikhonov under the matched "
            "spectral map.",
            "",
            "The explicit dense block encoding verifies that selected IEEE-derived "
            "weighted-Jacobian blocks can be embedded into a unitary model suitable "
            "for QSVT-style validation. This construction is not claimed to be a "
            "scalable oracle for full IEEE-scale PSSE.",
            "",
            "The end-to-end small-solver experiment verifies consistency between the "
            "matched Ridge/Tikhonov update and the QSVT-compatible polynomial "
            "implementation on selected IEEE-derived weighted-Jacobian subproblems.",
            "",
            "The circuit-level dense block-encoding experiment verifies that selected "
            "IEEE-derived weighted-Jacobian blocks can be represented as explicit "
            "circuit objects whose simulated block action reproduces the normalized "
            "matrix within numerical tolerance. These dense circuits are proof-of-concept "
            "selected-subproblem constructions, not scalable sparse-oracle block "
            "encodings for full IEEE-scale PSSE.",
            "",
            "The integrated small QSVT circuit sequence verifies, for a selected small "
            "IEEE-derived weighted-Jacobian block, that a synthesized low-degree "
            "QSVT sequence reproduces the polynomial transform and the matched "
            "Ridge/Tikhonov update within controlled statevector simulation error. "
            "This is selected-subproblem circuit-level evidence only.",
            "",
            "The observable-first readout experiment estimates selected energy-style "
            "update observables from the QSVT-compatible output-state distribution "
            "using shot-based simulation. Signed coordinate differences are reported "
            "as requiring phase/sign-aware readout and are not treated as directly "
            "accessible from ordinary computational-basis sampling.",
            "",
            "The sparse-oracle model verifies a classical sparse index/value oracle "
            "emulator for generated IEEE/PYPOWER weighted Jacobians. The resulting "
            "register-size, padding, and normalization diagnostics are oracle-level "
            "resource estimates, not compiled reversible oracle gate counts.",
            "",
            "The nonlinear AC per-iteration diagnostic records the conditioning, "
            "sparse-access normalization, and bounded-polynomial degree requirements "
            "that a QSVT-compatible regularized update would face inside a classical "
            "nonlinear AC workflow. QSVT is not executed inside the nonlinear loop.",
            "",
            "The final robustness audits document phase-synthesis feasibility "
            "boundaries, phase-aware readout requirements, simple noise sensitivity, "
            "measurement-row composition effects, a tiny reversible sparse-oracle "
            "lookup prototype, repeatability on one additional selected subproblem, "
            "and alpha/degree tradeoffs. These remain audits and diagnostics.",
            "",
            "The fuller gate-level QSVT coverage audit expands selected-subproblem "
            "circuit-level validation under a fixed dense-circuit budget. Successful "
            "rows provide consistency evidence; skipped rows identify feasibility "
            "boundaries and do not demonstrate full IEEE-scale QSVT execution.",
            "",
            "The degree-remediation audit checks flagged update-level mismatches "
            "with higher-degree QSVT polynomials. Successful remediation indicates "
            "degree adequacy rather than circuit-convention failure; unresolved "
            "rows are reported as dense-circuit feasibility boundaries.",
            "",
            "The 8x8 phase-synthesis rescue contracts non-admissible but accurate "
            "actual-singular-value polynomials into a QSVT-admissible amplitude "
            "range, then rescales back to the matched Ridge/Tikhonov physical "
            "filter. Successful rescue indicates a target-admissibility issue "
            "rather than a circuit-convention failure.",
            "",
            "## Experiment 3A: Circuit-Level Dense Block-Encoding Verification",
            "",
            *circuit_lines["status"],
            *circuit_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "circuit_level_block_encoding_report.md`.",
            "- Interpretation: selected-subproblem circuit-level verification of dense "
            "block-encoding objects; not a scalable sparse-oracle construction.",
            "",
            "## Experiment 3B: Integrated Small QSVT Circuit Sequence",
            "",
            *integrated_lines["status"],
            *integrated_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "integrated_small_qsvt_circuit_report.md`.",
            "- Interpretation: selected-subproblem circuit-level QSVT sequence "
            "consistency check; not hardware execution and not a scalable solver.",
            "",
            "## Experiment 4: Observable-First Readout with Shot Simulation",
            "",
            *observable_lines["status"],
            *observable_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "observable_first_readout_report.md`.",
            "- Interpretation: selected-observable sampling diagnostic; not full-vector "
            "recovery and not hardware execution.",
            "",
            "## Experiment 5: Sparse-Oracle Scalable Block-Encoding Model",
            "",
            *sparse_lines["status"],
            *sparse_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "sparse_oracle_block_encoding_model_report.md`.",
            "- Interpretation: sparse-access oracle emulator and resource estimate; "
            "not a full reversible oracle circuit and not hardware execution.",
            "",
            "## Experiment 6: Nonlinear AC Per-Iteration QSVT Feasibility Diagnostic",
            "",
            *nonlinear_lines["status"],
            *nonlinear_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "nonlinear_ac_per_iteration_feasibility_report.md`.",
            "- Interpretation: classical nonlinear AC loop with per-iteration "
            "QSVT-compatible degree/resource diagnostics; not a nonlinear QSVT "
            "solver and not hardware execution.",
            "",
            "## Final Robustness Audits",
            "",
            *final_robustness_lines["status"],
            *final_robustness_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "final_robustness_audits_report.md`.",
            "- Interpretation: reviewer-facing audits and diagnostics only; no "
            "claim of scalable QSVT PSSE, hardware execution, full-vector readout, "
            "or QSVT-over-Ridge advantage.",
            "",
            "## Fuller Gate-Level QSVT Coverage Audit",
            "",
            *full_gate_lines["status"],
            *full_gate_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "full_gate_level_qsvt_coverage_report.md`.",
            "- Interpretation: expanded selected-subproblem gate-level audit under "
            "budget; dense proof-of-concept circuits only.",
            "",
            "## Full Gate-Level Degree Remediation Audit",
            "",
            *full_gate_remediation_lines["status"],
            *full_gate_remediation_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "full_gate_level_qsvt_degree_remediation_report.md`.",
            "- Interpretation: targeted higher-degree audit of flagged update-level "
            "mismatches; not a full coverage rerun.",
            "",
            "## 8x8 Phase-Synthesis Rescue",
            "",
            *phase_rescue_lines["status"],
            *phase_rescue_lines["findings"],
            "- Report: "
            "`outputs/tqe_qsvt_additional_experiments/reports/"
            "phase_synthesis_8x8_rescue_report.md`.",
            "- Interpretation: targeted rescue of unresolved 8x8 dense-circuit rows "
            "through admissible target contraction and matched physical rescaling; "
            "not full IEEE-scale execution.",
            "",
            "## 8. Remaining gaps",
            "",
            "- Full sparse-oracle/block-encoding construction for complete IEEE-scale "
            "PSSE remains future work.",
            "- Full-vector readout and scalable state preparation are not solved by "
            "these experiments.",
            "- Phase synthesis and gate-level validation are recorded only where "
            "explicitly attempted or available.",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _degree_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Degree sweep: not generated yet."],
            "findings": ["- Degree sweep findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Degree sweep: generated with no rows."],
            "findings": ["- Degree sweep produced no numerical rows."],
        }
    finite_required = frame["required_degree"].notna() if "required_degree" in frame else []
    pass_count = int(finite_required.sum()) if hasattr(finite_required, "sum") else 0
    no_pass_count = int(len(frame) - pass_count)
    status = [
        f"- Degree sweep: {len(frame)} summary rows; {pass_count} met target "
        f"precision; {no_pass_count} did not meet target precision within the "
        "tested degree grid.",
    ]
    findings = []
    if pass_count:
        finite = frame[finite_required]
        findings.append(
            "- Required degree range among successful summary rows: "
            f"{int(finite['required_degree'].min())} to {int(finite['required_degree'].max())}."
        )
    if "best_error_on_actual_singular_values" in frame:
        findings.append(
            "- Best actual-singular-value approximation error range: "
            f"{frame['best_error_on_actual_singular_values'].min():.3e} to "
            f"{frame['best_error_on_actual_singular_values'].max():.3e}."
        )
    return {"status": status, "findings": findings or ["- Degree sweep generated summary rows."]}


def _block_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Block-encoding demo: not generated yet."],
            "findings": ["- Block-encoding findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Block-encoding demo: generated with no rows."],
            "findings": ["- Block-encoding demo produced no numerical rows."],
        }
    status_counts = frame["run_status"].value_counts().to_dict() if "run_status" in frame else {}
    status = [f"- Block-encoding demo status counts: {status_counts}."]
    completed = frame[frame["run_status"] == "completed"] if "run_status" in frame else frame
    findings = []
    if not completed.empty:
        findings.append(
            "- Maximum completed block Frobenius error: "
            f"{completed['block_error_frobenius'].max():.3e}; maximum unitarity Frobenius error: "
            f"{completed['unitarity_error_frobenius'].max():.3e}."
        )
        findings.append(
            "- Total qubit range for completed dense dilation unitaries: "
            f"{int(completed['total_qubits'].min())} to {int(completed['total_qubits'].max())}."
        )
    return {"status": status, "findings": findings or ["- No completed block-encoding rows."]}


def _end_to_end_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- End-to-end QSVT-vs-Ridge check: not generated yet."],
            "findings": ["- End-to-end QSVT-vs-Ridge findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- End-to-end QSVT-vs-Ridge check: generated with no rows."],
            "findings": ["- End-to-end QSVT-vs-Ridge produced no numerical rows."],
        }
    status_counts = frame["run_status"].value_counts().to_dict() if "run_status" in frame else {}
    target_met = int(frame["target_met"].sum()) if "target_met" in frame else 0
    gate_counts = (
        frame["gate_simulation_status"].value_counts().to_dict()
        if "gate_simulation_status" in frame
        else {}
    )
    status = [
        f"- End-to-end QSVT-vs-Ridge status counts: {status_counts}; "
        f"target-met rows: {target_met}/{len(frame)}; gate statuses: {gate_counts}.",
    ]
    completed = frame[frame["run_status"] == "completed"] if "run_status" in frame else frame
    findings = []
    if not completed.empty:
        findings.append(
            "- Relative update error range: "
            f"{completed['relative_update_error'].min():.3e} to "
            f"{completed['relative_update_error'].max():.3e}."
        )
        findings.append(
            "- Residual-ratio gap range: "
            f"{completed['residual_gap'].min():.3e} to "
            f"{completed['residual_gap'].max():.3e}."
        )
    return {"status": status, "findings": findings or ["- No completed end-to-end rows."]}


def _circuit_block_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Circuit-level block-encoding verification: not generated yet."],
            "findings": ["- Circuit-level block-encoding findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Circuit-level block-encoding verification: generated with no rows."],
            "findings": ["- Circuit-level block-encoding verification produced no numerical rows."],
        }
    simulation_counts = (
        frame["simulation_status"].value_counts().to_dict() if "simulation_status" in frame else {}
    )
    transpile_counts = (
        frame["transpilation_status"].value_counts().to_dict()
        if "transpilation_status" in frame
        else {}
    )
    status = [
        "- Circuit-level block-encoding verification statuses: "
        f"simulation={simulation_counts}; transpilation={transpile_counts}.",
    ]
    completed = (
        frame[frame["simulation_status"].astype(str).str.contains("completed", na=False)]
        if "simulation_status" in frame
        else frame
    )
    findings = []
    if not completed.empty:
        findings.append(
            "- Maximum circuit block Frobenius error: "
            f"{completed['block_fro_error'].max():.3e}; maximum statevector action error: "
            f"{completed['max_statevector_action_abs_error'].max():.3e}."
        )
        findings.append(
            "- Circuit qubit range: "
            f"{int(completed['num_qubits'].min())} to {int(completed['num_qubits'].max())}; "
            "transpiled CX count range for completed decompositions: "
            f"{_range_or_na(frame, 'transpiled_cx_count', 'transpilation_status', 'completed')}."
        )
    return {
        "status": status,
        "findings": findings or ["- No completed circuit-level block-encoding rows."],
    }


def _integrated_qsvt_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Integrated small QSVT circuit sequence: not generated yet."],
            "findings": ["- Integrated small QSVT circuit findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Integrated small QSVT circuit sequence: generated with no rows."],
            "findings": ["- Integrated small QSVT circuit sequence produced no numerical rows."],
        }
    sequence_counts = (
        frame["qsvt_sequence_status"].value_counts().to_dict()
        if "qsvt_sequence_status" in frame
        else {}
    )
    simulation_counts = (
        frame["simulation_status"].value_counts().to_dict() if "simulation_status" in frame else {}
    )
    status = [
        "- Integrated small QSVT circuit sequence statuses: "
        f"sequence={sequence_counts}; simulation={simulation_counts}.",
    ]
    completed = (
        frame[frame["simulation_status"].astype(str).str.contains("completed", na=False)]
        if "simulation_status" in frame
        else frame
    )
    findings = []
    if not completed.empty:
        findings.append(
            "- Integrated QSVT transform block Frobenius error range: "
            f"{completed['transform_block_error_fro'].min():.3e} to "
            f"{completed['transform_block_error_fro'].max():.3e}."
        )
        if "relative_update_error" in completed:
            values = pd.to_numeric(completed["relative_update_error"], errors="coerce").dropna()
            if not values.empty:
                findings.append(
                    "- Integrated QSVT relative update error range: "
                    f"{values.min():.3e} to {values.max():.3e}."
                )
        if "num_U_calls" in completed:
            findings.append(
                "- Integrated QSVT U-call range: "
                f"{int(completed['num_U_calls'].min())} to "
                f"{int(completed['num_U_calls'].max())}."
            )
    return {
        "status": status,
        "findings": findings or ["- No completed integrated QSVT circuit rows."],
    }


def _observable_readout_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Observable-first readout: not generated yet."],
            "findings": ["- Observable-first readout findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Observable-first readout: generated with no rows."],
            "findings": ["- Observable-first readout produced no numerical rows."],
        }
    status_counts = (
        frame["simulation_status"].value_counts().to_dict() if "simulation_status" in frame else {}
    )
    shot_accessible = int(frame["shot_accessible"].sum()) if "shot_accessible" in frame else 0
    status = [
        "- Observable-first readout statuses: "
        f"{status_counts}; shot-accessible summary rows: {shot_accessible}/{len(frame)}.",
    ]
    findings = []
    accessible = (
        frame[frame["shot_accessible"].astype(bool)] if "shot_accessible" in frame else frame
    )
    if not accessible.empty:
        rel = pd.to_numeric(
            accessible["mean_rel_error_vs_qsvt_statevector"], errors="coerce"
        ).dropna()
        ci = pd.to_numeric(accessible["ci95_width"], errors="coerce").dropna()
        if not rel.empty:
            findings.append(
                "- Observable shot relative error versus QSVT statevector range: "
                f"{rel.min():.3e} to {rel.max():.3e}."
            )
        if not ci.empty:
            findings.append(
                f"- Observable empirical 95% CI width range: {ci.min():.3e} to {ci.max():.3e}."
            )
    return {"status": status, "findings": findings or ["- No shot-accessible readout rows."]}


def _sparse_oracle_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Sparse-oracle block-encoding model: not generated yet."],
            "findings": ["- Sparse-oracle block-encoding findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Sparse-oracle block-encoding model: generated with no rows."],
            "findings": ["- Sparse-oracle block-encoding model produced no numerical rows."],
        }
    status_counts = frame["status"].value_counts().to_dict() if "status" in frame else {}
    status = [f"- Sparse-oracle block-encoding model statuses: {status_counts}."]
    completed = frame[frame["status"] == "completed"] if "status" in frame else frame
    findings = []
    if not completed.empty:
        findings.append(
            "- Sparse Jacobian nnz range: "
            f"{int(completed['nnz'].min())} to {int(completed['nnz'].max())}; "
            "density range: "
            f"{completed['density'].min():.3e} to {completed['density'].max():.3e}."
        )
        findings.append(
            "- Sparse-oracle max row sparsity range: "
            f"{int(completed['s'].min())} to {int(completed['s'].max())}; "
            "normalization overhead range: "
            f"{completed['normalization_overhead_sparse_max'].min():.3e} to "
            f"{completed['normalization_overhead_sparse_max'].max():.3e}."
        )
        findings.append(
            "- Sparse-oracle reconstruction Frobenius error range: "
            f"{completed['reconstruction_fro_error'].min():.3e} to "
            f"{completed['reconstruction_fro_error'].max():.3e}."
        )
    return {"status": status, "findings": findings or ["- No completed sparse-oracle rows."]}


def _nonlinear_feasibility_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Nonlinear AC per-iteration feasibility diagnostic: not generated yet."],
            "findings": ["- Nonlinear AC per-iteration feasibility findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": [
                "- Nonlinear AC per-iteration feasibility diagnostic: generated with no rows."
            ],
            "findings": [
                "- Nonlinear AC per-iteration feasibility diagnostic produced no numerical rows."
            ],
        }
    status = [
        f"- Nonlinear AC per-iteration feasibility diagnostic summary rows: {len(frame)}.",
    ]
    findings = []
    if "percentage_target_met" in frame:
        findings.append(
            "- Per-iteration target-met percentage range: "
            f"{frame['percentage_target_met'].min():.2f}% to "
            f"{frame['percentage_target_met'].max():.2f}%."
        )
    if "median_required_degree" in frame:
        required = pd.to_numeric(frame["median_required_degree"], errors="coerce").dropna()
        if not required.empty:
            findings.append(
                "- Median required-degree range across aggregate rows: "
                f"{required.min():.1f} to {required.max():.1f}."
            )
    if "max_condition_number" in frame:
        condition = (
            pd.to_numeric(frame["max_condition_number"], errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )
        if not condition.empty:
            findings.append(
                "- Aggregate maximum condition-number range: "
                f"{condition.min():.3e} to {condition.max():.3e}."
            )
    return {
        "status": status,
        "findings": findings or ["- Nonlinear AC per-iteration feasibility summary generated."],
    }


def _final_robustness_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Final robustness audits: not generated yet."],
            "findings": ["- Final robustness audit findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Final robustness audits: generated with no rows."],
            "findings": ["- Final robustness audits produced no summary rows."],
        }
    status_counts = frame["status"].value_counts().to_dict() if "status" in frame else {}
    completed = int(frame["completed_rows"].sum()) if "completed_rows" in frame else 0
    failed = int(frame["failed_rows"].sum()) if "failed_rows" in frame else 0
    skipped = int(frame["skipped_rows"].sum()) if "skipped_rows" in frame else 0
    status = [
        f"- Final robustness audits: {len(frame)} sub-experiments; statuses={status_counts}; "
        f"completed rows={completed}, failed rows={failed}, skipped rows={skipped}.",
    ]
    findings = []
    if "key_metric" in frame:
        for row in frame.itertuples(index=False):
            findings.append(f"- {row.sub_experiment}: {row.key_metric}.")
    return {
        "status": status,
        "findings": findings or ["- Final robustness audit summary generated."],
    }


def _full_gate_level_coverage_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Full gate-level QSVT coverage audit: not generated yet."],
            "findings": ["- Full gate-level QSVT coverage findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Full gate-level QSVT coverage audit: generated with no rows."],
            "findings": ["- Full gate-level QSVT coverage audit produced no rows."],
        }
    phase_success = (
        int((frame["phase_synthesis_status"] == "completed").sum())
        if "phase_synthesis_status" in frame
        else 0
    )
    circuit_success = (
        int(frame["qsvt_circuit_status"].isin(["completed", "circuit_object_built"]).sum())
        if "qsvt_circuit_status" in frame
        else 0
    )
    simulation_success = (
        int((frame["simulation_status"] == "completed").sum())
        if "simulation_status" in frame
        else 0
    )
    transpiled = (
        int((frame["transpilation_status"] == "completed").sum())
        if "transpilation_status" in frame
        else 0
    )
    skipped = (
        int(frame["simulation_status"].astype(str).str.contains("skipped", na=False).sum())
        if "simulation_status" in frame
        else 0
    )
    status = [
        "- Full gate-level QSVT coverage audit: "
        f"{len(frame)} attempted rows; phase successes={phase_success}; "
        f"circuit successes={circuit_success}; simulations={simulation_success}; "
        f"transpiled={transpiled}; skipped simulation rows={skipped}.",
    ]
    findings = []
    relative = pd.to_numeric(
        frame.get("circuit_vs_ridge_relative_update_error", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not relative.empty:
        findings.append(
            "- Coverage circuit-vs-Ridge relative update error range: "
            f"{relative.min():.3e} to {relative.max():.3e}."
        )
    success = pd.to_numeric(
        frame.get("success_probability", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not success.empty:
        findings.append(
            "- Coverage residual-state success probability range: "
            f"{success.min():.3e} to {success.max():.3e}."
        )
    depth = pd.to_numeric(frame.get("raw_depth", pd.Series(dtype=float)), errors="coerce").dropna()
    if not depth.empty:
        findings.append(
            f"- Coverage raw circuit depth range: {int(depth.min())} to {int(depth.max())}."
        )
    cx = pd.to_numeric(
        frame.get("transpiled_cx_count", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not cx.empty:
        findings.append(
            f"- Coverage transpiled CX count range: {int(cx.min())} to {int(cx.max())}."
        )
    return {
        "status": status,
        "findings": findings or ["- Full gate-level QSVT coverage summary generated."],
    }


def _full_gate_level_remediation_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- Full gate-level degree remediation: not generated yet."],
            "findings": ["- Full gate-level degree remediation findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- Full gate-level degree remediation: generated with no rows."],
            "findings": ["- Full gate-level degree remediation produced no rows."],
        }
    final = frame.drop_duplicates(["case_name", "subproblem_size"], keep="last")
    status_counts = final["case_final_status"].value_counts().to_dict()
    status = [
        "- Full gate-level degree remediation: "
        f"{len(final)} flagged true-mismatch rows; final statuses={status_counts}.",
    ]
    findings = []
    for row in final.itertuples(index=False):
        findings.append(
            "- Degree remediation "
            f"{row.case_name} {int(row.subproblem_size)}x{int(row.subproblem_size)}: "
            f"{row.case_final_status} after candidates up to degree "
            f"{int(row.candidate_degree)}."
        )
    return {"status": status, "findings": findings}


def _phase_synthesis_8x8_rescue_findings(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "status": ["- 8x8 phase-synthesis rescue: not generated yet."],
            "findings": ["- 8x8 phase-synthesis rescue findings unavailable."],
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": ["- 8x8 phase-synthesis rescue: generated with no rows."],
            "findings": ["- 8x8 phase-synthesis rescue produced no rows."],
        }
    rescued = int(frame["rescued"].astype(bool).sum()) if "rescued" in frame else 0
    status = [
        "- 8x8 phase-synthesis rescue: "
        f"{len(frame)} target rows; rescued={rescued}; "
        f"unrescued={len(frame) - rescued}.",
    ]
    findings = []
    for row in frame.itertuples(index=False):
        outcome = "rescued" if bool(row.rescued) else "unrescued"
        findings.append(
            "- 8x8 rescue "
            f"{row.case_name} {int(row.subproblem_size)}x{int(row.subproblem_size)}: "
            f"{outcome} at best degree {int(row.best_degree)}; "
            f"relative update error {_format_scientific(row.best_relative_update_error)}, "
            f"residual gap {_format_scientific(row.best_residual_gap)}."
        )
    return {"status": status, "findings": findings}


def _format_scientific(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "n/a"
    return f"{float(numeric):.3e}"


def _range_or_na(
    frame: pd.DataFrame,
    value_column: str,
    status_column: str,
    status_value: str,
) -> str:
    if value_column not in frame or status_column not in frame:
        return "n/a"
    subset = frame[frame[status_column] == status_value]
    values = pd.to_numeric(subset[value_column], errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"{int(values.min())} to {int(values.max())}"


def _collect_seeds(value: Any) -> list[int]:
    seeds: list[int] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "seed":
                with suppress(Exception):
                    seeds.append(int(item))
            else:
                seeds.extend(_collect_seeds(item))
    elif isinstance(value, list | tuple):
        for item in value:
            seeds.extend(_collect_seeds(item))
    return sorted(set(seeds))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
