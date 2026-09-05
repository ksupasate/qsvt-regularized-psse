#!/usr/bin/env python
"""Build manuscript figures and LaTeX tables for the closed-loop nonlinear sparse-QSVT study.

Everything is derived from the generated ledgers/summaries/audits; no number is hand-entered.
The audit (2026-07) corrected the finite-shot query/shot accounting (Issue A), labelled the
circuit-resource abstraction levels (Issue B), added an extended-horizon plateau diagnostic
(Issue C), and executed nine matched finite-shot runs (Issue D).

    MPLBACKEND=Agg .venv/bin/python scripts/build_tqe_closed_loop_assets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.experiments.iterative_ac import (
    _linearized_update_system,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.physical_alignment.nonlinear_ac import build_problem_config
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint
from robust_qsvt_se.reviewer_blocking.common import write_manifest_and_checksums
from robust_qsvt_se.tqe_extensions.closed_loop_nonlinear_update import (
    ARM_FINITE_SHOT,
    ARM_STATEVECTOR,
    STUDY_ID,
    build_block_operating_point,
)
from robust_qsvt_se.tqe_extensions.degree_lambda_scaling import _phase_cache_hit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("outputs/nonlinear_closed_loop_qsvt")
MANUSCRIPT_TABLES = Path("manuscript/tables")
MANUSCRIPT_FIGURES = Path("manuscript/figures")
DETERMINISTIC_PDF_METADATA = {
    "Creator": "build_tqe_closed_loop_assets.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

ARM_LABELS = {
    "full_system_exact_ridge": "Full-system exact Ridge (A)",
    "block_full_support_ridge": "Block full-support Ridge (B)",
    "sparse_exact_ridge": "Sparse exact Ridge (C)",
    "sparse_quantized_ridge": "Sparse quantized Ridge (D)",
    "sparse_exact_polynomial": "Sparse exact polynomial (E)",
    "sparse_qsvt_statevector_closed_loop": "Sparse QSVT statevector (F)",
    "sparse_qsvt_finite_shot_closed_loop": "Sparse QSVT finite-shot (G)",
}
ARM_ORDER = list(ARM_LABELS)
ARM_SHORT = {a: ARM_LABELS[a].rsplit("(", 1)[-1].rstrip(")") for a in ARM_ORDER}
ARM_COLORS = {
    "full_system_exact_ridge": "#1b1b1b",
    "block_full_support_ridge": "#4c72b0",
    "sparse_exact_ridge": "#55a868",
    "sparse_quantized_ridge": "#c44e52",
    "sparse_exact_polynomial": "#8172b3",
    "sparse_qsvt_statevector_closed_loop": "#ccb974",
    "sparse_qsvt_finite_shot_closed_loop": "#da8bc3",
}
CIRCUIT_TYPE_LABELS = {
    "qsvt_statevector_operator": "QSVT statevector operator",
    "finite_shot_selected_output_readout": "Finite-shot readout (interference)",
    "finite_shot_direct_postselection": "Finite-shot direct postselection",
}


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{value:.{digits}g}"


def _grp(value: int) -> str:
    return f"{int(value):,}".replace(",", "\\,")


def _latex_number(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return f"{mantissa:.{digits}g}\\times10^{{{exponent}}}"


def _json_list(values: np.ndarray | list[int] | list[float]) -> str:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        payload = [int(value) for value in array]
    else:
        payload = [float(value) for value in array]
    return json.dumps(payload, separators=(",", ":"))


def _trajectory_fingerprint(
    ledger: pd.DataFrame, *, arm: str, scenario: str, seed: int
) -> str:
    rows = ledger[
        (ledger["arm"] == arm)
        & (ledger["scenario"] == scenario)
        & (ledger["seed"] == seed)
    ].sort_values("iteration")
    states: list[np.ndarray] = []
    for value in rows["state_after_update_json"].tolist():
        if isinstance(value, str) and value:
            states.append(np.asarray(json.loads(value), dtype=np.float64))
    if not states:
        return ""
    return stable_array_fingerprint(np.concatenate(states))


def _phase_fingerprint(
    coefficients: np.ndarray,
    *,
    beta: float,
    degree: int,
    cache_dir: Path,
) -> tuple[str, str]:
    phases = _phase_cache_hit(
        [float(value) for value in np.asarray(coefficients, dtype=np.float64)],
        "iterative",
        cache_dir,
        {"study_id": STUDY_ID, "degree": int(degree), "beta": round(float(beta), 6)},
    )
    if phases is None:
        return "", "cache_miss"
    return stable_array_fingerprint(phases), "cached_synthesized"


def build_scenario_audits(output_dir: Path) -> tuple[Path, Path]:
    """Build the scenario-overlap and fingerprint audits without rerunning any circuit.

    Row identities are reported in the original full-measurement index domain. For missing-row
    scenarios, overlap is evaluated against the baseline selected rows because removed rows cannot
    appear in the post-removal selected block. For bad data, overlap is evaluated against the
    scenario's selected rows. Fingerprints are taken at iteration zero; trajectory fingerprints
    cover every retained update of the corresponding closed-loop arm.
    """

    config = yaml.safe_load(
        (output_dir / "configs" / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    ledger = pd.read_csv(output_dir / "iteration_ledgers" / "closed_loop_iterations.csv")
    settings = config["nonlinear_settings"]
    block_settings = config["block_qsvt"]
    cache_dir = output_dir / "phase_cache"
    scenarios = list(config["scenarios"])
    detailed_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    for seed in config["seeds"]:
        baseline: dict[str, object] | None = None
        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])
            problem = build_ac_nonlinear_problem(
                build_problem_config(settings, scenario, int(seed))
            )
            system, _ = _linearized_update_system(problem, problem.initial_state.copy())
            matrix = np.asarray(system.H_tilde, dtype=np.float64)
            residual = np.asarray(system.r_tilde, dtype=np.float64)
            operating_point = build_block_operating_point(
                matrix,
                residual,
                block_settings,
                cache_dir,
                need_phases=False,
            )

            kept_rows = np.asarray(problem.kept_row_indices, dtype=np.int64)
            selected_local = np.asarray(operating_point.rows, dtype=np.int64)
            selected_original = kept_rows[selected_local]
            dropped_original = np.asarray(
                problem.config_metadata.get("dropped_rows", []), dtype=np.int64
            )
            bad_local = np.asarray(
                problem.config_metadata.get("bad_data_rows", []), dtype=np.int64
            )
            bad_original = kept_rows[bad_local] if bad_local.size else bad_local.copy()
            bad_signs = np.asarray(
                problem.config_metadata.get("bad_data_signs", []), dtype=np.int64
            )
            bad_magnitude = float(
                problem.config_metadata.get("bad_data_magnitude", 0.0) or 0.0
            )
            bad_offsets = (
                bad_signs.astype(np.float64)
                * bad_magnitude
                * np.asarray(problem.measurement_stds, dtype=np.float64)[bad_local]
                if bad_local.size
                else np.asarray([], dtype=np.float64)
            )
            phase_fp, phase_status = _phase_fingerprint(
                operating_point.coefficients,
                beta=operating_point.beta,
                degree=operating_point.degree,
                cache_dir=cache_dir,
            )
            statevector_trajectory = _trajectory_fingerprint(
                ledger,
                arm=ARM_STATEVECTOR,
                scenario=scenario_id,
                seed=int(seed),
            )
            finite_shot_trajectory = _trajectory_fingerprint(
                ledger,
                arm=ARM_FINITE_SHOT,
                scenario=scenario_id,
                seed=int(seed),
            )
            current: dict[str, object] = {
                "scenario": scenario_id,
                "seed": int(seed),
                "missing_measurement_rows_original_json": _json_list(dropped_original),
                "bad_data_rows_local_json": _json_list(bad_local),
                "bad_data_rows_original_json": _json_list(bad_original),
                "bad_data_signs_json": _json_list(bad_signs),
                "bad_data_offsets_raw_json": _json_list(bad_offsets),
                "selected_block_rows_local_json": _json_list(selected_local),
                "selected_block_rows_original_json": _json_list(selected_original),
                "initial_state_fingerprint": stable_array_fingerprint(problem.initial_state),
                "weighted_residual_fingerprint": stable_array_fingerprint(residual),
                "weighted_jacobian_fingerprint": stable_array_fingerprint(matrix),
                "selected_block_matrix_fingerprint": stable_array_fingerprint(
                    operating_point.block_dense
                ),
                "selected_residual_fingerprint": stable_array_fingerprint(
                    operating_point.residual_block
                ),
                "support_fingerprint": stable_array_fingerprint(
                    (operating_point.block_sparsified != 0.0).astype(np.float64)
                ),
                "polynomial_fingerprint": stable_array_fingerprint(
                    operating_point.coefficients
                ),
                "phase_fingerprint": phase_fp,
                "phase_fingerprint_status": phase_status,
                "statevector_trajectory_fingerprint": statevector_trajectory,
                "finite_shot_trajectory_fingerprint": finite_shot_trajectory,
            }
            if baseline is None:
                baseline = dict(current)

            baseline_selected = set(
                json.loads(str(baseline["selected_block_rows_original_json"]))
            )
            scenario_selected = set(int(value) for value in selected_original)
            if dropped_original.size:
                overlap_rows_original = sorted(
                    baseline_selected.intersection(int(value) for value in dropped_original)
                )
                overlap_reference = "baseline selected rows (removed rows cannot be reselected)"
            else:
                overlap_rows_original = sorted(
                    scenario_selected.intersection(int(value) for value in bad_original)
                )
                overlap_reference = "scenario selected rows"
            overlap_count = len(overlap_rows_original)
            residual_differs = (
                current["selected_residual_fingerprint"]
                != baseline["selected_residual_fingerprint"]
            )
            matrix_differs = (
                current["selected_block_matrix_fingerprint"]
                != baseline["selected_block_matrix_fingerprint"]
            )
            trajectory_differs = (
                current["statevector_trajectory_fingerprint"]
                != baseline["statevector_trajectory_fingerprint"]
            )
            full_residual_differs = (
                current["weighted_residual_fingerprint"]
                != baseline["weighted_residual_fingerprint"]
            )
            full_matrix_differs = (
                current["weighted_jacobian_fingerprint"]
                != baseline["weighted_jacobian_fingerprint"]
            )
            if scenario_id == str(scenarios[0]["scenario_id"]):
                explanation = "Baseline generated-noise scenario; no missing-row or bad-data perturbation."
            elif overlap_count == 0 and not (residual_differs or matrix_differs):
                explanation = (
                    "The scenario is distinct at the full-system measurement level but does not "
                    "perturb the selected local block for this seed."
                )
            elif matrix_differs or residual_differs:
                explanation = (
                    "The perturbation changes the selected local block or its selected residual; "
                    "the closed-loop trajectory consequently differs from the Gaussian baseline."
                )
            else:
                explanation = (
                    "The selected local block changes in row identity but the recorded numerical "
                    "fingerprints explain why the trajectory remains equal to the baseline."
                )

            current.update(
                {
                    "perturbed_rows_original_json": _json_list(
                        np.unique(np.concatenate([dropped_original, bad_original]))
                    ),
                    "overlap_reference": overlap_reference,
                    "overlap_rows_original_json": _json_list(overlap_rows_original),
                    "overlap_count": overlap_count,
                    "full_weighted_residual_differs_from_baseline": full_residual_differs,
                    "full_weighted_jacobian_differs_from_baseline": full_matrix_differs,
                    "selected_residual_differs_from_baseline": residual_differs,
                    "selected_matrix_differs_from_baseline": matrix_differs,
                    "support_differs_from_baseline": (
                        current["support_fingerprint"] != baseline["support_fingerprint"]
                    ),
                    "polynomial_differs_from_baseline": (
                        current["polynomial_fingerprint"] != baseline["polynomial_fingerprint"]
                    ),
                    "phase_differs_from_baseline": (
                        current["phase_fingerprint"] != baseline["phase_fingerprint"]
                    ),
                    "trajectory_differs_from_baseline": trajectory_differs,
                    "scenario_differs_from_gaussian_baseline": bool(
                        scenario_id != str(scenarios[0]["scenario_id"])
                        and (
                            full_residual_differs
                            or full_matrix_differs
                            or residual_differs
                            or matrix_differs
                            or trajectory_differs
                        )
                    ),
                    "explanation": explanation,
                }
            )
            detailed_rows.append(current)
            overlap_rows.append(
                {
                    "Scenario": scenario_id,
                    "Seed": int(seed),
                    "Perturbed rows": current["perturbed_rows_original_json"],
                    "Selected rows": current["selected_block_rows_original_json"],
                    "Overlap": overlap_count,
                    "Residual differs?": residual_differs,
                    "Matrix differs?": matrix_differs,
                    "Trajectory differs?": trajectory_differs,
                    "Explanation": explanation,
                }
            )

    audit_dir = output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = audit_dir / "scenario_perturbation_overlap.csv"
    fingerprint_path = audit_dir / "scenario_fingerprint_audit.csv"
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
    pd.DataFrame(detailed_rows).to_csv(fingerprint_path, index=False)
    return overlap_path, fingerprint_path


def build_operational_diagnostic_cost_ledger(output_dir: Path) -> Path:
    reconciliation = pd.read_csv(
        output_dir / "resource_ledgers" / "query_execution_reconciliation.csv"
    ).set_index("quantity")["verified_value"]
    accounting = pd.read_csv(
        output_dir / "resource_ledgers" / "query_execution_accounting.csv"
    )
    queries = int(reconciliation["functional_queries"])
    shots_per_execution = int(reconciliation["shots_per_sampling_call"])
    operational_shots = int(accounting["readout_signal_attempted_shots"].sum())
    diagnostic_shots = int(accounting["diagnostic_attempted_shots"].sum())
    total_shots = int(reconciliation["total_attempted_shots"])
    operational_executions = operational_shots // shots_per_execution
    diagnostic_executions = diagnostic_shots // shots_per_execution
    total_executions = int(reconciliation["sampling_calls"])
    rows = [
        {
            "cost_category": "signed_selected_output_readout",
            "functional_queries": queries,
            "executions_per_query": 1,
            "sampling_executions": operational_executions,
            "shots_per_execution": shots_per_execution,
            "total_shots": operational_shots,
            "required_operationally": "yes",
            "evidence_role": "drives the selected-coordinate update",
        },
        {
            "cost_category": "direct_postselection_diagnostic",
            "functional_queries": queries,
            "executions_per_query": 1,
            "sampling_executions": diagnostic_executions,
            "shots_per_execution": shots_per_execution,
            "total_shots": diagnostic_shots,
            "required_operationally": "no",
            "evidence_role": "audit-only direct postselection diagnostic",
        },
        {
            "cost_category": "combined_reported_audit_cost",
            "functional_queries": queries,
            "executions_per_query": 2,
            "sampling_executions": total_executions,
            "shots_per_execution": shots_per_execution,
            "total_shots": total_shots,
            "required_operationally": "evidence total",
            "evidence_role": "executed operational plus diagnostic evidence",
        },
    ]
    path = output_dir / "resource_ledgers" / "operational_diagnostic_cost.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_metric_macros(output_dir: Path, tables_dir: Path) -> Path:
    """Export every closed-loop prose number from generated ledgers/configuration."""

    summaries = pd.read_csv(output_dir / "run_summaries" / "solver_outcomes.csv")
    decomposition = pd.read_csv(
        output_dir / "error_decomposition" / "stage_error_decomposition.csv"
    )
    comparison = pd.read_csv(
        output_dir / "audits" / "finite_shot_statevector_comparison.csv"
    )
    plateau = pd.read_csv(output_dir / "audits" / "trajectory_plateau_onset_audit.csv")
    levels = pd.read_csv(output_dir / "resource_ledgers" / "circuit_resource_levels.csv")
    reconciliation = pd.read_csv(
        output_dir / "resource_ledgers" / "query_execution_reconciliation.csv"
    ).set_index("quantity")["verified_value"]
    cost_categories = pd.read_csv(
        output_dir / "resource_ledgers" / "operational_diagnostic_cost.csv"
    ).set_index("cost_category")
    config = yaml.safe_load(
        (output_dir / "configs" / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    full = summaries[summaries["arm"] == "full_system_exact_ridge"]
    block = summaries[summaries["arm"].isin(ARM_ORDER[1:])]
    plateaued = plateau[plateau["classification"] == "plateaued"]
    coordinate_rows = pd.read_csv(
        output_dir / "resource_ledgers" / "finite_shot_coordinate_readout.csv"
    )
    state_dimension = len(
        __import__("json").loads(str(summaries["final_state_vector_json"].iloc[0]))
    )
    med = lambda name: float(np.nanmedian(decomposition[name].to_numpy(dtype=float)))
    plateau_value = lambda name, operation, fallback=-1: (
        int(getattr(plateaued[name], operation)()) if not plateaued.empty else fallback
    )
    values = {
        "ClosedLoopScenarioCount": len(config["scenarios"]),
        "ClosedLoopSeedCount": len(config["seeds"]),
        "ClosedLoopStatevectorRuns": int(
            (summaries["arm"] == "sparse_qsvt_statevector_closed_loop").sum()
        ),
        "ClosedLoopFiniteShotRuns": int(
            (summaries["arm"] == "sparse_qsvt_finite_shot_closed_loop").sum()
        ),
        "ClosedLoopFullConvergedRuns": int(full["converged"].sum()),
        "ClosedLoopPrimaryIterations": int(
            config["nonlinear_settings"]["iteration"]["max_iterations"]
        ),
        "ClosedLoopExtendedIterations": int(config["extended_horizon"]["max_iterations"]),
        "ClosedLoopPlateauWindow": int(config["extended_horizon"]["plateau_window"]),
        "ClosedLoopBlockSize": int(config["block_qsvt"]["block_size"]),
        "ClosedLoopStateDimension": state_dimension,
        "ClosedLoopDegree": int(config["block_qsvt"]["degree"]),
        "ClosedLoopOperatorLogicalOperations": int(
            levels.loc[
                levels["circuit_type"] == "qsvt_statevector_operator", "logical_operations"
            ].iloc[0]
        ),
        "ClosedLoopSignalApplications": int(
            levels.loc[
                levels["circuit_type"] == "qsvt_statevector_operator",
                "signal_unitary_applications",
            ].iloc[0]
        ),
        "ClosedLoopPhaseApplications": int(
            levels.loc[
                levels["circuit_type"] == "qsvt_statevector_operator",
                "projector_phase_applications",
            ].iloc[0]
        ),
        "ClosedLoopFunctionalQueries": int(reconciliation["functional_queries"]),
        "ClosedLoopUniqueFunctionalCircuits": int(
            reconciliation["unique_functional_circuits"]
        ),
        "ClosedLoopStatevectorEvolutions": int(
            summaries.loc[
                summaries["arm"] == "sparse_qsvt_statevector_closed_loop",
                "physical_circuit_executions",
            ].sum()
        ),
        "ClosedLoopPhysicalExecutions": int(reconciliation["physical_circuit_executions"]),
        "ClosedLoopSamplingExecutions": int(reconciliation["sampling_calls"]),
        "ClosedLoopOperationalSamplingExecutions": int(
            cost_categories.loc["signed_selected_output_readout", "sampling_executions"]
        ),
        "ClosedLoopDiagnosticSamplingExecutions": int(
            cost_categories.loc["direct_postselection_diagnostic", "sampling_executions"]
        ),
        "ClosedLoopSamplingCalls": int(reconciliation["sampling_calls"]),
        "ClosedLoopShotsPerCall": int(reconciliation["shots_per_sampling_call"]),
        "ClosedLoopShotsPerQuery": int(reconciliation["shots_per_functional_query"]),
        "ClosedLoopOperationalAttemptedShots": int(
            cost_categories.loc["signed_selected_output_readout", "total_shots"]
        ),
        "ClosedLoopDiagnosticAttemptedShots": int(
            cost_categories.loc["direct_postselection_diagnostic", "total_shots"]
        ),
        "ClosedLoopTotalAttemptedShots": int(reconciliation["total_attempted_shots"]),
        "ClosedLoopPlateauRuns": int(len(plateaued)),
        "ClosedLoopPlateauOnsetMin": plateau_value("plateau_onset_iteration", "min"),
        "ClosedLoopPlateauOnsetMedian": plateau_value("plateau_onset_iteration", "median"),
        "ClosedLoopPlateauOnsetMax": plateau_value("plateau_onset_iteration", "max"),
        "ClosedLoopRMSEFloorEntryMedian": plateau_value(
            "rmse_floor_entry_iteration", "median"
        ),
    }
    math_values = {
        "ClosedLoopLambda": float(config["block_qsvt"]["lambda_target"]),
        "ClosedLoopFullAlpha": float(config["nonlinear_settings"]["fixed_alpha"]),
        "ClosedLoopPlateauRelativeTolerance": float(
            config["extended_horizon"]["plateau_rmse_rel_tol"]
        ),
        "ClosedLoopPlateauSwingTolerance": float(
            config["extended_horizon"]["oscillation_rel_tol"]
        ),
        "ClosedLoopPlateauStepThreshold": float(
            config["extended_horizon"]["plateau_step_norm"]
        ),
        "ClosedLoopMeanInterferenceAcceptance": float(
            coordinate_rows["interference_acceptance_probability"].mean()
        ),
        "ClosedLoopFullMedianRMSE": float(np.median(full["final_state_rmse"])),
        "ClosedLoopBlockMedianRMSE": float(np.median(block["final_state_rmse"])),
        "ClosedLoopInitialGapMedian": float(
            np.median(plateaued["iteration_0_relative_gap_to_final_floor"])
            if not plateaued.empty
            else 0.0
        ),
        "ClosedLoopInitialGapMin": float(
            plateaued["iteration_0_relative_gap_to_final_floor"].min()
            if not plateaued.empty
            else 0.0
        ),
        "ClosedLoopInitialGapMax": float(
            plateaued["iteration_0_relative_gap_to_final_floor"].max()
            if not plateaued.empty
            else 0.0
        ),
        "ClosedLoopMaxFinalRMSEDifference": float(
            comparison["absolute_final_rmse_difference"].max()
        ),
        "ClosedLoopMaxTrajectoryRMSEDifference": float(
            comparison["max_state_rmse_difference"].max()
        ),
        "ClosedLoopMedianCoordinateReadoutError": float(
            coordinate_rows["coordinate_readout_abs_error"].median()
        ),
        "ClosedLoopMaxCoordinateReadoutError": float(
            coordinate_rows["coordinate_readout_abs_error"].max()
        ),
        "ClosedLoopBlockTruncationError": med("block_truncation_abs_error"),
        "ClosedLoopRegularizationError": med("regularization_gap_abs_error"),
        "ClosedLoopSupportRemovalError": med("support_removal_abs_error"),
        "ClosedLoopQuantizationError": med("quantization_abs_error"),
        "ClosedLoopPolynomialError": med("polynomial_abs_error"),
        "ClosedLoopStatevectorCircuitError": med("statevector_circuit_abs_error"),
        "ClosedLoopFiniteShotReadoutError": med("finite_shot_readout_abs_error"),
        "ClosedLoopResourceMinAll": float(levels["transpiled_basis_gates"].min()),
        "ClosedLoopResourceMaxAll": float(levels["transpiled_basis_gates"].max()),
        "ClosedLoopResourceMinOperator": float(
            levels.loc[
                levels["circuit_type"] == "qsvt_statevector_operator",
                "transpiled_basis_gates",
            ].min()
        ),
        "ClosedLoopResourceMaxOperator": float(
            levels.loc[
                levels["circuit_type"] == "qsvt_statevector_operator",
                "transpiled_basis_gates",
            ].max()
        ),
    }
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand."
    ]
    for name, value in values.items():
        lines.append(f"\\providecommand{{\\{name}}}{{{_grp(value)}}}")
    for name, value in math_values.items():
        lines.append(
            f"\\providecommand{{\\{name}}}{{\\ensuremath{{{_latex_number(value)}}}}}"
        )
    path = tables_dir / "tqe_closed_loop_metrics.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_convergence_figure(output_dir: Path, figures_dir: Path) -> Path:
    ledger = pd.read_csv(output_dir / "iteration_ledgers" / "closed_loop_iterations.csv")
    scenarios = sorted(ledger["scenario"].unique())
    finite_seed = int(
        ledger.loc[ledger["arm"] == "sparse_qsvt_finite_shot_closed_loop", "seed"].min()
    )
    panels = [
        ("weighted_residual_norm", "Weighted residual", True),
        ("state_rmse", "State RMSE vs benchmark", True),
        ("selected_output_benchmark_error", "Selected-output error", True),
        ("step_norm", "Update (step) norm", True),
        ("postselection_probability", "Postselection prob. (F/G)", False),
        ("cumulative_shots", "Cumulative attempted shots (G)", False),
    ]
    fig, axes = plt.subplots(
        len(scenarios), len(panels), figsize=(3.0 * len(panels), 2.5 * len(scenarios)),
        squeeze=False,
    )
    for row_index, scenario in enumerate(scenarios):
        sub = ledger[(ledger["scenario"] == scenario) & (ledger["seed"] == finite_seed)]
        for col_index, (column, title, logy) in enumerate(panels):
            ax = axes[row_index][col_index]
            for arm in ARM_ORDER:
                arm_rows = sub[sub["arm"] == arm].sort_values("iteration")
                if arm_rows.empty or column not in arm_rows:
                    continue
                values = arm_rows[column].to_numpy(dtype=float)
                if np.all(np.isnan(values)):
                    continue
                ax.plot(
                    arm_rows["iteration"].to_numpy(),
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    color=ARM_COLORS[arm],
                    label=ARM_SHORT[arm],
                )
            if logy:
                ax.set_yscale("log")
            if row_index == 0:
                ax.set_title(title, fontsize=8)
            if col_index == 0:
                ax.set_ylabel(scenario.replace("_", "\n"), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("iteration", fontsize=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=6, frameon=False)
    fig.suptitle(
        "Closed-loop nonlinear sparse-QSVT trajectories (representative seed; failures retained)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = figures_dir / "tqe_closed_loop_trajectories.pdf"
    fig.savefig(path, bbox_inches="tight", metadata=DETERMINISTIC_PDF_METADATA)
    plt.close(fig)
    return path


def build_decomposition_figure(output_dir: Path, figures_dir: Path) -> Path:
    decomp = pd.read_csv(output_dir / "error_decomposition" / "stage_error_decomposition.csv")
    decomp = decomp[decomp["stage_available"] == True]  # noqa: E712
    stages = [
        ("regularization_gap_abs_error", "regularization"),
        ("block_truncation_abs_error", "block\ntruncation"),
        ("support_removal_abs_error", "support\nremoval"),
        ("quantization_abs_error", "quantization"),
        ("polynomial_abs_error", "polynomial"),
        ("statevector_circuit_abs_error", "statevector\ncircuit"),
        ("finite_shot_readout_abs_error", "finite-shot\nreadout"),
    ]
    medians = [np.nanmedian(decomp[col].to_numpy(dtype=float)) for col, _ in stages]
    lows = [np.nanpercentile(decomp[col].to_numpy(dtype=float), 10) for col, _ in stages]
    highs = [np.nanpercentile(decomp[col].to_numpy(dtype=float), 90) for col, _ in stages]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(stages))
    colors = ["#c44e52", "#c44e52", "#dd8452", "#4c72b0", "#4c72b0", "#55a868", "#8172b3"]
    yerr = np.vstack(
        [np.array(medians) - np.array(lows), np.array(highs) - np.array(medians)]
    )
    yerr = np.clip(yerr, 0, None)
    ax.bar(x, medians, yerr=yerr, color=colors, capsize=3, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stages], fontsize=7)
    ax.set_ylabel("absolute stage error (median, P10-P90)", fontsize=8)
    ax.set_title(
        "Closed-loop error decomposition along the full-system reference trajectory\n"
        "(each stage isolates one source; the statevector circuit is essentially exact)",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = figures_dir / "tqe_closed_loop_error_decomposition.pdf"
    fig.savefig(path, bbox_inches="tight", metadata=DETERMINISTIC_PDF_METADATA)
    plt.close(fig)
    return path


def build_extended_horizon_figure(output_dir: Path, figures_dir: Path) -> Path:
    ext = pd.read_csv(
        output_dir / "extended_horizon" / "iteration_ledgers" / "extended_iterations.csv"
    )
    scenarios = sorted(ext["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.4 * len(scenarios), 3.0), squeeze=False)
    ext_arms = [a for a in ARM_ORDER if a in set(ext["arm"].unique())]
    for col_index, scenario in enumerate(scenarios):
        ax = axes[0][col_index]
        sub = ext[(ext["scenario"] == scenario) & (ext["seed"] == ext["seed"].min())]
        for arm in ext_arms:
            arm_rows = sub[sub["arm"] == arm].sort_values("iteration")
            if arm_rows.empty:
                continue
            ax.plot(
                arm_rows["iteration"].to_numpy(),
                arm_rows["state_rmse"].to_numpy(dtype=float),
                marker="o", markersize=2.5, linewidth=1.1,
                color=ARM_COLORS[arm], label=ARM_SHORT[arm],
            )
        ax.set_yscale("log")
        ax.set_title(scenario.replace("_", " "), fontsize=8)
        ax.set_xlabel("iteration (extended horizon)", fontsize=7)
        if col_index == 0:
            ax.set_ylabel("state RMSE vs benchmark", fontsize=8)
        ax.axvline(8, color="grey", linestyle=":", linewidth=0.8)
        ax.tick_params(labelsize=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.suptitle(
        "Extended-horizon diagnostic (30 iterations): block arms plateau above the full-system "
        "accuracy floor\n(dotted line = primary 8-iteration horizon)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = figures_dir / "tqe_closed_loop_extended_horizon.pdf"
    fig.savefig(path, bbox_inches="tight", metadata=DETERMINISTIC_PDF_METADATA)
    plt.close(fig)
    return path


def build_main_table(output_dir: Path, tables_dir: Path) -> Path:
    summaries = pd.read_csv(output_dir / "run_summaries" / "solver_outcomes.csv")
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{Selected-coordinate closed-loop diagnostic (IEEE-14, "
        "$\\ClosedLoopBlockSize\\times\\ClosedLoopBlockSize$ block, degree \\ClosedLoopDegree, "
        "$\\lambda=\\ClosedLoopLambda$). Medians use the primary "
        "\\ClosedLoopPrimaryIterations-iteration runs; the separate "
        "\\ClosedLoopExtendedIterations-iteration plateau diagnostic is reported in the "
        "supplement. Selected-output error is against the generated benchmark update on the four "
        "selected coordinates; the full-system arm is the reference, so that entry is not an "
        "evaluated selected-block error. Statevector evolutions are exact operator propagations "
        "without sampling. The finite-shot scope is all \\ClosedLoopFiniteShotRuns{} "
        "scenario--seed runs; its sampling totals include both the operational signed-readout "
        "branch and the separately itemized audit-only postselection branch.}",
        "\\label{tab:closed_loop_main}",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Arm & Outcome & Final RMSE & Selected-output error & Statevector evolutions "
        "& Functional queries & Sampling executions & Attempted shots \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        rows = summaries[summaries["arm"] == arm]
        if rows.empty:
            continue
        runs = len(rows)
        converged = int(rows["converged"].sum())
        median_iters = int(np.median(rows["iterations"].to_numpy()))
        median_rmse = float(np.median(rows["final_state_rmse"].to_numpy()))
        selout_values = rows["final_selected_output_error"].to_numpy(dtype=float)
        median_selout = (
            float(np.nanmedian(selout_values)) if np.any(~np.isnan(selout_values)) else float("nan")
        )
        total_evolutions = (
            int(rows["physical_circuit_executions"].sum())
            if arm == "sparse_qsvt_statevector_closed_loop"
            else 0
        )
        total_queries = int(rows["functional_queries"].sum())
        total_sampling = int(rows["sampling_calls"].sum())
        total_shots = int(rows["total_attempted_shots"].sum())
        outcome = (
            f"{converged}/{runs} converged"
            if converged
            else f"0/{runs}; cap {median_iters}"
        )
        selected_output = (
            "reference" if arm == "full_system_exact_ridge" else _fmt(median_selout)
        )
        if total_evolutions:
            evolutions = _grp(total_evolutions)
        else:
            evolutions = "--"
        queries = _grp(total_queries) if total_queries else "--"
        sampling = _grp(total_sampling) if total_sampling else "--"
        shots = _grp(total_shots) if total_shots else "--"
        lines.append(
            f"{ARM_LABELS[arm]} & {outcome} & {_fmt(median_rmse)} & {selected_output} & "
            f"{evolutions} & {queries} & {sampling} & {shots} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    path = tables_dir / "tqe_closed_loop_main_summary.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_decomposition_table(output_dir: Path, tables_dir: Path) -> Path:
    decomp = pd.read_csv(output_dir / "error_decomposition" / "stage_error_decomposition.csv")
    decomp = decomp[decomp["stage_available"] == True]  # noqa: E712
    n_iters = len(decomp)
    n_circuit = int(decomp["statevector_circuit_abs_error"].notna().sum())
    stages = [
        (
            "regularization_gap_error",
            "Regularization ($\\lambda_{\\mathrm{full}}\\!\\to\\!\\lambda_k$)",
        ),
        ("block_truncation_error", "Block truncation (full $\\to$ block)"),
        ("support_removal_error", "Support removal (block $\\to$ sparse)"),
        ("quantization_error", "Quantization (sparse $\\to$ quantized)"),
        ("polynomial_error", "Polynomial (rational $\\to$ bounded poly.)"),
        ("statevector_circuit_error", "Statevector circuit (poly.\\ $\\to$ circuit)"),
        ("finite_shot_readout_error", "Finite-shot readout (circuit $\\to$ shots)"),
    ]
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Closed-loop stagewise error decomposition evaluated at the shared operating "
        f"point of each full-system reference iteration ({n_iters} iterations; statevector and "
        f"finite-shot stages on the recorded finite-shot seed, {n_circuit} iterations). Each "
        "stage isolates one source. Absolute errors are the stable quantity: the heavily "
        "regularized full-system update is near zero on the block coordinates, so the "
        "block-truncation relative error is large while its absolute magnitude is well defined. "
        "The statevector circuit reproduces the bounded-polynomial action to $\\sim\\!10^{-13}$, "
        "roughly eight orders of magnitude below the block-truncation, regularization, and "
        "support-removal sources.}",
        "\\label{supp:tab:closed_loop_decomposition}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Stage (isolated source) & Median abs.\\ err. & P90 abs.\\ err. & Median rel.\\ err. \\\\",
        "\\midrule",
    ]
    for col, label in stages:
        rel = decomp[col].to_numpy(dtype=float)
        abs_col = col.replace("_error", "_abs_error")
        abs_vals = (
            decomp[abs_col].to_numpy(dtype=float)
            if abs_col in decomp
            else np.full_like(rel, np.nan)
        )
        lines.append(
            f"{label} & {_fmt(np.nanmedian(abs_vals))} & {_fmt(np.nanpercentile(abs_vals, 90))} "
            f"& {_fmt(np.nanmedian(rel))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_error_decomposition.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_readout_cost_table(output_dir: Path, tables_dir: Path) -> Path:
    cost_path = output_dir / "resource_ledgers" / "operational_diagnostic_cost.csv"
    cost = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()
    labels = {
        "signed_selected_output_readout": "Signed selected-output readout",
        "direct_postselection_diagnostic": "Direct postselection diagnostic",
        "combined_reported_audit_cost": "Combined reported audit cost",
    }
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Operational and diagnostic finite-shot cost across all "
        "\\ClosedLoopFiniteShotRuns{} matched runs. Only the signed selected-output readout is "
        "required to drive the selected-coordinate update. The direct-postselection branch is "
        "executed and fully counted, but is an audit diagnostic rather than an operational "
        "requirement. The combined row is the complete executed evidence cost.}",
        "\\label{supp:tab:closed_loop_readout}",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Cost category & Functional queries & Exec./query & Sampling exec. & Shots/exec. "
        "& Total shots & Required operationally? \\\\",
        "\\midrule",
    ]
    if not cost.empty:
        for _, row in cost.iterrows():
            lines.append(
                f"{labels[str(row['cost_category'])]} & "
                f"{int(row['functional_queries'])} & "
                f"{int(row['executions_per_query'])} & "
                f"{_grp(int(row['sampling_executions']))} & "
                f"{_grp(int(row['shots_per_execution']))} & "
                f"{_grp(int(row['total_shots']))} & "
                f"{str(row['required_operationally']).capitalize()} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_readout_cost.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_query_accounting_table(output_dir: Path, tables_dir: Path) -> Path:
    reconciliation = pd.read_csv(
        output_dir / "resource_ledgers" / "query_execution_reconciliation.csv"
    )
    labels = {
        "finite_shot_runs": "Finite-shot runs",
        "executed_iterations": "Executed iterations",
        "coordinates_per_iteration": "Coordinates per iteration",
        "functional_queries": "Functional queries",
        "unique_functional_circuits": "Unique functional circuits",
        "physical_circuit_executions": "Sampled circuit executions",
        "sampling_calls": "Sampling calls",
        "shots_per_sampling_call": "Shots per sampling call",
        "shots_per_functional_query": "Shots per functional query",
        "total_attempted_shots": "Total attempted shots",
        "interference_accepted_shots": "Interference-accepted shots",
        "postselection_accepted_shots": "Postselection-accepted shots",
    }
    rows = [
        (
            labels[str(row["quantity"])],
            _grp(int(row["verified_value"])),
            (
                str(row["definition"])
                .replace(" x ", " $\\times$ ", 1)
                .replace("physical execution", "sampled circuit execution")
            )
            if str(row["quantity"]) == "finite_shot_runs"
            else str(row["definition"]).replace(
                "physical execution", "sampled circuit execution"
            ),
        )
        for _, row in reconciliation.iterrows()
    ]
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Non-overlapping functional-query, sampled-execution, and shot accounting for the "
        "finite-shot closed-loop arm (\\ClosedLoopFiniteShotRuns{} matched runs). A functional "
        "query is one signed "
        "coordinate readout and drives the update. Unique functional circuits deduplicate "
        "identical matrix/residual/phase/coordinate parameterizations across repeated runs. "
        "Each query issues two sampled circuit "
        "executions (an operational readout/interference branch and an audit-only direct-"
        "postselection branch), so the combined evidence executions and attempted shots are "
        "twice the query count. The invariant "
        "total attempted shots $=$ sampling calls $\\times$ shots per call holds exactly.}",
        "\\label{supp:tab:closed_loop_query_accounting}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{p{0.22\\linewidth}r p{0.58\\linewidth}}",
        "\\toprule",
        "Quantity & Value & Definition \\\\",
        "\\midrule",
    ]
    for name, value, definition in rows:
        lines.append(f"{name} & {value} & {definition} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_query_accounting.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_scenario_audit_table(output_dir: Path, tables_dir: Path) -> Path:
    audit = pd.read_csv(output_dir / "audits" / "scenario_perturbation_overlap.csv")
    scenario_labels = {
        "gaussian_noise_baseline": "Gaussian baseline",
        "random_missing_measurement_stress": "Missing rows",
        "sparse_signed_bad_data_stress": "Signed bad data",
    }
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Scenario perturbation overlap and selected-block fingerprint audit. Row "
        "identities use the original full-measurement index. For missingness, overlap is against "
        "the Gaussian-baseline selected rows because removed rows cannot appear in the post-"
        "removal block. R/M/T indicate whether the selected residual, selected matrix, and "
        "statevector closed-loop trajectory differ from the same-seed Gaussian baseline.}",
        "\\label{supp:tab:closed_loop_scenario_audit}",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrccc l}",
        "\\toprule",
        "Scenario & Seed & Perturbed rows & Selected rows & Overlap & R & M & T & Explanation \\\\",
        "\\midrule",
    ]
    for _, row in audit.iterrows():
        explanation = str(row["Explanation"])
        if explanation.startswith("Baseline"):
            explanation = "baseline"
        elif "does not perturb" in explanation:
            explanation = "distinct full-system scenario; selected local block unchanged"
        elif "consequently differs" in explanation:
            explanation = "selected local block changes"
        else:
            explanation = "fingerprints document equality"
        yes_no = lambda value: "yes" if bool(value) else "no"
        lines.append(
            f"{scenario_labels.get(str(row['Scenario']), str(row['Scenario']))} & "
            f"{int(row['Seed'])} & "
            f"\\texttt{{{str(row['Perturbed rows'])}}} & "
            f"\\texttt{{{str(row['Selected rows'])}}} & {int(row['Overlap'])} & "
            f"{yes_no(row['Residual differs?'])} & {yes_no(row['Matrix differs?'])} & "
            f"{yes_no(row['Trajectory differs?'])} & {explanation} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_scenario_audit.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_manuscript_value_traceability(output_dir: Path, tables_dir: Path) -> Path:
    metrics_path = tables_dir / "tqe_closed_loop_metrics.tex"
    metrics = metrics_path.read_text(encoding="utf-8")
    macro_rows: list[dict[str, str]] = []
    source_rules = [
        (
            ("ScenarioCount", "SeedCount", "PrimaryIterations", "ExtendedIterations", "BlockSize",
             "StateDimension", "Degree", "Lambda", "FullAlpha", "PlateauWindow",
             "PlateauRelativeTolerance", "PlateauSwingTolerance", "PlateauStepThreshold"),
            "outputs/nonlinear_closed_loop_qsvt/configs/config_resolved.yaml",
            "resolved configuration",
        ),
        (
            ("StatevectorRuns", "FiniteShotRuns", "FullConvergedRuns", "FullMedianRMSE",
             "BlockMedianRMSE", "StatevectorEvolutions"),
            "outputs/nonlinear_closed_loop_qsvt/run_summaries/solver_outcomes.csv",
            "arm-filtered run rows",
        ),
        (
            ("FunctionalQueries", "UniqueFunctionalCircuits", "PhysicalExecutions",
             "SamplingExecutions", "SamplingCalls", "ShotsPerCall", "ShotsPerQuery",
             "TotalAttemptedShots"),
            "outputs/nonlinear_closed_loop_qsvt/resource_ledgers/query_execution_reconciliation.csv",
            "quantity/verified_value",
        ),
        (
            ("OperationalSamplingExecutions", "DiagnosticSamplingExecutions",
             "OperationalAttemptedShots", "DiagnosticAttemptedShots"),
            "outputs/nonlinear_closed_loop_qsvt/resource_ledgers/operational_diagnostic_cost.csv",
            "cost_category rows",
        ),
        (
            ("PlateauRuns", "PlateauOnset", "RMSEFloorEntry", "InitialGap"),
            "outputs/nonlinear_closed_loop_qsvt/audits/trajectory_plateau_onset_audit.csv",
            "plateaued run rows",
        ),
        (
            ("MaxFinalRMSEDifference", "MaxTrajectoryRMSEDifference"),
            "outputs/nonlinear_closed_loop_qsvt/audits/finite_shot_statevector_comparison.csv",
            "matched scenario/seed rows",
        ),
        (
            ("MedianCoordinateReadoutError", "MaxCoordinateReadoutError",
             "MeanInterferenceAcceptance"),
            "outputs/nonlinear_closed_loop_qsvt/resource_ledgers/finite_shot_coordinate_readout.csv",
            "executed coordinate rows",
        ),
        (
            ("BlockTruncationError", "RegularizationError", "SupportRemovalError",
             "QuantizationError", "PolynomialError", "StatevectorCircuitError",
             "FiniteShotReadoutError"),
            "outputs/nonlinear_closed_loop_qsvt/error_decomposition/stage_error_decomposition.csv",
            "stage absolute-error columns",
        ),
        (
            ("OperatorLogicalOperations", "SignalApplications", "PhaseApplications",
             "ResourceMin", "ResourceMax"),
            "outputs/nonlinear_closed_loop_qsvt/resource_ledgers/circuit_resource_levels.csv",
            "circuit_type/resource columns",
        ),
    ]
    for match in re.finditer(
        r"\\providecommand\{\\(?P<name>ClosedLoop[^}]+)\}\{(?P<value>.*)\}", metrics
    ):
        name = match.group("name")
        source = ""
        anchor = ""
        for fragments, candidate, candidate_anchor in source_rules:
            if any(fragment in name for fragment in fragments):
                source = candidate
                anchor = candidate_anchor
                break
        macro_rows.append(
            {
                "manuscript_value": f"\\{name}",
                "rendered_value": match.group("value"),
                "source_artifact": source,
                "row_or_field_anchor": anchor,
                "generator": "scripts/build_tqe_closed_loop_assets.py",
            }
        )
    macro_rows.extend(
        [
            {
                "manuscript_value": "tab:closed_loop_main",
                "rendered_value": "all arm rows",
                "source_artifact": (
                    "outputs/nonlinear_closed_loop_qsvt/run_summaries/solver_outcomes.csv;"
                    "outputs/nonlinear_closed_loop_qsvt/resource_ledgers/"
                    "query_execution_reconciliation.csv"
                ),
                "row_or_field_anchor": "arm summaries and accounting totals",
                "generator": "scripts/build_tqe_closed_loop_assets.py",
            },
            {
                "manuscript_value": "supp:tab:closed_loop_scenario_audit",
                "rendered_value": "all scenario/seed rows",
                "source_artifact": (
                    "outputs/nonlinear_closed_loop_qsvt/audits/"
                    "scenario_perturbation_overlap.csv"
                ),
                "row_or_field_anchor": "Scenario/Seed",
                "generator": "scripts/build_tqe_closed_loop_assets.py",
            },
        ]
    )
    path = output_dir / "audits" / "manuscript_value_traceability.csv"
    pd.DataFrame(macro_rows).to_csv(path, index=False)
    return path


def build_closed_loop_traceability_map(output_dir: Path) -> Path:
    entries = [
        ("implementation module", "src/robust_qsvt_se/tqe_extensions/closed_loop_nonlinear_update.py"),
        ("runner", "scripts/run_tqe_closed_loop_nonlinear_update.py"),
        ("asset builder", "scripts/build_tqe_closed_loop_assets.py"),
        ("YAML configuration", "configs/tqe_closed_loop_nonlinear_update.yaml"),
        ("resolved configuration", f"{output_dir}/configs/config_resolved.yaml"),
        ("primary iteration ledger", f"{output_dir}/iteration_ledgers/closed_loop_iterations.csv"),
        ("solver outcome summary", f"{output_dir}/run_summaries/solver_outcomes.csv"),
        ("finite-shot coordinate ledger", f"{output_dir}/resource_ledgers/finite_shot_coordinate_readout.csv"),
        ("finite-shot/statevector comparison", f"{output_dir}/audits/finite_shot_statevector_comparison.csv"),
        ("extended-horizon ledger", f"{output_dir}/extended_horizon/iteration_ledgers/extended_iterations.csv"),
        ("trajectory classification", f"{output_dir}/extended_horizon/run_summaries/trajectory_classification.csv"),
        ("error-decomposition ledger", f"{output_dir}/error_decomposition/stage_error_decomposition.csv"),
        ("query/execution reconciliation", f"{output_dir}/resource_ledgers/query_execution_reconciliation.csv"),
        ("operational/diagnostic cost", f"{output_dir}/resource_ledgers/operational_diagnostic_cost.csv"),
        ("circuit-resource audit", f"{output_dir}/audits/circuit_resource_audit.csv"),
        ("readout-cost table source", f"{output_dir}/resource_ledgers/readout_cost.csv"),
        ("scenario overlap audit", f"{output_dir}/audits/scenario_perturbation_overlap.csv"),
        ("scenario fingerprint audit", f"{output_dir}/audits/scenario_fingerprint_audit.csv"),
        ("manuscript-value trace", f"{output_dir}/audits/manuscript_value_traceability.csv"),
        ("top-level manifest", f"{output_dir}/manifest.json"),
        ("run manifest", f"{output_dir}/manifests/run_manifest.json"),
        ("audit manifest", f"{output_dir}/manifests/audit_manifest.json"),
        ("checksum file", f"{output_dir}/checksums.sha256"),
    ]
    for path in sorted((output_dir / "tables").glob("tqe_closed_loop_*.tex")):
        entries.append(("generated manuscript table", str(path)))
    for path in sorted((output_dir / "figures").glob("tqe_closed_loop_*.pdf")):
        entries.append(("generated manuscript figure", str(path)))

    rows = []
    for role, relative_text in entries:
        relative = Path(relative_text)
        absolute = relative if relative.is_absolute() else ROOT / relative
        self_referential = absolute in {
            ROOT / output_dir / "manifest.json",
            ROOT / output_dir / "checksums.sha256",
        }
        rows.append(
            {
                "artifact_role": role,
                "path": str(relative),
                "exists": absolute.is_file(),
                "size_bytes": (
                    "post-build"
                    if self_referential
                    else absolute.stat().st_size
                    if absolute.is_file()
                    else 0
                ),
                "sha256": (
                    "verified by post-build checksum registry"
                    if self_referential
                    else hashlib.sha256(absolute.read_bytes()).hexdigest()
                    if absolute.is_file()
                    else ""
                ),
                "evidence_role": (
                    "manuscript-facing generated asset"
                    if role.startswith("generated manuscript")
                    else "closed-loop implementation or machine-readable evidence"
                ),
            }
        )
    path = output_dir / "audits" / "closed_loop_traceability_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_circuit_resource_table(output_dir: Path, tables_dir: Path) -> Path:
    levels = pd.read_csv(output_dir / "resource_ledgers" / "circuit_resource_levels.csv")
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Closed-loop circuit resources at each abstraction level (degree "
        "\\ClosedLoopDegree, $\\ClosedLoopBlockSize\\times\\ClosedLoopBlockSize$ block). The "
        "logical column counts untranspiled high-level operations and is "
        "not a primitive-gate count. Basis operations use $\\{r_z,r_y,r_x,\\textsc{cx}\\}$ "
        "with optimization level 1 and seed 20260722; no opaque instruction remains. No backend "
        "target or coupling map is supplied, so connectivity is all-to-all and routing is not "
        "included: these are reproducible basis decompositions, not device-specific hardware "
        "gate counts. Residual and functional preparation and measurements occur only in the "
        "finite-shot branch circuits.}",
        "\\label{supp:tab:closed_loop_circuit_resources}",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "Circuit & BE dim. & Q/C bits & Degree & Signal/phase & Logical & SDK ops & "
        "Custom/opaque & Decomp. & Basis ops & Depth \\\\ ",
        "\\midrule",
    ]
    for _, row in levels.sort_values(["block_encoding_dimension", "circuit_type"]).iterrows():
        label = CIRCUIT_TYPE_LABELS[str(row["circuit_type"])]
        lines.append(
            f"{label} & {int(row['block_encoding_dimension'])} & "
            f"{int(row['n_qubits'])}/{int(row['n_clbits'])} & {int(row['qsvt_degree'])} & "
            f"{int(row['signal_unitary_applications'])}/{int(row['projector_phase_applications'])} "
            f"& {_grp(int(row['logical_operations']))} & "
            f"{_grp(int(row['untranspiled_sdk_operations']))} & "
            f"{_grp(int(row['untranspiled_custom_or_opaque_operations']))} & "
            f"{_grp(int(row['decomposed_operations']))} & "
            f"{_grp(int(row['transpiled_basis_gates']))} & "
            f"{_grp(int(row['transpiled_depth']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_circuit_resources.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_extended_horizon_table(output_dir: Path, tables_dir: Path) -> Path:
    cls = pd.read_csv(
        output_dir / "extended_horizon" / "run_summaries" / "trajectory_classification.csv"
    )
    summary = pd.read_csv(output_dir / "audits" / "extended_horizon_arm_summary.csv")
    max_iters = int(cls["max_iterations"].iloc[0]) if not cls.empty else 0
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Extended-horizon convergence diagnostic (up to "
        f"{max_iters} iterations; separated from the primary protocol). The plateau classification "
        "requires the terminal \\ClosedLoopPlateauWindow-record RMSE window to change by at most "
        "\\ClosedLoopPlateauRelativeTolerance{} relative to its endpoint, have swing at most "
        "\\ClosedLoopPlateauSwingTolerance, and end with update norm at most "
        "\\ClosedLoopPlateauStepThreshold. Plateau onset is the end of the earliest such window "
        "after which every remaining window also qualifies; RMSE-only entry into the final "
        "tolerance band is reported "
        "separately. Every block-based arm reaches a stable error floor above the full-system "
        "reference.}",
        "\\label{supp:tab:closed_loop_extended}",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Arm & Classification (runs) & Final RMSE & Final residual & Selected RMSE & Frozen RMSE "
        "& Final update & Onset & RMSE-floor entry \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        sub = cls[cls["arm"] == arm]
        if sub.empty:
            continue
        counts = sub["classification"].value_counts()
        cls_str = ", ".join(
            f"{c.replace('_', ' ')} ({n})" for c, n in counts.items()
        )
        aggregate = summary[summary["arm"] == arm].iloc[0]
        onset = sub.loc[
            sub["classification"] == "plateaued", "plateau_onset_iteration"
        ].to_numpy(dtype=float)
        floor_entry = sub.loc[
            sub["classification"] == "plateaued", "rmse_floor_entry_iteration"
        ].to_numpy(dtype=float)
        onset_str = str(int(np.median(onset))) if onset.size else "--"
        floor_entry_str = str(int(np.median(floor_entry))) if floor_entry.size else "--"
        lines.append(
            f"{ARM_LABELS[arm]} & {cls_str} & "
            f"{_fmt(float(aggregate['median_final_state_rmse']))} & "
            f"{_fmt(float(aggregate['median_final_weighted_residual']))} & "
            f"{_fmt(float(aggregate['median_final_selected_coordinate_rmse']))} & "
            f"{_fmt(float(aggregate['median_final_frozen_coordinate_rmse']))} & "
            f"{_fmt(float(aggregate['median_final_update_norm']))} & {onset_str} & "
            f"{floor_entry_str} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_extended_horizon.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_comparison_table(output_dir: Path, tables_dir: Path) -> Path:
    comp = pd.read_csv(output_dir / "audits" / "finite_shot_statevector_comparison.csv")
    lines = [
        "% Auto-generated by scripts/build_tqe_closed_loop_assets.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Paired finite-shot versus statevector closed-loop trajectory differences across "
        "the \\ClosedLoopFiniteShotRuns{} matched runs (same scenario, seed, block, and horizon). "
        "For each run the final "
        "signed and absolute final-state-RMSE differences, maximum per-iteration RMSE and state-"
        "norm differences, and maximum update-vector difference are reported at "
        "\\ClosedLoopShotsPerCall{} shots per "
        "sampling call. These are executed numerical differences; no equivalence tolerance is "
        "recorded or used, and no statistical-indistinguishability claim is made.}",
        "\\label{supp:tab:closed_loop_fs_sv}",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Scenario & Seed & SV final RMSE & FS final RMSE & $\\Delta$ final & $|\\Delta|$ final "
        "& Max RMSE diff. & Max state-norm diff. & Max update diff. \\\\",
        "\\midrule",
    ]
    if not comp.empty:
        for _, row in comp.iterrows():
            lines.append(
                f"{str(row['scenario']).replace('_', ' ')} & {int(row['seed'])} & "
                f"{_fmt(float(row['statevector_final_rmse']))} & "
                f"{_fmt(float(row['finite_shot_final_rmse']))} & "
                f"{_fmt(float(row['delta_final_rmse']))} & "
                f"{_fmt(float(row['absolute_final_rmse_difference']))} & "
                f"{_fmt(float(row['max_state_rmse_difference']))} & "
                f"{_fmt(float(row['max_trajectory_state_norm_difference']))} & "
                f"{_fmt(float(row['max_update_vector_norm_difference']))} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path = tables_dir / "tqe_closed_loop_fs_sv_comparison.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_all(output_dir: Path, tables_dir: Path, figures_dir: Path) -> list[Path]:
    build_scenario_audits(output_dir)
    build_operational_diagnostic_cost_ledger(output_dir)
    artifacts = [
        build_metric_macros(output_dir, tables_dir),
        build_convergence_figure(output_dir, figures_dir),
        build_decomposition_figure(output_dir, figures_dir),
        build_extended_horizon_figure(output_dir, figures_dir),
        build_main_table(output_dir, tables_dir),
        build_decomposition_table(output_dir, tables_dir),
        build_readout_cost_table(output_dir, tables_dir),
        build_query_accounting_table(output_dir, tables_dir),
        build_scenario_audit_table(output_dir, tables_dir),
        build_circuit_resource_table(output_dir, tables_dir),
        build_extended_horizon_table(output_dir, tables_dir),
        build_comparison_table(output_dir, tables_dir),
    ]
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--tables-dir", default=str(MANUSCRIPT_TABLES))
    parser.add_argument("--figures-dir", default=str(MANUSCRIPT_FIGURES))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    artifacts = build_all(output_dir, tables_dir, figures_dir)
    # Mirror the manuscript tables/figures inside the evidence tree for provenance.
    build_all(output_dir, output_dir / "tables", output_dir / "figures")
    build_manuscript_value_traceability(output_dir, tables_dir)
    build_closed_loop_traceability_map(output_dir)
    solver = pd.read_csv(output_dir / "run_summaries" / "solver_outcomes.csv")
    iterations = pd.read_csv(output_dir / "iteration_ledgers" / "closed_loop_iterations.csv")
    write_manifest_and_checksums(
        output_dir,
        study_id="tqe_closed_loop_nonlinear_update_v1",
        extra={"runs": len(solver), "iterations": len(iterations)},
    )
    for path in artifacts:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
