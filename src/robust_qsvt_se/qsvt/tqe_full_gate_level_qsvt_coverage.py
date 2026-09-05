from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    FULL_GATE_LEVEL_COVERAGE_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import load_sweep_subproblem
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import (
    fit_actual_singular_interpolating_polynomial,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import construct_padded_block_encoding
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    DEFAULT_BASIS_GATES,
    IntegratedEvaluation,
    PhaseSynthesisResult,
    evaluate_qsvt_transform,
    run_ieee_selected_block,
    synthesize_qsvt_phases,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL_TOL = 1.0e-14

COVERAGE_COLUMNS = [
    "tier",
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "alpha",
    "epsilon_target",
    "degree",
    "degree_source",
    "phase_count",
    "phase_synthesis_status",
    "qsvt_circuit_status",
    "simulation_status",
    "transpilation_status",
    "num_qubits",
    "num_U_calls",
    "num_U_dagger_calls",
    "num_phase_rotations",
    "raw_depth",
    "transpiled_depth",
    "transpiled_cx_count",
    "transpiled_total_ops",
    "transform_block_fro_error",
    "transform_block_spectral_error",
    "circuit_vs_polynomial_fro_error",
    "circuit_vs_ridge_relative_update_error",
    "absolute_update_error",
    "residual_gap",
    "success_probability",
    "runtime_seconds",
    "failure_or_skip_reason",
]

SUMMARY_COLUMNS = [
    "tier",
    "case_name",
    "subproblem_size",
    "degree",
    "phase_synthesis_status",
    "qsvt_circuit_status",
    "simulation_status",
    "transpilation_status",
    "transform_block_fro_error",
    "circuit_vs_ridge_relative_update_error",
    "residual_gap",
    "success_probability",
    "raw_depth",
    "transpiled_depth",
    "transpiled_cx_count",
]


@dataclass(frozen=True, slots=True)
class DegreeSelection:
    degree: int
    source: str
    target_met: bool
    reason: str


def run_full_gate_level_qsvt_coverage(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / FULL_GATE_LEVEL_COVERAGE_DIR)
    figures_dir = paths["figures"]
    tables_dir = paths["tables"]
    reports_dir = paths["reports"]

    rows = [evaluate_coverage_case(case_spec, resolved) for case_spec in resolved["case_specs"]]
    results = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    summary = summarize_coverage_results(results)

    artifacts = _write_outputs(
        config=resolved,
        output_dir=output_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
        results=results,
        summary=summary,
        started_at=started_at,
    )
    metadata_path = artifacts["metadata_json"]
    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts={key: str(value) for key, value in artifacts.items()},
    )
    metadata.update(
        {
            "status_counts": coverage_status_counts(results),
            "attempted_cases": len(results),
            "phase_synthesis_successes": int(
                (results["phase_synthesis_status"] == "completed").sum()
            ),
            "circuit_construction_successes": int(
                results["qsvt_circuit_status"].isin(["completed", "circuit_object_built"]).sum()
            ),
            "simulations_completed": int((results["simulation_status"] == "completed").sum()),
            "transpiled_circuits": int((results["transpilation_status"] == "completed").sum()),
            "claim_boundary": (
                "Expanded selected-subproblem gate-level audit under budget; dense "
                "proof-of-concept circuits only."
            ),
        }
    )
    write_json(metadata_path, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: path for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results,
        "summary": summary,
        "artifacts": artifacts,
    }


def evaluate_coverage_case(case_spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    spec = _normalized_case_spec(case_spec)
    if bool(spec.get("skip_by_budget", False)):
        degree_selection = select_degree_from_previous_sweep(
            summary_path=Path(config["degree_summary_path"]),
            results_path=Path(config["degree_results_path"]),
            case_name=str(spec["case_name"]),
            subproblem_size=int(spec["subproblem_size"]),
            selection_criterion=str(spec["selection_mode"]),
            alpha=float(config["alpha"]),
            epsilon_target=float(config["epsilon_target"]),
            fallback_degree=int(config["fallback_degree"]),
        )
        return skipped_by_budget_row(
            case_spec=spec,
            config=config,
            degree_selection=degree_selection,
            runtime_seconds=time.perf_counter() - started,
            reason=str(spec.get("skip_reason", "skipped by configured computational budget")),
        )

    degree_selection = select_degree_from_previous_sweep(
        summary_path=Path(config["degree_summary_path"]),
        results_path=Path(config["degree_results_path"]),
        case_name=str(spec["case_name"]),
        subproblem_size=int(spec["subproblem_size"]),
        selection_criterion=str(spec["selection_mode"]),
        alpha=float(config["alpha"]),
        epsilon_target=float(config["epsilon_target"]),
        fallback_degree=int(config["fallback_degree"]),
    )
    try:
        if bool(spec.get("circuit_object_only", False)):
            row = build_circuit_object_only_row(
                case_spec=spec,
                config=config,
                degree_selection=degree_selection,
            )
        else:
            evaluation = run_ieee_selected_block(
                _integrated_case_config(
                    config=config,
                    spec=spec,
                    degree=degree_selection.degree,
                )
            )
            row = coverage_row_from_integrated_evaluation(
                evaluation=evaluation,
                tier=str(spec["tier"]),
                selection_criterion=str(spec["selection_mode"]),
                degree_selection=degree_selection,
            )
    except Exception as exc:
        row = failure_row(
            case_spec=spec,
            config=config,
            degree_selection=degree_selection,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    row["runtime_seconds"] = float(time.perf_counter() - started)
    if degree_selection.reason and not str(row["failure_or_skip_reason"]):
        row["failure_or_skip_reason"] = degree_selection.reason
    return row


def coverage_row_from_integrated_evaluation(
    *,
    evaluation: IntegratedEvaluation,
    tier: str,
    selection_criterion: str,
    degree_selection: DegreeSelection,
) -> dict[str, Any]:
    row = evaluation.row
    out = empty_coverage_row(
        tier=tier,
        case_name=str(row.get("case_name", "unknown")),
        subproblem_size=int(row.get("subproblem_size", 0)),
        selection_criterion=selection_criterion,
    )
    out.update(
        {
            "alpha": row.get("alpha", np.nan),
            "epsilon_target": row.get("epsilon_target", np.nan),
            "degree": int(row.get("degree", degree_selection.degree)),
            "degree_source": degree_selection.source,
            "phase_count": int(row.get("phase_count", 0)),
            "phase_synthesis_status": row.get("phase_synthesis_status", "not_completed"),
            "qsvt_circuit_status": row.get("qsvt_sequence_status", "not_completed"),
            "simulation_status": row.get("simulation_status", "not_completed"),
            "transpilation_status": row.get("transpilation_status", "not_attempted"),
            "num_qubits": row.get("num_qubits", np.nan),
            "num_U_calls": row.get("num_U_calls", 0),
            "num_U_dagger_calls": row.get("num_U_dagger_calls", 0),
            "num_phase_rotations": row.get("num_phase_rotations", 0),
            "raw_depth": row.get("raw_circuit_depth", np.nan),
            "transpiled_depth": row.get("transpiled_depth", np.nan),
            "transpiled_cx_count": row.get("transpiled_cx_count", np.nan),
            "transpiled_total_ops": row.get("transpiled_total_ops", np.nan),
            "transform_block_fro_error": row.get("transform_block_error_fro", np.nan),
            "transform_block_spectral_error": row.get("transform_block_error_spectral", np.nan),
            "circuit_vs_polynomial_fro_error": row.get(
                "circuit_vs_polynomial_fro_error",
                np.nan,
            ),
            "circuit_vs_ridge_relative_update_error": row.get("relative_update_error", np.nan),
            "absolute_update_error": row.get("absolute_update_error", np.nan),
            "residual_gap": row.get("residual_gap", np.nan),
            "success_probability": row.get("success_probability_residual_state", np.nan),
            "failure_or_skip_reason": row.get("failure_or_skip_reason", ""),
        }
    )
    return out


def build_circuit_object_only_row(
    *,
    case_spec: dict[str, Any],
    config: dict[str, Any],
    degree_selection: DegreeSelection,
) -> dict[str, Any]:
    subproblem = load_sweep_subproblem(
        {
            "case_name": str(case_spec["case_name"]),
            "subproblem_size": int(case_spec["subproblem_size"]),
            "selection_mode": str(case_spec["selection_mode"]),
        },
        seed=int(config["seed"]),
    )
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(A, compute_uv=False)
    gamma = float(singular_values[0]) if singular_values.size else 1.0
    encoding = construct_padded_block_encoding(A, gamma=gamma)
    cheb, _C_alpha = fit_actual_singular_interpolating_polynomial(
        alpha=float(config["alpha"]),
        gamma=gamma,
        singular_values=singular_values,
        degree=int(degree_selection.degree),
    )
    coefficients = _pad_odd_coefficients(
        cheb.convert(kind=Polynomial).coef,
        int(degree_selection.degree),
    )
    phase_result = synthesize_qsvt_phases(
        coefficients,
        angle_solver=str(config["angle_solver"]),
    )
    if phase_result.status != "completed":
        return phase_failure_row(
            case_spec=case_spec,
            config=config,
            degree_selection=degree_selection,
            phase_result=phase_result,
            reason=phase_result.failure_reason,
        )
    bundle = build_structured_qsvt_operator_circuit(
        np.asarray(encoding.U, dtype=np.complex128),
        np.asarray(phase_result.phases, dtype=np.float64),
        encoded_dimension=int(encoding.A_bar_padded.shape[0]),
    )
    calls = qsvt_call_counts(len(phase_result.phases))
    row = empty_coverage_row(
        tier=str(case_spec["tier"]),
        case_name=str(case_spec["case_name"]),
        subproblem_size=int(case_spec["subproblem_size"]),
        selection_criterion=str(case_spec["selection_mode"]),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(degree_selection.degree),
            "degree_source": degree_selection.source,
            "phase_count": int(phase_result.phases.size),
            "phase_synthesis_status": phase_result.status,
            "qsvt_circuit_status": "circuit_object_built",
            "simulation_status": "skipped_by_budget",
            "transpilation_status": "skipped_by_budget",
            "num_qubits": int(bundle.n_qubits),
            "num_U_calls": calls["num_U_calls"],
            "num_U_dagger_calls": calls["num_U_dagger_calls"],
            "num_phase_rotations": int(phase_result.phases.size),
            "raw_depth": int(bundle.qsvt_operator_circuit.depth()),
            "failure_or_skip_reason": (
                "Tier 3 circuit-object-only audit: Operator/Statevector simulation and "
                "basis-gate transpilation skipped by configured budget."
            ),
        }
    )
    return row


def evaluate_tiny_mock_gate_case() -> dict[str, Any]:
    matrix = np.diag([0.2, 0.5]).astype(np.float64)
    residual = np.array([1.0, -0.5], dtype=np.float64)
    encoding = construct_padded_block_encoding(matrix, gamma=1.0)
    polynomial = Polynomial([0.0, 0.5])
    phase_result = synthesize_qsvt_phases(
        np.array([0.0, 0.5], dtype=np.float64),
        angle_solver="root-finding",
    )
    evaluation = evaluate_qsvt_transform(
        run_type="tiny_mock",
        case_name="tiny_mock",
        subproblem_size=2,
        A=matrix,
        b=residual,
        A_bar_padded=encoding.A_bar_padded,
        U_A=encoding.U,
        gamma=1.0,
        C_alpha=1.0,
        alpha=1.0e-2,
        epsilon_target=1.0e-2,
        degree=1,
        polynomial=polynomial,
        phases=phase_result.phases,
        phase_result=phase_result,
        basis_gates=DEFAULT_BASIS_GATES,
        transpile_qubit_limit=3,
        transpile_optimization_level=0,
    )
    return coverage_row_from_integrated_evaluation(
        evaluation=evaluation,
        tier="unit_test",
        selection_criterion="mock",
        degree_selection=DegreeSelection(
            degree=1,
            source="unit_test_known_polynomial",
            target_met=True,
            reason="",
        ),
    )


def select_degree_from_previous_sweep(
    *,
    summary_path: Path,
    results_path: Path,
    case_name: str,
    subproblem_size: int,
    selection_criterion: str,
    alpha: float,
    epsilon_target: float,
    fallback_degree: int,
) -> DegreeSelection:
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        matches = _degree_summary_matches(
            summary,
            case_name=case_name,
            subproblem_size=subproblem_size,
            selection_criterion=selection_criterion,
            alpha=alpha,
            epsilon_target=epsilon_target,
        )
        if not matches.empty:
            row = matches.sort_values(
                ["required_degree", "best_available_degree"],
                na_position="last",
            ).iloc[0]
            required = pd.to_numeric(pd.Series([row.get("required_degree")]), errors="coerce").iloc[
                0
            ]
            if np.isfinite(required):
                return DegreeSelection(
                    degree=int(required),
                    source="degree_alpha_precision_summary_required_degree",
                    target_met=True,
                    reason="",
                )
            best = pd.to_numeric(
                pd.Series([row.get("best_available_degree")]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(best):
                return DegreeSelection(
                    degree=int(best),
                    source="degree_alpha_precision_summary_best_available_degree",
                    target_met=False,
                    reason="previous sweep did not meet target; using best available degree",
                )
    if results_path.exists():
        results = pd.read_csv(results_path)
        matches = _degree_result_matches(
            results,
            case_name=case_name,
            subproblem_size=subproblem_size,
            selection_criterion=selection_criterion,
            alpha=alpha,
            epsilon_target=epsilon_target,
        )
        if not matches.empty:
            successful = matches[matches["meets_epsilon_on_actual_singular_values"].astype(bool)]
            if not successful.empty:
                return DegreeSelection(
                    degree=int(successful.sort_values("degree").iloc[0]["degree"]),
                    source="degree_alpha_precision_results_smallest_meeting_degree",
                    target_met=True,
                    reason="",
                )
            best = matches.sort_values("max_approximation_error_on_actual_singular_values").iloc[0]
            return DegreeSelection(
                degree=int(best["degree"]),
                source="degree_alpha_precision_results_best_available_degree",
                target_met=False,
                reason="previous sweep did not meet target; using best available degree",
            )
    return DegreeSelection(
        degree=int(fallback_degree),
        source="configured_fallback_degree",
        target_met=False,
        reason=(
            "no matching previous degree-alpha-precision sweep row found; using configured "
            f"fallback degree {int(fallback_degree)}"
        ),
    )


def _degree_summary_matches(
    frame: pd.DataFrame,
    *,
    case_name: str,
    subproblem_size: int,
    selection_criterion: str,
    alpha: float,
    epsilon_target: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[
        (frame["case_name"] == str(case_name))
        & (frame["subproblem_size"].astype(int) == int(subproblem_size))
        & (frame["selection_criterion"] == str(selection_criterion))
        & np.isclose(frame["alpha"].astype(float), float(alpha))
        & np.isclose(frame["epsilon_target"].astype(float), float(epsilon_target))
    ]


def _degree_result_matches(
    frame: pd.DataFrame,
    *,
    case_name: str,
    subproblem_size: int,
    selection_criterion: str,
    alpha: float,
    epsilon_target: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[
        (frame["case_name"] == str(case_name))
        & (frame["subproblem_size"].astype(int) == int(subproblem_size))
        & (frame["selection_criterion"] == str(selection_criterion))
        & np.isclose(frame["alpha"].astype(float), float(alpha))
        & np.isclose(frame["epsilon_target"].astype(float), float(epsilon_target))
    ]


def ridge_update_comparison_metrics(
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
) -> dict[str, float]:
    ridge = np.asarray(ridge_update, dtype=np.float64)
    qsvt = np.asarray(qsvt_update, dtype=np.float64)
    delta = qsvt - ridge
    absolute = float(np.linalg.norm(delta))
    return {
        "absolute_update_error": absolute,
        "relative_update_error": absolute / max(float(np.linalg.norm(ridge)), SMALL_TOL),
        "max_component_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def phase_failure_row(
    *,
    case_spec: dict[str, Any],
    config: dict[str, Any],
    degree_selection: DegreeSelection,
    phase_result: PhaseSynthesisResult,
    reason: str,
) -> dict[str, Any]:
    row = empty_coverage_row(
        tier=str(case_spec["tier"]),
        case_name=str(case_spec["case_name"]),
        subproblem_size=int(case_spec["subproblem_size"]),
        selection_criterion=str(case_spec["selection_mode"]),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(degree_selection.degree),
            "degree_source": degree_selection.source,
            "phase_count": int(phase_result.phases.size),
            "phase_synthesis_status": phase_result.status,
            "qsvt_circuit_status": "skipped_phase_synthesis_failed",
            "simulation_status": "skipped_phase_synthesis_failed",
            "transpilation_status": "not_attempted",
            "failure_or_skip_reason": reason,
        }
    )
    return row


def skipped_by_budget_row(
    *,
    case_spec: dict[str, Any],
    config: dict[str, Any],
    degree_selection: DegreeSelection,
    runtime_seconds: float,
    reason: str,
) -> dict[str, Any]:
    row = empty_coverage_row(
        tier=str(case_spec["tier"]),
        case_name=str(case_spec["case_name"]),
        subproblem_size=int(case_spec["subproblem_size"]),
        selection_criterion=str(case_spec["selection_mode"]),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(degree_selection.degree),
            "degree_source": degree_selection.source,
            "phase_synthesis_status": "not_attempted",
            "qsvt_circuit_status": "skipped_by_budget",
            "simulation_status": "skipped_by_budget",
            "transpilation_status": "skipped_by_budget",
            "runtime_seconds": float(runtime_seconds),
            "failure_or_skip_reason": reason,
        }
    )
    return row


def failure_row(
    *,
    case_spec: dict[str, Any],
    config: dict[str, Any],
    degree_selection: DegreeSelection,
    status: str,
    reason: str,
) -> dict[str, Any]:
    row = empty_coverage_row(
        tier=str(case_spec["tier"]),
        case_name=str(case_spec["case_name"]),
        subproblem_size=int(case_spec["subproblem_size"]),
        selection_criterion=str(case_spec["selection_mode"]),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(degree_selection.degree),
            "degree_source": degree_selection.source,
            "phase_synthesis_status": "not_completed",
            "qsvt_circuit_status": status,
            "simulation_status": status,
            "transpilation_status": "not_attempted",
            "failure_or_skip_reason": reason,
        }
    )
    return row


def empty_coverage_row(
    *,
    tier: str,
    case_name: str,
    subproblem_size: int,
    selection_criterion: str,
) -> dict[str, Any]:
    row = {column: np.nan for column in COVERAGE_COLUMNS}
    row.update(
        {
            "tier": tier,
            "case_name": case_name,
            "subproblem_size": int(subproblem_size),
            "selection_criterion": selection_criterion,
            "degree": np.nan,
            "degree_source": "",
            "phase_count": 0,
            "phase_synthesis_status": "not_attempted",
            "qsvt_circuit_status": "not_attempted",
            "simulation_status": "not_attempted",
            "transpilation_status": "not_attempted",
            "num_U_calls": 0,
            "num_U_dagger_calls": 0,
            "num_phase_rotations": 0,
            "runtime_seconds": 0.0,
            "failure_or_skip_reason": "",
        }
    )
    return row


def qsvt_call_counts(phase_count: int) -> dict[str, int]:
    u_calls = 0
    u_dagger_calls = 0
    for _index in range(1, int(phase_count) - 1, 2):
        u_calls += 1
        u_dagger_calls += 1
    if int(phase_count) % 2 == 0:
        u_calls += 1
    return {"num_U_calls": u_calls, "num_U_dagger_calls": u_dagger_calls}


def summarize_coverage_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return results[SUMMARY_COLUMNS].copy()


def coverage_status_counts(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "phase_synthesis_status": _value_counts(results, "phase_synthesis_status"),
        "qsvt_circuit_status": _value_counts(results, "qsvt_circuit_status"),
        "simulation_status": _value_counts(results, "simulation_status"),
        "transpilation_status": _value_counts(results, "transpilation_status"),
    }


def _write_outputs(
    *,
    config: dict[str, Any],
    output_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    reports_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    started_at: str,
) -> dict[str, Path]:
    artifacts = {
        "results_csv": output_dir / "full_gate_level_qsvt_coverage_results.csv",
        "metadata_json": output_dir / "full_gate_level_qsvt_coverage_metadata.json",
        "summary_table_csv": tables_dir / "table_full_gate_level_qsvt_coverage_summary.csv",
        "errors_figure": figures_dir / "figure_full_gate_level_qsvt_errors.png",
        "depth_cx_figure": figures_dir / "figure_full_gate_level_qsvt_depth_cx.png",
        "success_probability_figure": figures_dir
        / "figure_full_gate_level_qsvt_success_probability.png",
        "report": reports_dir / "full_gate_level_qsvt_coverage_report.md",
    }
    results.to_csv(artifacts["results_csv"], index=False)
    summary.to_csv(artifacts["summary_table_csv"], index=False)
    _plot_errors(results, artifacts["errors_figure"])
    _plot_depth_cx(results, artifacts["depth_cx_figure"])
    _plot_success_probability(results, artifacts["success_probability_figure"])
    artifacts["report"].write_text(
        _report_markdown(
            config=config,
            results=results,
            summary=summary,
            started_at=started_at,
            artifacts=artifacts,
        ),
        encoding="utf-8",
    )
    return artifacts


def _plot_errors(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    labels = _labels(frame)
    x = np.arange(len(labels))
    if frame.empty:
        plt.text(0.5, 0.5, "No coverage rows", ha="center", va="center")
    else:
        width = 0.25
        plt.bar(
            x - width,
            _positive_for_log(frame["transform_block_fro_error"]),
            width,
            label="transform Frobenius",
        )
        plt.bar(
            x,
            _positive_for_log(frame["circuit_vs_polynomial_fro_error"]),
            width,
            label="circuit vs polynomial",
        )
        plt.bar(
            x + width,
            _positive_for_log(frame["circuit_vs_ridge_relative_update_error"]),
            width,
            label="relative update error",
        )
        plt.yscale("log")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("error")
        plt.legend()
    plt.title("Full Gate-Level QSVT Coverage Errors")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_depth_cx(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    labels = _labels(frame)
    x = np.arange(len(labels))
    if frame.empty:
        plt.text(0.5, 0.5, "No coverage rows", ha="center", va="center")
    else:
        width = 0.28
        plt.bar(
            x - width,
            pd.to_numeric(frame["raw_depth"], errors="coerce").fillna(0),
            width,
            label="raw depth",
        )
        plt.bar(
            x,
            pd.to_numeric(frame["transpiled_depth"], errors="coerce").fillna(0),
            width,
            label="transpiled depth",
        )
        plt.bar(
            x + width,
            pd.to_numeric(frame["transpiled_cx_count"], errors="coerce").fillna(0),
            width,
            label="CX count",
        )
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("count")
        plt.legend()
    plt.title("Full Gate-Level QSVT Depth and CX Counts")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_success_probability(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    labels = _labels(frame)
    x = np.arange(len(labels))
    if frame.empty:
        plt.text(0.5, 0.5, "No coverage rows", ha="center", va="center")
    else:
        plt.bar(x, pd.to_numeric(frame["success_probability"], errors="coerce").fillna(0.0))
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("success probability")
        plt.ylim(bottom=0)
    plt.title("Full Gate-Level QSVT Residual-State Success Probability")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _report_markdown(
    *,
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    started_at: str,
    artifacts: dict[str, Path],
) -> str:
    counts = coverage_status_counts(results)
    completed = (
        results[results["simulation_status"] == "completed"] if not results.empty else results
    )
    transpiled = (
        results[results["transpilation_status"] == "completed"] if not results.empty else results
    )
    skipped = (
        results[
            results["simulation_status"].astype(str).str.contains("skipped", na=False)
            | results["transpilation_status"].astype(str).str.contains("skipped", na=False)
        ]
        if not results.empty
        else results
    )
    return "\n".join(
        [
            "# Full Gate-Level QSVT Coverage Audit",
            "",
            "## Goal",
            "",
            "This audit expands full gate-level QSVT validation across selected small "
            "IEEE-derived subproblems under a fixed computational budget.",
            "",
            "## Configuration",
            "",
            f"- Command: `{current_command()}`",
            f"- Started at: `{started_at}`",
            f"- Alpha: `{config['alpha']}`",
            f"- Epsilon target: `{config['epsilon_target']}`",
            f"- Basis gates: `{config['basis_gates']}`",
            f"- Transpile qubit limit: `{config['transpile_qubit_limit']}`",
            f"- Case count: `{len(config['case_specs'])}`",
            "",
            "## Status",
            "",
            f"- Attempted rows: {len(results)}",
            "- Phase synthesis successes: "
            f"{_count_status(results, 'phase_synthesis_status', 'completed')}",
            f"- Circuit construction successes: {_count_circuit_successes(results)}",
            f"- Simulations completed: {_count_status(results, 'simulation_status', 'completed')}",
            f"- Transpiled circuits: {_count_status(results, 'transpilation_status', 'completed')}",
            f"- Status counts: `{counts}`",
            "",
            "## Key Numerical Findings",
            "",
            *_metric_lines(completed, transpiled),
            "",
            "## Skipped or Budget-Limited Rows",
            "",
            *_skipped_lines(skipped),
            "",
            "## Claim-Safe Interpretation",
            "",
            "This audit expands full gate-level QSVT validation across selected small "
            "IEEE-derived subproblems. Successful rows provide circuit-level "
            "consistency evidence; skipped rows identify resource boundaries of dense "
            "proof-of-concept circuits. The audit does not demonstrate full "
            "IEEE-scale QSVT execution, scalable sparse-oracle implementation, "
            "hardware execution, or quantum speedup.",
            "",
            "## Limitations",
            "",
            "- Dense block-encoding unitaries are loaded or constructed explicitly.",
            "- Larger selected blocks are simulated or transpiled only within the "
            "configured budget.",
            "- This is selected-subproblem evidence and does not imply QSVT-over-Ridge "
            "superiority.",
            "",
            "## Recommended Manuscript Placement",
            "",
            "Use the summary counts and one compact error/resource figure in the supplement. "
            "The main manuscript can cite this as expanded selected-subproblem "
            "gate-level audit evidence if space permits.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{artifacts['results_csv']}`",
            f"- Summary table: `{artifacts['summary_table_csv']}`",
            f"- Error figure: `{artifacts['errors_figure']}`",
            f"- Depth/CX figure: `{artifacts['depth_cx_figure']}`",
            f"- Success probability figure: `{artifacts['success_probability_figure']}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _metric_lines(completed: pd.DataFrame, transpiled: pd.DataFrame) -> list[str]:
    if completed.empty:
        return ["- No completed simulation rows."]
    lines = [
        "- Transform Frobenius error range: "
        f"{completed['transform_block_fro_error'].min():.3e} to "
        f"{completed['transform_block_fro_error'].max():.3e}.",
        "- Circuit-vs-polynomial Frobenius error range: "
        f"{completed['circuit_vs_polynomial_fro_error'].min():.3e} to "
        f"{completed['circuit_vs_polynomial_fro_error'].max():.3e}.",
        "- Circuit-vs-Ridge relative update error range: "
        f"{completed['circuit_vs_ridge_relative_update_error'].min():.3e} to "
        f"{completed['circuit_vs_ridge_relative_update_error'].max():.3e}.",
        "- Residual gap range: "
        f"{completed['residual_gap'].min():.3e} to {completed['residual_gap'].max():.3e}.",
        "- Success probability range: "
        f"{completed['success_probability'].min():.3e} to "
        f"{completed['success_probability'].max():.3e}.",
        "- Raw depth range: "
        f"{int(completed['raw_depth'].min())} to {int(completed['raw_depth'].max())}.",
    ]
    if not transpiled.empty:
        lines.append(
            "- Transpiled depth/CX ranges: "
            f"depth {int(transpiled['transpiled_depth'].min())}-"
            f"{int(transpiled['transpiled_depth'].max())}, "
            f"CX {int(transpiled['transpiled_cx_count'].min())}-"
            f"{int(transpiled['transpiled_cx_count'].max())}."
        )
    return lines


def _skipped_lines(skipped: pd.DataFrame) -> list[str]:
    if skipped.empty:
        return ["- No skipped rows."]
    lines = []
    for row in skipped.itertuples(index=False):
        lines.append(
            f"- {row.tier} {row.case_name} {int(row.subproblem_size)}x"
            f"{int(row.subproblem_size)}: simulation={row.simulation_status}, "
            f"transpilation={row.transpilation_status}; reason={row.failure_or_skip_reason}"
        )
    return lines


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 123,
        "alpha": 1.0e-2,
        "epsilon_target": 1.0e-2,
        "fallback_degree": 5,
        "angle_solver": "root-finding",
        "basis_gates": DEFAULT_BASIS_GATES,
        "transpile_qubit_limit": 3,
        "transpile_optimization_level": 1,
        "artifact_match_rtol": 1.0e-9,
        "artifact_match_atol": 1.0e-8,
        "degree_summary_path": str(root / "tables" / "table_degree_alpha_precision_summary.csv"),
        "degree_results_path": str(
            root / "degree_alpha_precision_sweep" / "degree_alpha_precision_sweep_results.csv"
        ),
        "end_to_end_results_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_results.csv"
        ),
        "block_results_path": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "block_matrices_dir": str(root / "explicit_block_encoding_demo" / "matrices"),
        "case_specs": _default_case_specs(),
    }
    if config:
        resolved.update(config)
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    resolved["case_specs"] = [_normalized_case_spec(value) for value in resolved["case_specs"]]
    return resolved


def _default_case_specs() -> list[dict[str, Any]]:
    return [
        {"tier": "tier1_required", "case_name": "ieee14", "subproblem_size": 4},
        {"tier": "tier1_required", "case_name": "ieee30", "subproblem_size": 4},
        {"tier": "tier1_required", "case_name": "ieee57", "subproblem_size": 4},
        {"tier": "tier1_required", "case_name": "ieee118", "subproblem_size": 4},
        {"tier": "tier2_attempted", "case_name": "ieee14", "subproblem_size": 8},
        {"tier": "tier2_attempted", "case_name": "ieee57", "subproblem_size": 8},
        {
            "tier": "tier3_optional",
            "case_name": "ieee57",
            "subproblem_size": 16,
            "circuit_object_only": True,
        },
    ]


def _normalized_case_spec(case_spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(case_spec)
    spec.setdefault("tier", "unspecified")
    spec.setdefault("selection_mode", "high_leverage")
    spec.setdefault("circuit_object_only", False)
    spec.setdefault("skip_by_budget", False)
    spec["case_name"] = str(spec["case_name"])
    spec["subproblem_size"] = int(spec["subproblem_size"])
    spec["selection_mode"] = str(spec["selection_mode"])
    return spec


def _integrated_case_config(
    *,
    config: dict[str, Any],
    spec: dict[str, Any],
    degree: int,
) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]),
        "subproblem_spec": {
            "case_name": str(spec["case_name"]),
            "subproblem_size": int(spec["subproblem_size"]),
            "selection_mode": str(spec["selection_mode"]),
        },
        "alpha": float(config["alpha"]),
        "epsilon_target": float(config["epsilon_target"]),
        "degree": int(degree),
        "angle_solver": str(config["angle_solver"]),
        "basis_gates": list(config["basis_gates"]),
        "transpile_qubit_limit": int(config["transpile_qubit_limit"]),
        "transpile_optimization_level": int(config["transpile_optimization_level"]),
        "block_results_path": str(config["block_results_path"]),
        "block_matrices_dir": str(config["block_matrices_dir"]),
        "end_to_end_results_path": str(config["end_to_end_results_path"]),
        "artifact_match_rtol": float(config["artifact_match_rtol"]),
        "artifact_match_atol": float(config["artifact_match_atol"]),
    }


def _pad_odd_coefficients(coefficients: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.size < int(degree) + 1:
        values = np.pad(values, (0, int(degree) + 1 - values.size))
    values = values[: int(degree) + 1].copy()
    values[0::2] = 0.0
    values[np.abs(values) < 1.0e-14] = 0.0
    return values


def _labels(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return [f"{row.case_name}-{int(row.subproblem_size)}" for row in frame.itertuples(index=False)]


def _positive_for_log(values: pd.Series) -> np.ndarray:
    return np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 1.0e-18)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def _count_status(frame: pd.DataFrame, column: str, value: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((frame[column] == value).sum())


def _count_circuit_successes(frame: pd.DataFrame) -> int:
    if frame.empty or "qsvt_circuit_status" not in frame:
        return 0
    return int(frame["qsvt_circuit_status"].isin(["completed", "circuit_object_built"]).sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run full gate-level QSVT coverage audit")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--epsilon-target", type=float, default=1.0e-2)
    parser.add_argument("--transpile-qubit-limit", type=int, default=3)
    args = parser.parse_args(argv)
    run = run_full_gate_level_qsvt_coverage(
        {
            "output_root": args.output_root,
            "alpha": args.alpha,
            "epsilon_target": args.epsilon_target,
            "transpile_qubit_limit": args.transpile_qubit_limit,
            "degree_summary_path": str(
                Path(args.output_root) / "tables" / "table_degree_alpha_precision_summary.csv"
            ),
            "degree_results_path": str(
                Path(args.output_root)
                / "degree_alpha_precision_sweep"
                / "degree_alpha_precision_sweep_results.csv"
            ),
            "end_to_end_results_path": str(
                Path(args.output_root)
                / "end_to_end_qsvt_vs_ridge"
                / "end_to_end_qsvt_vs_ridge_results.csv"
            ),
            "block_results_path": str(
                Path(args.output_root)
                / "explicit_block_encoding_demo"
                / "block_encoding_demo_results.csv"
            ),
            "block_matrices_dir": str(
                Path(args.output_root) / "explicit_block_encoding_demo" / "matrices"
            ),
        }
    )
    print(f"Wrote full gate-level QSVT coverage audit to {run['output_dir']}")


if __name__ == "__main__":
    main()
