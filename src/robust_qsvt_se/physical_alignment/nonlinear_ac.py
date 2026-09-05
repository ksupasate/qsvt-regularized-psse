"""Multi-seed nonlinear AC integration with matched Ridge/QSVT targets."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_physical_alignment_mpl")
)

import numpy as np
import pandas as pd

from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.experiments.iterative_ac import (
    ACNonlinearProblem,
    _linearized_update_system,
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.measurement.ac_linear import ac_measurements_and_jacobian
from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import run_full_rectangular_qsvt
from robust_qsvt_se.paper.tqe_revision_core import select_alpha_gcv
from robust_qsvt_se.physical_alignment.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    environment_provenance,
)
from robust_qsvt_se.physical_alignment.config import load_campaign_config
from robust_qsvt_se.physical_alignment.structures import load_instance
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint

STUDY_ID = "tqe_physical_alignment_nonlinear_ac_v1"
EVIDENCE_EXACT = "exact classical matrix action"
EVIDENCE_STATEVECTOR = "executed statevector"
EVIDENCE_MODELED = "modeled"
EVIDENCE_FAILED = "failed"


def build_problem_config(
    settings: Mapping[str, Any], scenario: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "run_name": f"{STUDY_ID}_{scenario['scenario_id']}_seed{seed}",
        "seed": int(seed),
        "system": {
            "case_name": settings["case_name"],
            "case_source": settings["case_source"],
            "mode": "nonlinear_ac_state_estimation",
            "measurement": dict(settings["measurement"]),
            "linearization": dict(settings["linearization"]),
            "iteration": dict(settings["iteration"]),
        },
        "scenario": {
            "name": scenario["scenario_id"],
            "noise_std": float(scenario["noise_std"]),
            "missing_ratio": float(scenario["missing_ratio"]),
            "bad_data": dict(scenario["bad_data"]),
        },
        "estimators": [{"name": "ridge", "alpha": float(settings["fixed_alpha"])}],
        "output": {"root": "outputs", "run_id": "not_written_by_problem_builder"},
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("a physical functional cannot be the zero vector")
    return values / norm


def nonlinear_functionals(problem: ACNonlinearProblem) -> list[dict[str, Any]]:
    angle_buses = list(problem.case.angle_state_buses)
    voltage_buses = list(problem.case.voltage_state_buses)
    angle_count = len(angle_buses)
    dimension = len(problem.true_state)
    rows: list[dict[str, Any]] = []

    angle = np.zeros(dimension)
    angle[0] = 1.0
    rows.append(
        {
            "functional_id": f"coordinate_angle_bus_{angle_buses[0]}",
            "functional_family": "coordinate_angle_update",
            "vector": _unit(angle),
            "bus_ids": [int(angle_buses[0])],
        }
    )
    voltage_bus = 14 if 14 in voltage_buses else int(voltage_buses[-1])
    voltage = np.zeros(dimension)
    voltage[angle_count + voltage_buses.index(voltage_bus)] = 1.0
    rows.append(
        {
            "functional_id": f"coordinate_voltage_bus_{voltage_bus}",
            "functional_family": "coordinate_voltage_magnitude_update",
            "vector": _unit(voltage),
            "bus_ids": [voltage_bus],
        }
    )
    for branch in problem.case.branches:
        from_bus = int(branch.from_bus)
        to_bus = int(branch.to_bus)
        if from_bus in angle_buses and to_bus in angle_buses:
            difference = np.zeros(dimension)
            difference[angle_buses.index(from_bus)] = 1.0
            difference[angle_buses.index(to_bus)] = -1.0
            rows.append(
                {
                    "functional_id": f"branch_angle_difference_{from_bus}_{to_bus}",
                    "functional_family": "real_branch_angle_difference",
                    "vector": _unit(difference),
                    "bus_ids": [from_bus, to_bus],
                }
            )
            break
    weak_area = [bus for bus in (12, 13, 14) if bus in angle_buses]
    if weak_area:
        aggregate = np.zeros(dimension)
        for bus in weak_area:
            aggregate[angle_buses.index(bus)] = 1.0
        rows.append(
            {
                "functional_id": "weak_area_angle_aggregate",
                "functional_family": "connected_area_angle_aggregate",
                "vector": _unit(aggregate),
                "bus_ids": weak_area,
            }
        )
    weak_voltage = [bus for bus in (12, 13, 14) if bus in voltage_buses]
    if weak_voltage:
        aggregate = np.zeros(dimension)
        for bus in weak_voltage:
            aggregate[angle_count + voltage_buses.index(bus)] = 1.0
        rows.append(
            {
                "functional_id": "weak_area_voltage_aggregate",
                "functional_family": "connected_area_voltage_aggregate",
                "vector": _unit(aggregate),
                "bus_ids": weak_voltage,
            }
        )
    if not all(abs(float(np.linalg.norm(row["vector"])) - 1.0) <= 1.0e-12 for row in rows):
        raise RuntimeError("nonlinear physical functional normalization failed")
    return rows


def _solver_alpha(
    solver: str, matrix: np.ndarray, residual: np.ndarray, settings: Mapping[str, Any]
) -> float | None:
    if solver in {"ridge_fixed_alpha", "qsvt_target_exact_fixed_alpha"}:
        return float(settings["fixed_alpha"])
    if solver == "ridge_gcv_selected_alpha":
        return select_alpha_gcv(
            matrix, residual, np.asarray(settings["gcv_alpha_grid"], dtype=np.float64)
        )
    if solver in {
        "ridge_degree_feasible_alpha",
        "qsvt_target_exact_degree_feasible_alpha",
    }:
        beta = float(np.linalg.svd(matrix, compute_uv=False)[0])
        return float(settings["degree_feasible_normalized_lambda"]) * beta**2
    return None


def _solve_update(
    solver: str,
    system: Any,
    alpha: float | None,
    settings: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if solver == "pseudoinverse":
        result = PseudoinverseEstimator(rcond=1.0e-12).solve(system)
    elif solver.startswith("ridge_"):
        if alpha is None:
            raise RuntimeError("Ridge alpha is unavailable")
        result = RidgeEstimator(alpha).solve(system)
    elif solver == "truncated_svd":
        result = TruncatedSVDEstimator(float(settings["tsvd_tau"])).solve(system)
    elif solver == "huber_irls":
        huber = settings["huber"]
        result = HuberIRLSEstimator(
            delta=float(huber["delta"]),
            max_iterations=int(huber["max_iterations"]),
            tolerance=float(huber["tolerance"]),
        ).solve(system)
    elif solver.startswith("qsvt_target_exact_"):
        if alpha is None:
            raise RuntimeError("QSVT-target alpha is unavailable")
        result = QSVTSpectralEstimator(alpha).solve(system)
    else:
        raise ValueError(f"unknown nonlinear solver {solver}")

    diagnostics: dict[str, Any] = {
        "matched_qsvt_ridge_update_absolute_error": None,
        "matched_qsvt_ridge_update_relative_error": None,
        "matched_qsvt_ridge_pass": None,
        "qsvt_evidence_status": None,
        "qsvt_degree": None,
        "polynomial_admissibility_status": None,
        "postselection_probability": None,
        "statevector_action_error": None,
    }
    if solver.startswith("qsvt_target_exact_") and not result.failed:
        ridge = RidgeEstimator(float(alpha)).solve(system)
        difference = np.asarray(result.x_hat) - np.asarray(ridge.x_hat)
        absolute = float(np.linalg.norm(difference))
        relative = absolute / max(float(np.linalg.norm(ridge.x_hat)), 1.0e-30)
        diagnostics.update(
            {
                "matched_qsvt_ridge_update_absolute_error": absolute,
                "matched_qsvt_ridge_update_relative_error": relative,
                "matched_qsvt_ridge_pass": bool(
                    relative <= float(settings["qsvt_ridge_update_relative_tolerance"])
                ),
                "qsvt_evidence_status": EVIDENCE_EXACT,
                "polynomial_admissibility_status": "not_applicable_exact_spectral_target",
            }
        )
    return result, diagnostics


def _problem_perturbation_record(problem: ACNonlinearProblem) -> dict[str, Any]:
    true_measurements, _, rows = ac_measurements_and_jacobian(
        problem.case, problem.true_state, problem.measurement_config
    )
    kept = true_measurements[problem.kept_row_indices]
    labels = [rows[index].label for index in problem.kept_row_indices]
    if labels != problem.measurement_labels:
        raise RuntimeError("raw-measurement layout mismatch")
    perturbation = problem.z - kept
    metadata = problem.config_metadata
    return {
        "raw_measurement_perturbation_path_used": True,
        "measurement_model": "z=h(x_true)+e+b",
        "measurement_count": len(problem.z),
        "dropped_measurement_count": len(metadata.get("dropped_rows", ())),
        "bad_data_count": int(metadata.get("bad_data_count", 0)),
        "bad_data_signs": json.dumps(metadata.get("bad_data_signs", []), separators=(",", ":")),
        "measurement_perturbation_l2_norm": float(np.linalg.norm(perturbation)),
        "measurement_stds_fingerprint": stable_array_fingerprint(problem.measurement_stds),
        "measurement_covariance_model": "diagonal implicit R_ii=sigma_i^2",
    }


def _run_solver(
    problem: ACNonlinearProblem,
    *,
    solver: str,
    scenario_id: str,
    seed: int,
    settings: Mapping[str, Any],
    configuration_id: str,
    configuration_hash: str,
    perturbation_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    iteration_settings = settings["iteration"]
    max_iterations = int(iteration_settings["max_iterations"])
    update_tolerance = float(iteration_settings["update_tolerance"])
    residual_tolerance = float(iteration_settings["residual_tolerance"])
    damping = float(iteration_settings["damping"])
    max_update_norm = float(iteration_settings["max_update_norm"])
    residual_growth_limit = float(iteration_settings["residual_growth_limit"])
    min_voltage = float(settings["linearization"]["min_voltage_magnitude"])
    angle_count = len(problem.case.angle_state_buses)
    run_id = f"{configuration_hash}|{scenario_id}|{seed}|{solver}"
    started = time.perf_counter()
    state = problem.initial_state.copy()
    iteration_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    converged = False
    failed = False
    failure_reason = ""
    selected_alphas: list[float] = []

    for iteration in range(max_iterations):
        system, residual_before = _linearized_update_system(problem, state)
        matrix = np.asarray(system.H_tilde, dtype=np.float64)
        residual = np.asarray(system.r_tilde, dtype=np.float64)
        alpha = _solver_alpha(solver, matrix, residual, settings)
        if alpha is not None:
            selected_alphas.append(float(alpha))
        singular = np.linalg.svd(matrix, compute_uv=False)
        positive = singular[singular > 1.0e-14]
        kappa_h = float(positive.max() / positive.min()) if len(positive) else np.inf
        gram = matrix.T @ matrix
        gram_condition = (
            float(np.linalg.cond(gram + float(alpha) * np.eye(matrix.shape[1])))
            if alpha is not None
            else None
        )
        result, qsvt = _solve_update(solver, system, alpha, settings)
        base = {
            "configuration_id": configuration_id,
            "configuration_hash": configuration_hash,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "solver": solver,
            "iteration": int(iteration),
            "matrix_shape_rows": int(matrix.shape[0]),
            "matrix_shape_columns": int(matrix.shape[1]),
            "matrix_nnz": int(np.count_nonzero(matrix)),
            "jacobian_fingerprint": stable_array_fingerprint(matrix),
            "residual_fingerprint": stable_array_fingerprint(residual),
            "jacobian_rebuilt_this_iteration": True,
            "residual_rebuilt_from_raw_measurements_this_iteration": True,
            "kappa_H": kappa_h,
            "kappa_HtH_plus_alphaI": gram_condition,
            "selected_alpha": alpha,
            "weighted_residual_before": float(residual_before),
            "qsvt_degree": qsvt["qsvt_degree"],
            "polynomial_admissibility_status": qsvt["polynomial_admissibility_status"],
            "qsvt_evidence_status": qsvt["qsvt_evidence_status"],
            "statevector_action_error": qsvt["statevector_action_error"],
            "matched_qsvt_ridge_update_absolute_error": qsvt[
                "matched_qsvt_ridge_update_absolute_error"
            ],
            "matched_qsvt_ridge_update_relative_error": qsvt[
                "matched_qsvt_ridge_update_relative_error"
            ],
            "matched_qsvt_ridge_pass": qsvt["matched_qsvt_ridge_pass"],
            "postselection_probability": qsvt["postselection_probability"],
        }
        if result.failed or not np.all(np.isfinite(result.x_hat)):
            failed = True
            failure_reason = result.failure_reason or "update solver failed"
            iteration_rows.append(
                {
                    **base,
                    "update_norm": None,
                    "weighted_residual_after": None,
                    "state_rmse_after": None,
                    "angle_rmse_after": None,
                    "voltage_magnitude_rmse_after": None,
                    "converged": False,
                    "status": "failed",
                    "failure_reason": failure_reason,
                }
            )
            break
        update = np.asarray(result.x_hat, dtype=np.float64)
        update_norm = float(np.linalg.norm(update))
        if update_norm > max_update_norm:
            failed = True
            failure_reason = f"update norm exceeded max_update_norm={max_update_norm}"
            iteration_rows.append(
                {
                    **base,
                    "update_norm": update_norm,
                    "weighted_residual_after": None,
                    "state_rmse_after": None,
                    "angle_rmse_after": None,
                    "voltage_magnitude_rmse_after": None,
                    "converged": False,
                    "status": "failed",
                    "failure_reason": failure_reason,
                }
            )
            break
        next_state = state + damping * update
        next_state[angle_count:] = np.maximum(next_state[angle_count:], min_voltage)
        residual_after = _weighted_residual_norm(problem, next_state)
        if not np.isfinite(residual_after):
            failed = True
            failure_reason = "weighted residual became nonfinite"
        elif residual_before > 0.0 and residual_after / residual_before > residual_growth_limit:
            failed = True
            failure_reason = (
                f"weighted residual growth exceeded residual_growth_limit={residual_growth_limit}"
            )
        state_error = next_state - problem.true_state
        converged = bool(
            not failed and (update_norm <= update_tolerance or residual_after <= residual_tolerance)
        )
        iteration_rows.append(
            {
                **base,
                "update_norm": update_norm,
                "weighted_residual_after": float(residual_after),
                "state_rmse_after": float(np.sqrt(np.mean(state_error**2))),
                "angle_rmse_after": float(np.sqrt(np.mean(state_error[:angle_count] ** 2))),
                "voltage_magnitude_rmse_after": float(
                    np.sqrt(np.mean(state_error[angle_count:] ** 2))
                ),
                "converged": converged,
                "status": "failed" if failed else ("converged" if converged else "continued"),
                "failure_reason": failure_reason,
            }
        )
        if solver.startswith("qsvt_target_exact_"):
            equivalence_rows.append(
                {
                    **base,
                    "evidence_status": EVIDENCE_EXACT,
                    "status": "pass" if qsvt["matched_qsvt_ridge_pass"] else "fail",
                    "failure_reason": (
                        ""
                        if qsvt["matched_qsvt_ridge_pass"]
                        else "matched QSVT-target/Ridge tolerance exceeded"
                    ),
                }
            )
        if not failed:
            state = next_state
        if failed or converged:
            break

    if not converged and not failed:
        failure_reason = "max_iterations reached"
    final_error = state - problem.true_state
    selected_alpha_final = selected_alphas[-1] if selected_alphas else None
    raw = {
        "logical_key": run_id,
        "configuration_id": configuration_id,
        "configuration_hash": configuration_hash,
        "run_id": run_id,
        "run_kind": "full_nonlinear_loop",
        "case_name": settings["case_name"],
        "scenario_id": scenario_id,
        "seed": int(seed),
        "solver": solver,
        "converged": converged,
        "failed": failed,
        "status": "failed" if failed else ("converged" if converged else "non_converged"),
        "failure_reason": failure_reason,
        "iteration_count": len(iteration_rows),
        "final_full_state_rmse": float(np.sqrt(np.mean(final_error**2))),
        "final_angle_rmse": float(np.sqrt(np.mean(final_error[:angle_count] ** 2))),
        "final_voltage_magnitude_rmse": float(np.sqrt(np.mean(final_error[angle_count:] ** 2))),
        "final_weighted_residual": float(_weighted_residual_norm(problem, state)),
        "selected_alpha_final": selected_alpha_final,
        "selected_alpha_min": min(selected_alphas) if selected_alphas else None,
        "selected_alpha_max": max(selected_alphas) if selected_alphas else None,
        "qsvt_degree": None,
        "polynomial_admissibility_status": (
            "not_applicable_exact_spectral_target"
            if solver.startswith("qsvt_target_exact_")
            else "not_applicable_classical_solver"
        ),
        "qsvt_evidence_status": (
            EVIDENCE_EXACT if solver.startswith("qsvt_target_exact_") else None
        ),
        "statevector_action_error": None,
        "max_matched_qsvt_ridge_update_relative_error": (
            max(row["matched_qsvt_ridge_update_relative_error"] or 0.0 for row in iteration_rows)
            if solver.startswith("qsvt_target_exact_") and iteration_rows
            else None
        ),
        "postselection_probability": None,
        "runtime_seconds": time.perf_counter() - started,
        **perturbation_record,
    }
    return raw, iteration_rows, equivalence_rows, state.copy()


def _selected_output_rows(
    raw_run: Mapping[str, Any],
    problem: ACNonlinearProblem,
    final_state: np.ndarray,
    floor: float,
    functionals: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    truth_update = problem.true_state - problem.initial_state
    estimated_update = np.asarray(final_state) - problem.initial_state
    rows = []
    for functional in functionals or nonlinear_functionals(problem):
        vector = np.asarray(functional["vector"], dtype=np.float64)
        y_true = float(vector @ truth_update)
        y_estimated = float(vector @ estimated_update)
        absolute = abs(y_estimated - y_true)
        rows.append(
            {
                "logical_key": f"{raw_run['run_id']}|{functional['functional_id']}",
                "configuration_id": raw_run["configuration_id"],
                "configuration_hash": raw_run["configuration_hash"],
                "run_id": raw_run["run_id"],
                "case_name": raw_run["case_name"],
                "scenario_id": raw_run["scenario_id"],
                "seed": raw_run["seed"],
                "solver": raw_run["solver"],
                "functional_id": functional["functional_id"],
                "functional_family": functional["functional_family"],
                "bus_ids": json.dumps(functional["bus_ids"], separators=(",", ":")),
                "functional_norm": float(np.linalg.norm(vector)),
                "y_true_update": y_true,
                "y_estimated_update": y_estimated,
                "A_physical": absolute,
                "E_physical": absolute / max(abs(y_true), float(floor)),
                "normalization_floor": float(floor),
                "near_zero_y_true": abs(y_true) < float(floor),
                "status": raw_run["status"],
                "failure_reason": raw_run["failure_reason"],
            }
        )
    return rows


def _statevector_boundary_rows(
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boundary = settings["statevector_boundary"]
    execute_scenarios = set(boundary["execute_scenario_ids"])
    execute_seeds = {int(seed) for seed in boundary["execute_seed_ids"]}
    frozen_root = Path(config["source_structural_root"])
    registry = pd.read_csv(frozen_root / config["structure_design"]["instance_registry"])
    first_id = str(
        registry.loc[registry["ieee_case"].eq(settings["case_name"])]
        .sort_values(["structural_group_selection_order", "realization_order"])
        .iloc[0]["instance_id"]
    )
    frozen = load_instance(frozen_root, first_id)
    selected_rows = np.asarray(frozen.selected_rows, dtype=np.int64)
    selected_columns = np.asarray(frozen.selected_columns, dtype=np.int64)
    records: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    phase_cache = output_dir / "phase_cache"
    phase_cache.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            base = {
                "configuration_id": config["configuration_id"],
                "configuration_hash": config["configuration_hash"],
                "case_name": settings["case_name"],
                "scenario_id": scenario_id,
                "seed": int(seed),
                "iteration": 0,
                "block_source_instance_id": first_id,
                "selected_global_rows": json.dumps(selected_rows.tolist(), separators=(",", ":")),
                "selected_global_columns": json.dumps(
                    selected_columns.tolist(), separators=(",", ":")
                ),
                "selection_outcome_independent": True,
                "matrix_shape_rows": len(selected_rows),
                "matrix_shape_columns": len(selected_columns),
            }
            if scenario_id not in execute_scenarios or int(seed) not in execute_seeds:
                records.append(
                    {
                        **base,
                        "attempt_id": f"{scenario_id}|{seed}|modeled",
                        "evidence_status": EVIDENCE_MODELED,
                        "execution_status": "not_executed_campaign_boundary",
                        "failure_reason": "outside frozen statevector execution subset",
                        "alpha": None,
                        "normalized_lambda": float(boundary["normalized_lambda"]),
                        "degree": None,
                        "margin": None,
                        "polynomial_admissibility_status": "modeled_not_synthesized",
                        "statevector_action_relative_error": None,
                        "postselection_probability": None,
                        "circuit_vs_matvec_error": None,
                        "runtime_seconds": 0.0,
                    }
                )
                continue
            problem = build_ac_nonlinear_problem(
                build_problem_config(settings, scenario, int(seed))
            )
            system, _ = _linearized_update_system(problem, problem.initial_state.copy())
            matrix = np.asarray(system.H_tilde)[np.ix_(selected_rows, selected_columns)]
            residual = np.asarray(system.r_tilde)[selected_rows]
            beta = float(np.linalg.svd(matrix, compute_uv=False)[0])
            alpha = float(boundary["normalized_lambda"]) * beta**2
            executed = False
            for degree in boundary["degree_candidates"]:
                for margin in boundary["margin_candidates"]:
                    started = time.perf_counter()
                    result = run_full_rectangular_qsvt(
                        matrix,
                        residual,
                        alpha=alpha,
                        degree=int(degree),
                        margin=float(margin),
                        phase_cache_dir=phase_cache,
                        beta=beta,
                        run_circuit_path=bool(boundary["run_compiled_statevector_circuit"]),
                    )
                    status = str(result["status"])
                    evidence = (
                        EVIDENCE_STATEVECTOR if status.startswith("executed") else EVIDENCE_FAILED
                    )
                    relative = result.get("update_relative_error_vs_full_ridge")
                    row = {
                        **base,
                        "attempt_id": f"{scenario_id}|{seed}|d{degree}|m{margin}",
                        "evidence_status": evidence,
                        "execution_status": status,
                        "failure_reason": result.get("failure_reason", ""),
                        "alpha": alpha,
                        "normalized_lambda": alpha / beta**2,
                        "degree": int(degree),
                        "margin": float(margin),
                        "polynomial_admissibility_status": (
                            "bounded_certified" if status.startswith("executed") else status
                        ),
                        "statevector_action_relative_error": relative,
                        "postselection_probability": result.get("postselection_probability"),
                        "circuit_vs_matvec_error": result.get("circuit_vs_matvec_error"),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                    records.append(row)
                    if status.startswith("executed"):
                        passed = bool(
                            relative is not None
                            and float(relative) <= float(boundary["pass_relative_tolerance"])
                        )
                        equivalence.append(
                            {
                                **base,
                                "run_id": f"statevector|{scenario_id}|{seed}|0|{first_id}",
                                "solver": "qsvt_statevector_selected_block",
                                "selected_alpha": alpha,
                                "qsvt_degree": int(degree),
                                "polynomial_admissibility_status": "bounded_certified",
                                "qsvt_evidence_status": EVIDENCE_STATEVECTOR,
                                "statevector_action_error": float(relative),
                                "matched_qsvt_ridge_update_absolute_error": result.get(
                                    "update_max_abs_error_vs_full_ridge"
                                ),
                                "matched_qsvt_ridge_update_relative_error": float(relative),
                                "matched_qsvt_ridge_pass": passed,
                                "postselection_probability": result.get(
                                    "postselection_probability"
                                ),
                                "evidence_status": EVIDENCE_STATEVECTOR,
                                "status": "pass" if passed else "fail",
                                "failure_reason": ""
                                if passed
                                else "statevector tolerance exceeded",
                            }
                        )
                        executed = True
                        break
                if executed:
                    break
    return records, equivalence


def _summary_frames(raw: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_run = (
        selected.groupby(["run_id", "scenario_id", "seed", "solver"], sort=True)
        .agg(
            mean_selected_output_E_physical=("E_physical", "mean"),
            median_selected_output_E_physical=("E_physical", "median"),
            mean_selected_output_A_physical=("A_physical", "mean"),
        )
        .reset_index()
    )
    merged = raw.merge(selected_run, on=["run_id", "scenario_id", "seed", "solver"], how="left")
    group_columns = ["scenario_id", "solver"]
    scenario = (
        merged.groupby(group_columns, sort=True)
        .agg(
            run_count=("run_id", "count"),
            independent_seed_count=("seed", "nunique"),
            convergence_count=("converged", "sum"),
            failure_count=("failed", "sum"),
            non_converged_count=(
                "status",
                lambda values: int(values.eq("non_converged").sum()),
            ),
            median_final_full_state_rmse=("final_full_state_rmse", "median"),
            mean_final_full_state_rmse=("final_full_state_rmse", "mean"),
            median_final_angle_rmse=("final_angle_rmse", "median"),
            median_final_voltage_magnitude_rmse=("final_voltage_magnitude_rmse", "median"),
            median_final_weighted_residual=("final_weighted_residual", "median"),
            median_selected_output_E_physical=("median_selected_output_E_physical", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .reset_index()
    )
    scenario["convergence_rate"] = scenario["convergence_count"] / scenario["run_count"]
    convergence = (
        raw.groupby(["scenario_id", "solver", "status"], sort=True)
        .size()
        .rename("run_count")
        .reset_index()
    )
    return scenario, convergence


def run_nonlinear_campaign(
    config_path: str | Path = "configs/tqe_physical_alignment/campaign.json",
    *,
    output_dir: str | Path | None = None,
    scenario_subset: Sequence[str] | None = None,
    seed_subset: Sequence[int] | None = None,
    solver_subset: Sequence[str] | None = None,
    run_statevector_boundary: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    config = load_campaign_config(config_path)
    settings = config["nonlinear_ac"]
    destination = Path(output_dir or (Path(config["output_root"]) / "nonlinear_ac"))
    destination.mkdir(parents=True, exist_ok=True)
    requested_scenarios = set(
        scenario_subset or [row["scenario_id"] for row in settings["scenarios"]]
    )
    scenarios = [row for row in settings["scenarios"] if row["scenario_id"] in requested_scenarios]
    seeds = [int(seed) for seed in (seed_subset or settings["seeds"])]
    solvers = list(solver_subset or settings["solvers"])
    unknown_solvers = set(solvers) - set(settings["solvers"])
    if unknown_solvers:
        raise ValueError(f"unknown nonlinear solvers: {sorted(unknown_solvers)}")

    raw_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(seeds) * len(solvers)
    ordinal = 0
    for scenario in scenarios:
        for seed in seeds:
            problem = build_ac_nonlinear_problem(build_problem_config(settings, scenario, seed))
            perturbation_record = _problem_perturbation_record(problem)
            functionals = nonlinear_functionals(problem)
            for solver in solvers:
                ordinal += 1
                raw, iterations, equivalence, final_state = _run_solver(
                    problem,
                    solver=solver,
                    scenario_id=str(scenario["scenario_id"]),
                    seed=seed,
                    settings=settings,
                    configuration_id=config["configuration_id"],
                    configuration_hash=config["configuration_hash"],
                    perturbation_record=perturbation_record,
                )
                raw_rows.append(raw)
                iteration_rows.extend(iterations)
                equivalence_rows.extend(equivalence)
                selected_rows.extend(
                    _selected_output_rows(
                        raw,
                        problem,
                        final_state,
                        float(settings["selected_output_floor"]),
                        functionals,
                    )
                )
                if raw["status"] != "converged":
                    failure_rows.append(
                        {
                            "configuration_id": config["configuration_id"],
                            "configuration_hash": config["configuration_hash"],
                            "run_id": raw["run_id"],
                            "case_name": raw["case_name"],
                            "scenario_id": raw["scenario_id"],
                            "seed": raw["seed"],
                            "solver": raw["solver"],
                            "stage": "full_nonlinear_loop",
                            "status": raw["status"],
                            "failure_reason": raw["failure_reason"],
                        }
                    )
                if verbose:
                    print(
                        f"nonlinear run {ordinal}/{total}: {scenario['scenario_id']} "
                        f"seed={seed} solver={solver} status={raw['status']}",
                        flush=True,
                    )

    statevector_rows: list[dict[str, Any]] = []
    if run_statevector_boundary:
        statevector_rows, statevector_equivalence = _statevector_boundary_rows(
            config, settings, scenarios, seeds, destination
        )
        equivalence_rows.extend(statevector_equivalence)
        for row in statevector_rows:
            if row["evidence_status"] == EVIDENCE_FAILED:
                failure_rows.append(
                    {
                        "configuration_id": config["configuration_id"],
                        "configuration_hash": config["configuration_hash"],
                        "run_id": row["attempt_id"],
                        "case_name": row["case_name"],
                        "scenario_id": row["scenario_id"],
                        "seed": row["seed"],
                        "solver": "qsvt_statevector_selected_block",
                        "stage": "statevector_boundary",
                        "status": row["execution_status"],
                        "failure_reason": row["failure_reason"],
                    }
                )

    raw_frame = pd.DataFrame(raw_rows)
    iteration_frame = pd.DataFrame(iteration_rows)
    selected_frame = pd.DataFrame(selected_rows)
    equivalence_frame = pd.DataFrame(equivalence_rows)
    statevector_frame = pd.DataFrame(statevector_rows)
    failure_frame = pd.DataFrame(
        failure_rows,
        columns=(
            list(failure_rows[0])
            if failure_rows
            else [
                "configuration_id",
                "configuration_hash",
                "run_id",
                "case_name",
                "scenario_id",
                "seed",
                "solver",
                "stage",
                "status",
                "failure_reason",
            ]
        ),
    )
    if raw_frame["logical_key"].duplicated().any():
        raise RuntimeError("duplicate nonlinear run logical keys")
    if selected_frame["logical_key"].duplicated().any():
        raise RuntimeError("duplicate nonlinear selected-output logical keys")
    scenario_frame, convergence_frame = _summary_frames(raw_frame, selected_frame)
    atomic_write_csv(destination / "raw_runs.csv", raw_frame)
    atomic_write_csv(destination / "iteration_rows.csv", iteration_frame)
    atomic_write_csv(destination / "selected_output_rows.csv", selected_frame)
    atomic_write_csv(destination / "scenario_summary.csv", scenario_frame)
    atomic_write_csv(destination / "convergence_summary.csv", convergence_frame)
    atomic_write_csv(destination / "qsvt_ridge_equivalence.csv", equivalence_frame)
    atomic_write_csv(destination / "qsvt_statevector_rows.csv", statevector_frame)
    atomic_write_csv(destination / "failure_registry.csv", failure_frame)

    qsvt_classical = (
        equivalence_frame.loc[equivalence_frame["evidence_status"].eq(EVIDENCE_EXACT)]
        if "evidence_status" in equivalence_frame
        else pd.DataFrame()
    )
    validation = {
        "study_id": STUDY_ID,
        "configuration_id": config["configuration_id"],
        "configuration_hash": config["configuration_hash"],
        "full_design_run": scenario_subset is None
        and seed_subset is None
        and solver_subset is None,
        "scenario_count": int(raw_frame["scenario_id"].nunique()),
        "seed_count": int(raw_frame["seed"].nunique()),
        "solver_count": int(raw_frame["solver"].nunique()),
        "raw_run_count": len(raw_frame),
        "iteration_row_count": len(iteration_frame),
        "selected_output_row_count": len(selected_frame),
        "raw_measurement_perturbation_path_all_runs": bool(
            raw_frame["raw_measurement_perturbation_path_used"].all()
        ),
        "per_iteration_jacobian_rebuilt_all_rows": bool(
            iteration_frame["jacobian_rebuilt_this_iteration"].all()
        ),
        "qsvt_target_ridge_max_relative_error": (
            float(qsvt_classical["matched_qsvt_ridge_update_relative_error"].max())
            if not qsvt_classical.empty
            else None
        ),
        "qsvt_target_ridge_all_pass": bool(qsvt_classical["status"].eq("pass").all()),
        "executed_statevector_count": int(
            statevector_frame["evidence_status"].eq(EVIDENCE_STATEVECTOR).sum()
        )
        if not statevector_frame.empty
        else 0,
        "modeled_statevector_count": int(
            statevector_frame["evidence_status"].eq(EVIDENCE_MODELED).sum()
        )
        if not statevector_frame.empty
        else 0,
        "failure_registry_rows": len(failure_frame),
        "weighted_residual_and_rmse_are_distinct_fields": True,
    }
    provenance = {
        **validation,
        "resolved_nonlinear_configuration": settings,
        "environment": environment_provenance(Path.cwd()),
        "measurement_equation": "z=h(x_true)+e+b",
        "jacobian_equation": "H_k=dh/dx evaluated at x_k and rebuilt each iteration",
        "whitening": "rowwise division by measurement sigma; diagonal covariance remains implicit",
        "qsvt_target_definition_changed": False,
        "ridge_definition_changed": False,
        "statevector_scope": "one frozen outcome-independent 8x8 block at one selected iteration",
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(destination / "provenance.json", provenance)
    atomic_write_json(destination / "validation_summary.json", validation)
    report = [
        "# Nonlinear AC Integration Validation",
        "",
        f"- Scenarios: {validation['scenario_count']}",
        f"- Independent seeds: {validation['seed_count']}",
        f"- Full nonlinear solver runs: {validation['raw_run_count']}",
        f"- Converged runs: {int(raw_frame['converged'].sum())}",
        f"- Failed runs: {int(raw_frame['failed'].sum())}",
        "- Non-converged runs and statevector failures are retained in failure_registry.csv.",
        "- Raw measurements are perturbed before iteration; the residual and Jacobian are "
        "rebuilt at every iteration.",
        "- Full-state RMSE and weighted residual are retained as distinct quantities.",
        "- Exact classical QSVT-target rows implement the matched Ridge/Tikhonov spectral "
        "filter and are not a superiority comparison.",
        "- Maximum exact-target/Ridge update error: "
        f"{validation['qsvt_target_ridge_max_relative_error']}",
        f"- Executed statevector rows: {validation['executed_statevector_count']}",
        "- All executions are classical simulations with generated measurements; no hardware "
        "or field-data claim is supported.",
        "",
    ]
    atomic_write_text(destination / "validation_report.md", "\n".join(report))
    return validation
