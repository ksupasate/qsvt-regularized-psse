from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import current_command, git_commit, utc_timestamp
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, matrix_density
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.hardware_resource_estimator import (
    build_hardware_resource_report,
    estimate_hardware_resources,
)
from robust_qsvt_se.qsvt.norm_success import (
    NORM_RECOVERY_LIMITATION,
    estimate_success_probability_from_shots,
)
from robust_qsvt_se.qsvt.partial_observable_readout import (
    basis_probability,
    linear_functional_overlap,
    normalize_state,
    subset_probability,
)
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.qsvt.power_observables import (
    area_update_energy,
    branch_angle_difference,
    bus_angle_component,
    bus_voltage_component,
    component_observable,
    difference_observable,
    jacobian_row_observable,
    subset_energy_observable,
)
from robust_qsvt_se.qsvt.qsvt_update_workflow import (
    build_qsvt_update_workflow_artifacts,
    build_subproblem_from_engineering_system,
    run_qsvt_update_state_simulation,
)
from robust_qsvt_se.qsvt.scalable_block_encoding import (
    BlockEncodingModel,
    build_block_encoding_resource_report,
    estimate_block_encoding_resources,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.io import ensure_directory, write_json

PATHWAY_CLAIM_BOUNDARY = (
    "This is a full QSVT implementation pathway with small explicit simulations "
    "and IEEE-scale resource estimates. It does not demonstrate quantum speedup, "
    "full IEEE-scale hardware execution, or QSVT numerical superiority over "
    "Ridge/Tikhonov."
)
SELECTED_SUBPROBLEM_LIMITATION = (
    "This is a selected subproblem demonstration, not a full IEEE14 nonlinear AC QSVT solve."
)


def audit_current_qsvt_pathway(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_pathway_audit",
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = _capability_rows()
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "current_qsvt_capability_matrix.csv"
    summary_path = output_dir / "audit_summary.md"
    frame.to_csv(csv_path, index=False)
    summary_path.write_text(_audit_markdown(frame), encoding="utf-8")
    return {
        "output_dir": output_dir,
        "capability_matrix": frame,
        "artifacts": {"audit_summary": summary_path, "capability_matrix": csv_path},
    }


def build_power_observable_outputs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_observables",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "alpha": 1.0e-4,
        "seed": 123,
        "shots": [100, 1000, 10000],
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    metadata = build_state_metadata_from_system_metadata(system.metadata)
    metadata_frame = metadata.to_frame()
    metadata_path = output_dir / "state_index_metadata.csv"
    metadata_frame.to_csv(metadata_path, index=False)
    ridge_update = _ridge_update(system.H_tilde, system.r_tilde, alpha=float(resolved["alpha"]))
    ridge_state, _ = normalize_state(ridge_update)
    qsvt_target_state = ridge_state.copy()
    observables = _metadata_observables(system.H_tilde, system.metadata, metadata)
    definition_frame = pd.DataFrame([_observable_definition_row(obs) for obs in observables])
    exact_frame = _observable_exact_comparison(observables, ridge_state, qsvt_target_state)
    shot_frame = _observable_shot_frame(
        observables,
        qsvt_target_state,
        ridge_state,
        shots=list(resolved["shots"]),
        seed=int(resolved["seed"]),
    )
    definition_path = output_dir / "observable_definitions.csv"
    exact_path = output_dir / "observable_exact_comparison.csv"
    shot_path = output_dir / "observable_shot_readout.csv"
    summary_path = output_dir / "observable_summary.md"
    definition_frame.to_csv(definition_path, index=False)
    exact_frame.to_csv(exact_path, index=False)
    shot_frame.to_csv(shot_path, index=False)
    summary_path.write_text(
        _observable_markdown(resolved, matrix_source, metadata, exact_frame, shot_frame),
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "state_metadata": metadata_frame,
        "observable_summary": exact_frame,
        "shot_summary": shot_frame,
        "artifacts": {
            "state_index_metadata": metadata_path,
            "observable_definitions": definition_path,
            "observable_exact_comparison": exact_path,
            "observable_shot_readout": shot_path,
            "observable_summary": summary_path,
        },
    }


def run_alpha_degree_tradeoff(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_alpha_degree_tradeoff",
        "case": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_sizes": [4, 8, 16],
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
        "degrees": [15, 25, 35, 51, 75, 101],
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    observable_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for size in [int(value) for value in resolved["submatrix_sizes"]]:
        H_sub, r_sub, sub_meta = build_subproblem_from_engineering_system(
            case=str(resolved["case"]),
            matrix_source=str(resolved["matrix_source"]),
            submatrix_size=size,
            seed=int(resolved["seed"]),
            case_source=str(resolved["case_source"]),
        )
        B = H_sub.T
        for alpha in [float(value) for value in resolved["alphas"]]:
            ridge = _ridge_update(H_sub, r_sub, alpha=alpha)
            ridge_state, _ = normalize_state(ridge)
            for degree in [int(value) for value in resolved["degrees"]]:
                started = time.perf_counter()
                try:
                    result = run_qsvt_update_state_simulation(
                        H_tilde=H_sub,
                        r_tilde=r_sub,
                        alpha=alpha,
                        degree=_as_odd_degree(degree),
                        block_encoding_mode="sparse_access_oracle",
                        phase_method="polynomial_svd_resource_proxy",
                        seed=int(resolved["seed"]),
                    )
                    status = "ok"
                    failure = ""
                    qsvt_state = result.observable_ready_state
                    point_error, bounded_C = _bounded_filter_error(B, alpha, _as_odd_degree(degree))
                    state_error = result.phase_aligned_state_l2_error
                    overlap = result.normalized_state_overlap_abs
                    success_probability = result.success_probability_proxy
                    query_count = result.query_count_estimate
                    phase_count = result.phase_count
                    synthesized_degree = result.synthesized_degree
                except Exception as exc:
                    status = "failed"
                    failure = str(exc)
                    qsvt_state = ridge_state
                    point_error = float("nan")
                    bounded_C = float("nan")
                    state_error = float("nan")
                    overlap = float("nan")
                    success_probability = float("nan")
                    query_count = 2 * _as_odd_degree(degree) + 1
                    phase_count = _as_odd_degree(degree) + 1
                    synthesized_degree = _as_odd_degree(degree)
                runtime = time.perf_counter() - started
                obs_error = _max_default_observable_error(ridge_state, qsvt_state)
                row = {
                    "case": resolved["case"],
                    "submatrix_size": size,
                    "alpha": alpha,
                    "requested_degree": degree,
                    "synthesized_degree": synthesized_degree,
                    "phase_count": phase_count,
                    "maximum_pointwise_filter_error": point_error,
                    "qsvt_state_error_vs_ridge": state_error,
                    "normalized_state_overlap": overlap,
                    "observable_max_error": obs_error,
                    "success_probability_proxy": success_probability,
                    "bounded_scaling_constant_C": bounded_C,
                    "query_count_estimate": query_count,
                    "runtime_seconds": runtime,
                    "phase_synthesis_status": "polynomial_resource_proxy_no_phase_angles",
                    "status": status,
                    "failure_reason": failure,
                    "selected_state_labels": ";".join(sub_meta["selected_state_labels"]),
                    "claim_boundary": PATHWAY_CLAIM_BOUNDARY,
                }
                summary_rows.append(row)
                observable_rows.extend(
                    _tradeoff_observable_rows(
                        row,
                        ridge_state,
                        qsvt_state,
                    )
                )
                resource_rows.append(
                    {
                        "case": resolved["case"],
                        "submatrix_size": size,
                        "alpha": alpha,
                        "degree": degree,
                        "query_count_estimate": query_count,
                        "success_probability_proxy": success_probability,
                        "bounded_scaling_constant_C": bounded_C,
                        "resource_model": "polynomial_qsvt_target_proxy",
                    }
                )
    summary = pd.DataFrame(summary_rows)
    observable = pd.DataFrame(observable_rows)
    resource = pd.DataFrame(resource_rows)
    summary_path = output_dir / "alpha_degree_summary.csv"
    observable_path = output_dir / "alpha_degree_observable_summary.csv"
    resource_path = output_dir / "alpha_degree_resource_summary.csv"
    report_path = output_dir / "alpha_degree_tradeoff.md"
    summary.to_csv(summary_path, index=False)
    observable.to_csv(observable_path, index=False)
    resource.to_csv(resource_path, index=False)
    report_path.write_text(_alpha_degree_markdown(summary), encoding="utf-8")
    write_json(
        output_dir / "manifest.json",
        _manifest(resolved, [summary_path, observable_path, resource_path, report_path]),
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "observable_summary": observable,
        "resource_summary": resource,
        "artifacts": {
            "alpha_degree_summary": summary_path,
            "alpha_degree_observable_summary": observable_path,
            "alpha_degree_resource_summary": resource_path,
            "alpha_degree_tradeoff": report_path,
        },
    }


def run_ieee_scaling_study(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_scaling",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "implemented_submatrix_sizes": [4, 8, 16, 32],
        "alpha": 1.0e-4,
        "degree": 51,
        "seed": 123,
        "explicit_dense_cases": ["ieee14"],
        "explicit_dense_size_limit": 4,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    simulation_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for case in [str(value) for value in resolved["cases"]]:
        case_config = {
            "case_name": case,
            "case_source": resolved["case_source"],
            "matrix_source": "ieee14_ac_weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
        try:
            system, matrix_source = build_engineering_system(case_config)
            H = np.asarray(system.H_tilde, dtype=np.float64)
            singular = np.linalg.svd(H, compute_uv=False)
            condition = _condition_number(singular)
            matrix_rows.append(
                {
                    "case": case,
                    "status": "ok",
                    "matrix_source": matrix_source,
                    "measurement_rows": int(H.shape[0]),
                    "state_dimension": int(H.shape[1]),
                    "nonzeros": int(np.count_nonzero(np.abs(H) > 1.0e-12)),
                    "density": matrix_density(H),
                    "condition_number": condition,
                    "spectral_norm": float(singular[0]) if singular.size else 0.0,
                    "failure_reason": "",
                }
            )
            block_estimate = estimate_block_encoding_resources(
                H,
                BlockEncodingModel.SPARSE_ACCESS_ORACLE,
            )
            hardware = estimate_hardware_resources(
                H,
                qsvt_degree=int(resolved["degree"]),
                phase_count=_as_odd_degree(int(resolved["degree"])) + 1,
                block_encoding_model=BlockEncodingModel.SPARSE_ACCESS_ORACLE,
            )
            resource_rows.append(
                block_estimate.to_row(
                    {
                        "case": case,
                        "category": "resource estimate",
                        "logical_index_qubits": hardware.logical_index_qubits,
                        "total_logical_qubits": hardware.total_logical_qubits,
                        "phase_count": hardware.phase_count,
                        "query_count": hardware.query_count,
                        "state_preparation_assumption": hardware.state_preparation_assumption,
                        "block_encoding_assumption": hardware.block_encoding_assumption,
                    }
                )
            )
        except Exception as exc:
            matrix_rows.append(
                {
                    "case": case,
                    "status": "failed",
                    "matrix_source": "pypower_ac_weighted_jacobian",
                    "measurement_rows": None,
                    "state_dimension": None,
                    "nonzeros": None,
                    "density": None,
                    "condition_number": None,
                    "spectral_norm": None,
                    "failure_reason": str(exc),
                }
            )
            continue

        for size in [int(value) for value in resolved["implemented_submatrix_sizes"]]:
            category = "implemented simulation"
            status = "skipped"
            reason = "explicit dense runtime guard; resource estimate only"
            state_error = None
            success_probability = None
            beta = None
            if case in set(resolved["explicit_dense_cases"]) and size <= int(
                resolved["explicit_dense_size_limit"]
            ):
                try:
                    H_sub, r_sub, _ = build_subproblem_from_engineering_system(
                        case=case,
                        matrix_source=str(resolved["matrix_source"]),
                        submatrix_size=size,
                        seed=int(resolved["seed"]),
                        case_source=str(resolved["case_source"]),
                    )
                    result = run_qsvt_update_state_simulation(
                        H_sub,
                        r_sub,
                        alpha=float(resolved["alpha"]),
                        degree=_as_odd_degree(int(resolved["degree"])),
                        block_encoding_mode="explicit_dense",
                        phase_method="pennylane_poly_to_angles",
                        seed=int(resolved["seed"]),
                    )
                    status = "ok"
                    reason = ""
                    state_error = result.phase_aligned_state_l2_error
                    success_probability = result.success_probability_proxy
                    beta = result.beta
                except Exception as exc:
                    status = "failed"
                    reason = str(exc)
            simulation_rows.append(
                {
                    "case": case,
                    "submatrix_size": size,
                    "category": category if status == "ok" else "resource estimate",
                    "status": status,
                    "alpha": float(resolved["alpha"]),
                    "degree": int(resolved["degree"]),
                    "beta": beta,
                    "qsvt_state_error_vs_ridge": state_error,
                    "success_probability_proxy": success_probability,
                    "failure_or_skip_reason": reason,
                }
            )
    simulation = pd.DataFrame(simulation_rows)
    resource = pd.DataFrame(resource_rows)
    matrix = pd.DataFrame(matrix_rows)
    sim_path = output_dir / "implemented_simulation_summary.csv"
    resource_path = output_dir / "full_ieee_resource_estimates.csv"
    matrix_path = output_dir / "matrix_statistics_by_case.csv"
    report_path = output_dir / "qsvt_scaling_summary.md"
    simulation.to_csv(sim_path, index=False)
    resource.to_csv(resource_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    report_path.write_text(_scaling_markdown(simulation, resource, matrix), encoding="utf-8")
    write_json(
        output_dir / "manifest.json",
        _manifest(resolved, [sim_path, resource_path, matrix_path, report_path]),
    )
    return {
        "output_dir": output_dir,
        "implemented_simulations": simulation,
        "resource_estimates": resource,
        "matrix_statistics": matrix,
        "artifacts": {
            "implemented_simulation_summary": sim_path,
            "full_ieee_resource_estimates": resource_path,
            "matrix_statistics_by_case": matrix_path,
            "qsvt_scaling_summary": report_path,
        },
    }


def run_one_step_ac_update_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_one_step_ac_update_demo",
        "case": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    H_sub, r_sub, sub_meta = build_subproblem_from_engineering_system(
        case=str(resolved["case"]),
        matrix_source=str(resolved["matrix_source"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    result = run_qsvt_update_state_simulation(
        H_sub,
        r_sub,
        alpha=float(resolved["alpha"]),
        degree=_as_odd_degree(int(resolved["degree"])),
        block_encoding_mode="explicit_dense",
        phase_method="pennylane_poly_to_angles",
        seed=int(resolved["seed"]),
    )
    ridge_update = result.ridge_reference.update_vector
    qsvt_update = result.qsvt_unnormalized_vector
    initial_residual_norm = float(np.linalg.norm(r_sub))
    ridge_residual_norm = float(np.linalg.norm(H_sub @ ridge_update - r_sub))
    qsvt_residual_norm = float(np.linalg.norm(H_sub @ qsvt_update - r_sub))
    update_rmse = float(np.sqrt(np.mean((qsvt_update - ridge_update) ** 2)))
    ridge_state, _ = normalize_state(ridge_update)
    qsvt_state, _ = normalize_state(qsvt_update)
    observables = _default_observables(qsvt_state.size)
    observable_rows = []
    for observable in observables:
        ridge_value = _observable_value(ridge_state, observable)
        qsvt_value = _observable_value(qsvt_state, observable)
        observable_rows.append(
            {
                "observable_name": observable["observable_name"],
                "observable_type": observable["observable_type"],
                "ridge_exact_normalized": ridge_value,
                "qsvt_exact_normalized": qsvt_value,
                "absolute_error": abs(qsvt_value - ridge_value),
                "notes": "selected subproblem observable",
            }
        )
    update_row = {
        "case": resolved["case"],
        "matrix_source": sub_meta["matrix_source"],
        "submatrix_size": int(resolved["submatrix_size"]),
        "alpha": float(resolved["alpha"]),
        "requested_degree": int(resolved["degree"]),
        "synthesized_degree": result.synthesized_degree,
        "phase_count": result.phase_count,
        "update_rmse_vs_ridge_reference": update_rmse,
        "phase_aligned_state_l2_error_vs_ridge": result.phase_aligned_state_l2_error,
        "initial_subproblem_residual_norm": initial_residual_norm,
        "ridge_subproblem_residual_norm": ridge_residual_norm,
        "qsvt_subproblem_residual_norm": qsvt_residual_norm,
        "success_probability_proxy": result.success_probability_proxy,
        "limitation": SELECTED_SUBPROBLEM_LIMITATION,
    }
    update_frame = pd.DataFrame([update_row])
    observable_frame = pd.DataFrame(observable_rows)
    diagnostics = result.diagnostics()
    diagnostics.update(update_row)
    diagnostics.update(sub_meta)
    update_path = output_dir / "one_step_update_summary.csv"
    observable_path = output_dir / "one_step_observable_summary.csv"
    diagnostics_path = output_dir / "one_step_state_diagnostics.json"
    report_path = output_dir / "one_step_ac_update_report.md"
    update_frame.to_csv(update_path, index=False)
    observable_frame.to_csv(observable_path, index=False)
    write_json(diagnostics_path, diagnostics)
    report_path.write_text(_one_step_markdown(update_row, observable_frame), encoding="utf-8")
    write_json(
        output_dir / "manifest.json",
        _manifest(resolved, [update_path, observable_path, diagnostics_path, report_path]),
    )
    return {
        "output_dir": output_dir,
        "update_summary": update_frame,
        "observable_summary": observable_frame,
        "diagnostics": diagnostics,
        "artifacts": {
            "one_step_update_summary": update_path,
            "one_step_observable_summary": observable_path,
            "one_step_state_diagnostics": diagnostics_path,
            "one_step_ac_update_report": report_path,
        },
    }


def build_norm_success_outputs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_norm_success",
        "one_step_diagnostics": (
            "outputs/qsvt_one_step_ac_update_demo/one_step_state_diagnostics.json"
        ),
        "shots": [100, 1000, 10000, 100000],
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    diagnostics_path = Path(resolved["one_step_diagnostics"])
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"missing one-step diagnostics: {diagnostics_path}")
    diagnostics = pd.json_normalize(_read_json(diagnostics_path)).iloc[0].to_dict()
    summary_row = {
        "ridge_update_norm": diagnostics.get("ridge_update_norm"),
        "qsvt_state_norm_before_normalization": diagnostics.get(
            "qsvt_state_norm_before_normalization"
        ),
        "qsvt_state_norm_after_normalization": diagnostics.get(
            "qsvt_state_norm_after_normalization"
        ),
        "success_probability_proxy": diagnostics.get("success_probability_proxy"),
        "bounded_filter_scaling_C": diagnostics.get("bounded_target_scaling_C"),
        "beta_matrix_scaling": diagnostics.get("beta"),
        "residual_norm": diagnostics.get("initial_subproblem_residual_norm"),
        "diagnostic_rescaled_update_error": diagnostics.get("state_l2_error_against_ridge"),
        "norm_recovery_status": "classical_simulator_metadata_not_quantum_estimated",
        "limitation": NORM_RECOVERY_LIMITATION,
    }
    shot_rows = [
        estimate_success_probability_from_shots(
            float(summary_row["success_probability_proxy"]),
            int(shots),
            int(resolved["seed"]),
        )
        for shots in resolved["shots"]
    ]
    summary = pd.DataFrame([summary_row])
    shots = pd.DataFrame(shot_rows)
    summary_path = output_dir / "norm_success_summary.csv"
    shots_path = output_dir / "success_probability_shot_summary.csv"
    limitations_path = output_dir / "norm_recovery_limitations.md"
    summary.to_csv(summary_path, index=False)
    shots.to_csv(shots_path, index=False)
    limitations_path.write_text(
        "# Norm Recovery Limitations\n\n" + NORM_RECOVERY_LIMITATION + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "shot_summary": shots,
        "artifacts": {
            "norm_success_summary": summary_path,
            "success_probability_shot_summary": shots_path,
            "norm_recovery_limitations": limitations_path,
        },
    }


def build_full_engineering_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_engineering_report",
        "input_root": "outputs",
        "require_inputs": True,
    }
    if config:
        resolved.update(config)
    input_root = Path(resolved["input_root"])
    required = {
        "one_step": input_root / "qsvt_one_step_ac_update_demo" / "one_step_update_summary.csv",
        "tradeoff": input_root / "qsvt_alpha_degree_tradeoff" / "alpha_degree_summary.csv",
        "scaling": input_root / "full_qsvt_ieee_scaling" / "implemented_simulation_summary.csv",
        "resources": input_root / "full_qsvt_ieee_scaling" / "full_ieee_resource_estimates.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing and bool(resolved["require_inputs"]):
        raise FileNotFoundError(f"missing required pathway inputs: {missing}")
    audit = audit_current_qsvt_pathway()
    update_workflow = build_qsvt_update_workflow_artifacts()
    observables = build_power_observable_outputs()
    block = build_block_encoding_resource_report()
    hardware = build_hardware_resource_report()
    norm = build_norm_success_outputs()
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    capability = audit["capability_matrix"]
    implemented = pd.read_csv(required["scaling"])
    observable = observables["observable_summary"]
    tradeoff = pd.read_csv(required["tradeoff"])
    norm_summary = norm["summary"]
    resource = pd.read_csv(required["resources"])
    claim_support = _claim_support_matrix()

    files = {
        "qsvt_capability_matrix": output_dir / "qsvt_capability_matrix.csv",
        "implemented_qsvt_simulations": output_dir / "implemented_qsvt_simulations.csv",
        "qsvt_observable_readout_summary": output_dir / "qsvt_observable_readout_summary.csv",
        "qsvt_alpha_degree_tradeoff_summary": output_dir / "qsvt_alpha_degree_tradeoff_summary.csv",
        "qsvt_norm_success_summary": output_dir / "qsvt_norm_success_summary.csv",
        "qsvt_resource_summary": output_dir / "qsvt_resource_summary.csv",
        "qsvt_claim_support_matrix": output_dir / "qsvt_claim_support_matrix.csv",
        "full_qsvt_ieee_engineering_report": output_dir / "full_qsvt_ieee_engineering_report.md",
        "manifest": output_dir / "manifest.json",
    }
    capability.to_csv(files["qsvt_capability_matrix"], index=False)
    implemented.to_csv(files["implemented_qsvt_simulations"], index=False)
    observable.to_csv(files["qsvt_observable_readout_summary"], index=False)
    tradeoff.to_csv(files["qsvt_alpha_degree_tradeoff_summary"], index=False)
    norm_summary.to_csv(files["qsvt_norm_success_summary"], index=False)
    resource.to_csv(files["qsvt_resource_summary"], index=False)
    claim_support.to_csv(files["qsvt_claim_support_matrix"], index=False)
    files["full_qsvt_ieee_engineering_report"].write_text(
        _engineering_report_markdown(
            capability=capability,
            implemented=implemented,
            observable=observable,
            tradeoff=tradeoff,
            norm_summary=norm_summary,
            resource=resource,
        ),
        encoding="utf-8",
    )
    write_json(
        files["manifest"],
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "claim_boundary": PATHWAY_CLAIM_BOUNDARY,
            "files_generated": {key: str(path) for key, path in files.items()},
            "upstream_outputs": {key: str(path) for key, path in required.items()},
            "support_outputs": {
                "audit": str(audit["output_dir"]),
                "update_workflow": str(update_workflow["output_dir"]),
                "observables": str(observables["output_dir"]),
                "block_encoding": str(block["output_dir"]),
                "hardware": str(hardware["output_dir"]),
                "norm_success": str(norm["output_dir"]),
            },
        },
    )
    return {
        "output_dir": output_dir,
        "artifacts": files,
        "capability_matrix": capability,
        "implemented_simulations": implemented,
        "resource_summary": resource,
    }


def _capability_rows() -> list[dict[str, Any]]:
    def exists(path: str) -> bool:
        return Path(path).is_file()

    return [
        _capability(
            "scalar QSP phase validation",
            exists("src/robust_qsvt_se/qsvt/qsp_validation.py"),
            "src/robust_qsvt_se/qsvt/qsp_validation.py; src/robust_qsvt_se/qsvt/phase_synthesis.py",
            exists("tests/test_qsvt_phase_validation.py"),
            "scalar response only",
        ),
        _capability(
            "explicit dense block encoding",
            exists("src/robust_qsvt_se/qsvt/block_encoding.py"),
            "src/robust_qsvt_se/qsvt/block_encoding.py",
            exists("tests/test_block_encoding.py"),
            "small dense matrices only",
        ),
        _capability(
            "matrix-level QSVT",
            exists("src/robust_qsvt_se/qsvt/full_matrix_qsvt_demo.py"),
            "src/robust_qsvt_se/qsvt/full_matrix_qsvt_demo.py",
            exists("tests/test_full_matrix_qsvt_demo.py"),
            "small square power-of-two submatrices",
        ),
        _capability(
            "partial-observable readout",
            exists("src/robust_qsvt_se/qsvt/partial_readout_demo.py"),
            (
                "src/robust_qsvt_se/qsvt/partial_observable_readout.py; "
                "src/robust_qsvt_se/qsvt/partial_readout_demo.py"
            ),
            exists("tests/test_partial_observable_readout.py"),
            "simulator shot proxies only",
        ),
        _capability(
            "residual state preparation",
            exists("src/robust_qsvt_se/qsvt/qsvt_update_workflow.py"),
            "src/robust_qsvt_se/qsvt/qsvt_update_workflow.py",
            exists("tests/test_qsvt_update_workflow.py"),
            "amplitude-loading circuit not synthesized",
        ),
        _capability(
            "success probability tracking",
            exists("src/robust_qsvt_se/qsvt/norm_success.py"),
            "src/robust_qsvt_se/qsvt/norm_success.py",
            exists("tests/test_norm_success.py"),
            "proxy probability from simulator state",
        ),
        _capability(
            "norm recovery",
            exists("src/robust_qsvt_se/qsvt/norm_success.py"),
            "src/robust_qsvt_se/qsvt/norm_success.py",
            exists("tests/test_norm_success.py"),
            "not quantum-estimated unless future amplitude estimation is added",
        ),
        _capability(
            "bus/state observable mapping",
            exists("src/robust_qsvt_se/qsvt/state_metadata.py"),
            (
                "src/robust_qsvt_se/qsvt/state_metadata.py; "
                "src/robust_qsvt_se/qsvt/power_observables.py"
            ),
            exists("tests/test_state_metadata.py"),
            "depends on AC metadata availability",
        ),
        _capability(
            "scalable block-encoding model",
            exists("src/robust_qsvt_se/qsvt/scalable_block_encoding.py"),
            "src/robust_qsvt_se/qsvt/scalable_block_encoding.py",
            exists("tests/test_scalable_block_encoding.py"),
            "resource model only, no oracle synthesis",
        ),
        _capability(
            "IEEE-scale resource estimate",
            exists("scripts/run_full_qsvt_ieee_scaling_study.py"),
            (
                "scripts/run_full_qsvt_ieee_scaling_study.py; "
                "src/robust_qsvt_se/qsvt/hardware_resource_estimator.py"
            ),
            exists("tests/test_full_qsvt_ieee_report.py"),
            "resource estimates, not full hardware execution",
        ),
    ]


def _capability(
    capability: str,
    exists: bool,
    evidence: str,
    tested: bool,
    limitation: str,
) -> dict[str, Any]:
    return {
        "Capability": capability,
        "Exists?": bool(exists),
        "File evidence": evidence if exists else "",
        "Tested?": bool(tested),
        "Limitation": limitation,
    }


def _metadata_observables(
    H_tilde: np.ndarray,
    system_metadata: dict[str, Any],
    metadata: Any,
) -> list[dict[str, Any]]:
    angle_buses = list(system_metadata.get("angle_state_buses", []))
    voltage_buses = list(system_metadata.get("voltage_state_buses", []))
    observables: list[dict[str, Any]] = []
    if angle_buses:
        observables.append(bus_angle_component(int(angle_buses[0]), metadata))
    if voltage_buses:
        observables.append(bus_voltage_component(int(voltage_buses[0]), metadata))
    if len(angle_buses) >= 2:
        observables.append(
            branch_angle_difference(int(angle_buses[0]), int(angle_buses[1]), metadata)
        )
    if angle_buses or voltage_buses:
        bus_ids = list(
            dict.fromkeys(
                [*(int(bus) for bus in angle_buses[:2]), *(int(bus) for bus in voltage_buses[:1])]
            )
        )
        observables.append(area_update_energy(bus_ids, metadata))
    row_meta = {
        "label": system_metadata.get("measurement_labels", ["jacobian_row_0"])[0],
        "measurement_type": system_metadata.get("measurement_types", ["unknown"])[0],
        "buses": system_metadata.get("measurement_buses", [[]])[0],
    }
    observables.append(jacobian_row_observable(0, H_tilde, row_meta))
    return observables


def _observable_definition_row(observable: dict[str, Any]) -> dict[str, Any]:
    return {
        "observable_name": observable["observable_name"],
        "observable_type": observable["observable_type"],
        "indices": observable.get("indices", []),
        "power_system_quantity": observable.get("power_system_quantity", ""),
        "bus_id": observable.get("bus_id", ""),
        "from_bus": observable.get("from_bus", ""),
        "to_bus": observable.get("to_bus", ""),
        "notes": observable.get("notes", ""),
    }


def _observable_exact_comparison(
    observables: list[dict[str, Any]],
    ridge_state: np.ndarray,
    qsvt_state: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for observable in observables:
        ridge_value = _observable_value(ridge_state, observable)
        qsvt_value = _observable_value(qsvt_state, observable)
        rows.append(
            {
                "observable_name": observable["observable_name"],
                "observable_type": observable["observable_type"],
                "ridge_exact_normalized": ridge_value,
                "qsvt_target_proxy_exact_normalized": qsvt_value,
                "absolute_error": abs(qsvt_value - ridge_value),
                "notes": (
                    "Full-vector observable comparison uses the exact Ridge/QSVT target "
                    "reference on the full weighted Jacobian; small explicit QSVT readout "
                    "evidence is reported separately."
                ),
            }
        )
    return pd.DataFrame(rows)


def _observable_shot_frame(
    observables: list[dict[str, Any]],
    qsvt_state: np.ndarray,
    ridge_state: np.ndarray,
    *,
    shots: list[int],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for observable in observables:
        true_value = _observable_value(qsvt_state, observable)
        ridge_value = _observable_value(ridge_state, observable)
        for shot_count in shots:
            if observable["observable_type"] in {"basis_probability", "subset_probability"}:
                successes = int(rng.binomial(int(shot_count), float(np.clip(true_value, 0, 1))))
                estimate = successes / float(shot_count)
                standard_error = float(
                    np.sqrt(max(0.0, true_value * (1.0 - true_value)) / float(shot_count))
                )
            else:
                target = float(np.clip(true_value, -1.0, 1.0))
                plus = (1.0 + target) / 2.0
                successes = int(rng.binomial(int(shot_count), plus))
                estimate = 2.0 * successes / float(shot_count) - 1.0
                standard_error = float(np.sqrt(max(0.0, 1.0 - target**2) / float(shot_count)))
            rows.append(
                {
                    "observable_name": observable["observable_name"],
                    "observable_type": observable["observable_type"],
                    "shots": int(shot_count),
                    "seed": int(seed),
                    "true_qsvt_value": true_value,
                    "shot_estimate": estimate,
                    "standard_error": standard_error,
                    "absolute_sampling_error": abs(estimate - true_value),
                    "ridge_reference_value": ridge_value,
                    "absolute_error_vs_ridge": abs(estimate - ridge_value),
                    "notes": "shot proxy for selected observable only",
                }
            )
    return pd.DataFrame(rows)


def _observable_value(state: np.ndarray, observable: dict[str, Any]) -> float:
    obs_type = str(observable["observable_type"])
    if obs_type == "basis_probability":
        return basis_probability(state, int(observable["indices"][0]))
    if obs_type == "subset_probability":
        return subset_probability(state, list(observable["indices"]))
    if obs_type == "linear_overlap_real":
        return float(
            np.real(linear_functional_overlap(state, np.asarray(observable["coefficients"])))
        )
    raise ValueError(f"unsupported observable type: {obs_type}")


def _default_observables(dimension: int) -> list[dict[str, Any]]:
    observables = [component_observable(0, "component_0_probability")]
    if dimension > 1:
        observables.append(component_observable(1, "component_1_probability"))
        observables.append(subset_energy_observable([0, 1], "first_two_state_energy"))
        observables.append(difference_observable(0, 1, dimension, "difference_0_1_overlap"))
    return observables


def _ridge_update(H_tilde: np.ndarray, r_tilde: np.ndarray, *, alpha: float) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(
        np.asarray(H_tilde, dtype=np.float64), full_matrices=False
    )
    return Vt.T @ (ridge_filter(singular_values, alpha=float(alpha)) * (U.T @ r_tilde))


def _bounded_filter_error(B: np.ndarray, alpha: float, degree: int) -> tuple[float, float]:
    singular_values = np.linalg.svd(B, compute_uv=False)
    beta = max(float(singular_values[0]), np.finfo(float).eps)
    normalized = singular_values / beta
    alpha_norm = alpha / beta**2
    positive = normalized[normalized > 1.0e-14]
    domain_min = max(1.0e-6, min(0.95, 0.9 * float(np.min(positive))))
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha_norm,
        block_encoding_normalization=1.0,
        degree=_as_odd_degree(degree),
        domain_min=domain_min,
        domain_max=1.0,
        grid_size=max(512, degree + 2),
    )
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients) / approximation.scale_factor
    )
    target = normalized / (approximation.scale_factor * (normalized**2 + alpha_norm))
    return float(np.max(np.abs(polynomial(normalized) - target))), float(approximation.scale_factor)


def _max_default_observable_error(ridge_state: np.ndarray, qsvt_state: np.ndarray) -> float:
    return float(
        max(
            abs(_observable_value(qsvt_state, obs) - _observable_value(ridge_state, obs))
            for obs in _default_observables(ridge_state.size)
        )
    )


def _tradeoff_observable_rows(
    row: dict[str, Any],
    ridge_state: np.ndarray,
    qsvt_state: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for observable in _default_observables(ridge_state.size):
        ridge_value = _observable_value(ridge_state, observable)
        qsvt_value = _observable_value(qsvt_state, observable)
        rows.append(
            {
                "case": row["case"],
                "submatrix_size": row["submatrix_size"],
                "alpha": row["alpha"],
                "degree": row["requested_degree"],
                "observable_name": observable["observable_name"],
                "ridge_exact_normalized": ridge_value,
                "qsvt_exact_normalized": qsvt_value,
                "absolute_error": abs(qsvt_value - ridge_value),
            }
        )
    return rows


def _as_odd_degree(degree: int) -> int:
    value = int(degree)
    if value <= 0:
        raise ValueError("degree must be positive")
    return value if value % 2 == 1 else value - 1


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values)[np.asarray(singular_values) > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(config: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    return {
        "generated_at": utc_timestamp(),
        "command": current_command(),
        "git_commit": git_commit(),
        "input_config": config,
        "files_generated": [str(path) for path in files],
        "claim_boundary": PATHWAY_CLAIM_BOUNDARY,
    }


def _audit_markdown(frame: pd.DataFrame) -> str:
    implemented = int(frame["Exists?"].sum())
    return "\n".join(
        [
            "# Full QSVT IEEE Pathway Audit",
            "",
            PATHWAY_CLAIM_BOUNDARY,
            "",
            f"- Capabilities present: {implemented} / {len(frame)}.",
            "- Scalar phase validation, dense block encoding, matrix-level QSVT, and "
            "partial readout are separated from resource-estimate-only rows.",
            "- Dense simulations remain small selected subproblems; IEEE-scale rows are "
            "resource estimates.",
            "",
        ]
    )


def _observable_markdown(
    config: dict[str, Any],
    matrix_source: str,
    metadata: Any,
    exact: pd.DataFrame,
    shots: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Power-System Observable Definitions",
            "",
            PATHWAY_CLAIM_BOUNDARY,
            "",
            f"- Case: {config['case_name']}",
            f"- Matrix source: {matrix_source}",
            f"- State ordering: {metadata.ordering_note}",
            f"- Observables: {len(exact)}",
            f"- Max exact target/reference observable error: {exact['absolute_error'].max():.6g}",
            f"- Max shot sampling error: {shots['absolute_sampling_error'].max():.6g}",
            "",
            "The full-vector exact rows use the Ridge/QSVT target equivalence as a "
            "reference proxy. Small explicit QSVT readout evidence is kept in the "
            "partial-observable demo outputs.",
            "",
        ]
    )


def _alpha_degree_markdown(summary: pd.DataFrame) -> str:
    ok = summary[summary["status"] == "ok"]
    best = float(ok["qsvt_state_error_vs_ridge"].min()) if not ok.empty else float("nan")
    grouped = ok.groupby("alpha", as_index=False)["maximum_pointwise_filter_error"].min()
    lines = [
        "# QSVT Alpha-Degree Tradeoff",
        "",
        PATHWAY_CLAIM_BOUNDARY,
        "",
        f"- Rows: {len(summary)}",
        f"- Best state error vs Ridge reference: {best:.6g}",
        "",
        "The sweep uses bounded odd-polynomial QSVT target proxies for tractable "
        "tradeoff accounting. Smaller alpha generally increases the filter peak and "
        "can require higher degree for the same approximation tolerance.",
        "",
        "## Best pointwise error by alpha",
    ]
    for _, row in grouped.iterrows():
        lines.append(f"- alpha={row['alpha']:.1e}: {row['maximum_pointwise_filter_error']:.6g}")
    lines.append("")
    return "\n".join(lines)


def _scaling_markdown(
    simulation: pd.DataFrame,
    resource: pd.DataFrame,
    matrix: pd.DataFrame,
) -> str:
    executed = simulation[simulation["status"] == "ok"]
    return "\n".join(
        [
            "# Full QSVT IEEE Scaling Study",
            "",
            PATHWAY_CLAIM_BOUNDARY,
            "",
            f"- IEEE cases attempted: {', '.join(matrix['case'].astype(str))}",
            f"- Explicit dense simulations executed: {len(executed)}",
            f"- Resource estimate rows: {len(resource)}",
            "",
            "Implemented simulation rows are actual small explicit dense QSVT runs. "
            "Other rows are resource estimates or skipped by runtime guards.",
            "",
        ]
    )


def _one_step_markdown(row: dict[str, Any], observable: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# One-Step AC Update QSVT Subproblem Demo",
            "",
            PATHWAY_CLAIM_BOUNDARY,
            "",
            f"- Case: {row['case']}",
            f"- Submatrix size: {row['submatrix_size']}",
            f"- Update RMSE vs Ridge reference: {row['update_rmse_vs_ridge_reference']:.6g}",
            f"- Phase-aligned state error: {row['phase_aligned_state_l2_error_vs_ridge']:.6g}",
            f"- Max observable error: {observable['absolute_error'].max():.6g}",
            "",
            f"> {SELECTED_SUBPROBLEM_LIMITATION}",
            "",
        ]
    )


def _claim_support_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "small explicit matrix-level QSVT simulation",
                "support": "full_matrix_qsvt_demo and one-step selected subproblem outputs",
                "status": "supported for small selected subproblems",
            },
            {
                "claim": "IEEE-scale QSVT hardware execution",
                "support": "none",
                "status": "not claimed",
            },
            {
                "claim": "QSVT numerical superiority over Ridge/Tikhonov",
                "support": "none; same spectral target is compared",
                "status": "not claimed",
            },
            {
                "claim": "scalable block-encoding resource model",
                "support": "sparse/qRAM/LCU resource estimates",
                "status": "resource model only",
            },
            {
                "claim": "partial-observable readout workflow",
                "support": "partial readout and observable outputs",
                "status": "simulator-supported for selected observables",
            },
        ]
    )


def _engineering_report_markdown(
    *,
    capability: pd.DataFrame,
    implemented: pd.DataFrame,
    observable: pd.DataFrame,
    tradeoff: pd.DataFrame,
    norm_summary: pd.DataFrame,
    resource: pd.DataFrame,
) -> str:
    executed = implemented[implemented["status"] == "ok"]
    best_tradeoff = tradeoff["qsvt_state_error_vs_ridge"].min()
    return "\n".join(
        [
            "# Full QSVT IEEE Engineering Report",
            "",
            "## Scope",
            PATHWAY_CLAIM_BOUNDARY,
            "",
            "## What Is Implemented",
            f"- Capability rows present: {int(capability['Exists?'].sum())} / {len(capability)}.",
            f"- Explicit dense QSVT simulations executed in scaling study: {len(executed)}.",
            "",
            "## What Is Simulated Exactly",
            "Small selected square subproblems use explicit dense block encodings and "
            "PennyLane QSVT matrices.",
            "",
            "## What Is Estimated",
            "IEEE-scale rows use sparse-access/qRAM/LCU resource models and hardware "
            "proxy counts, without constructing full dense unitaries.",
            "",
            "## IEEE-Derived Matrices Used",
            f"- Resource cases: {', '.join(resource['case'].astype(str).unique())}.",
            "",
            "## QSVT Target Filter",
            "The target is the bounded Ridge/Tikhonov singular-value filter; direct "
            "Ridge/Tikhonov SVD remains the classical reference.",
            "",
            "## Residual State Preparation",
            "Residual vectors are explicitly normalized in the small workflow; amplitude "
            "state preparation is otherwise an oracle assumption.",
            "",
            "## Block-Encoding Models",
            "Dense explicit block encoding is used for small simulations. Sparse-access, "
            "qRAM row-state, and LCU entries are resource models.",
            "",
            "## QSVT Update-State Results",
            f"- Best alpha-degree state error row: {best_tradeoff:.6g}.",
            "",
            "## Partial-Observable Readout",
            f"- Observable rows: {len(observable)}.",
            f"- Max observable target/reference error: {observable['absolute_error'].max():.6g}.",
            "",
            "## Alpha-Degree Tradeoff",
            "The sweep reports filter approximation, state error, success proxy, query "
            "count, and observable error by alpha and degree.",
            "",
            "## Success Probability and Norm Recovery",
            f"- Norm recovery status: {norm_summary['norm_recovery_status'].iloc[0]}.",
            f"- Limitation: {NORM_RECOVERY_LIMITATION}",
            "",
            "## IEEE-Scale Resource Estimates",
            "Resource estimates include matrix dimensions, qubits, phase counts, query "
            "counts, state-preparation assumptions, and block-encoding assumptions.",
            "",
            "## Limitations",
            "- No quantum speedup or advantage is claimed.",
            "- No full IEEE-scale hardware-native QSVT circuit is executed.",
            "- Norm recovery is not a quantum-estimated routine in this package.",
            "- Full-vector readout is not assumed as the target output model.",
            "",
            "## Safe Manuscript Wording",
            "We implement a readout-aware QSVT state-estimation workflow with small "
            "explicit block-encoded simulations on IEEE-derived weighted-Jacobian "
            "subproblems and IEEE-scale resource estimates under sparse-access and "
            "related oracle assumptions. The workflow is compared against the "
            "Ridge/Tikhonov SVD reference and supports selected-observable readout "
            "studies, but it does not demonstrate quantum speedup or full IEEE-scale "
            "hardware execution.",
            "",
        ]
    )
