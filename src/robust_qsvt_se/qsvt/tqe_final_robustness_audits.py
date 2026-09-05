from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    FINAL_ROBUSTNESS_AUDITS_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    bounded_ridge_target,
    fit_bounded_ridge_polynomial,
    load_sweep_subproblem,
    qsvt_odd_degree,
)
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import (
    fit_actual_singular_interpolating_polynomial,
    ridge_update_svd,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import construct_padded_block_encoding
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    DEFAULT_BASIS_GATES,
    run_ieee_selected_block,
    synthesize_qsvt_phases,
)
from robust_qsvt_se.qsvt.tqe_nonlinear_ac_per_iteration_feasibility import (
    degree_feasibility_by_epsilon,
)
from robust_qsvt_se.qsvt.tqe_sparse_oracle_block_encoding_model import (
    SparseJacobianOracle,
    ceil_log2,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.seed import make_rng

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL_TOL = 1.0e-15

PHASE_AUDIT_COLUMNS = [
    "difficulty",
    "case_name",
    "subproblem_size",
    "alpha",
    "epsilon_target",
    "requested_degree",
    "used_degree",
    "polynomial_parity",
    "target_bounded_status",
    "max_error_dense_grid",
    "max_error_actual_singular_values",
    "phase_synthesis_status",
    "phase_count",
    "synthesis_runtime_seconds",
    "phase_residual_or_internal_error",
    "failure_mode",
    "notes",
]

SIGNED_READOUT_COLUMNS = [
    "observable_name",
    "observable_type",
    "coordinate_indices",
    "metadata_label",
    "ridge_signed_value",
    "qsvt_statevector_signed_value",
    "phase_aware_estimate",
    "abs_error_vs_ridge",
    "rel_error_vs_ridge",
    "sign_access_model",
    "basis_sampling_accessible",
    "phase_sign_access_required",
    "shot_count",
    "ci95_lower",
    "ci95_upper",
    "simulation_status",
    "failure_or_skip_reason",
]

NOISE_COLUMNS = [
    "noise_model",
    "p1",
    "p2",
    "readout_error",
    "shots",
    "success_probability_estimate",
    "observable_error_vs_ideal",
    "update_error_proxy_vs_ideal",
    "residual_gap_proxy",
    "distribution_total_variation_distance",
    "noisy_counts_available",
    "simulation_status",
    "failure_or_skip_reason",
]

PQ_COLUMNS = [
    "case_name",
    "row_set",
    "measurement_setting",
    "rows",
    "states",
    "redundancy",
    "rank",
    "sigma_min_nonzero",
    "sigma_max",
    "condition_number",
    "nnz",
    "density",
    "max_row_sparsity",
    "pinv_rmse",
    "ridge_rmse",
    "tsvd_rmse",
    "huber_rmse",
    "status",
    "failure_or_skip_reason",
]

TINY_ORACLE_COLUMNS = [
    "matrix_label",
    "matrix_shape",
    "row_qubits",
    "ell_qubits",
    "column_qubits",
    "total_qubits",
    "num_truth_table_rows",
    "padding_encoded_column",
    "circuit_constructed",
    "truth_table_passed",
    "unitarity_error_fro",
    "transpilation_status",
    "transpiled_depth",
    "transpiled_cx_count",
    "transpiled_total_ops",
    "simulation_status",
    "failure_or_skip_reason",
]

REPEAT_COLUMNS = [
    "case_name",
    "subproblem_size",
    "alpha",
    "epsilon_target",
    "degree",
    "phase_synthesis_status",
    "qsvt_sequence_status",
    "simulation_status",
    "transform_block_error_fro",
    "circuit_vs_polynomial_fro_error",
    "relative_update_error",
    "residual_gap",
    "success_probability",
    "raw_circuit_depth",
    "transpilation_status",
    "transpiled_depth",
    "transpiled_cx_count",
    "failure_or_skip_reason",
]

ALPHA_COLUMNS = [
    "case_name",
    "stress_setting",
    "alpha",
    "rmse",
    "residual_norm",
    "update_norm",
    "condition_number",
    "epsilon_target",
    "required_degree",
    "target_met",
    "chosen_alpha_by_rule",
    "rule_name",
    "rule_status",
    "notes",
]

FINAL_SUMMARY_COLUMNS = [
    "sub_experiment",
    "status",
    "rows",
    "completed_rows",
    "failed_rows",
    "skipped_rows",
    "key_metric",
    "main_paper_recommendation",
    "supplement_recommendation",
    "claim_boundary_note",
]


@dataclass(frozen=True, slots=True)
class UpdateComponents:
    ridge: np.ndarray
    qsvt: np.ndarray
    metadata: dict[str, Any]


def run_final_robustness_audits(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / FINAL_ROBUSTNESS_AUDITS_DIR)
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    phase = phase_synthesis_hard_case_audit(resolved)
    signed = signed_phase_aware_readout_diagnostic(resolved)
    noise = noise_sensitivity_integrated_qsvt(resolved)
    pq = reactive_pq_row_composition_ablation(resolved)
    oracle = tiny_reversible_sparse_oracle_lookup(resolved)
    repeat = integrated_qsvt_repeat_case(resolved)
    alpha = alpha_selection_diagnostic(resolved)
    final_summary = final_robustness_summary(
        phase=phase,
        signed=signed,
        noise=noise,
        pq=pq,
        oracle=oracle,
        repeat=repeat,
        alpha=alpha,
    )

    artifacts = _write_all_outputs(
        config=resolved,
        output_dir=output_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        phase=phase,
        signed=signed,
        noise=noise,
        pq=pq,
        oracle=oracle,
        repeat=repeat,
        alpha=alpha,
        final_summary=final_summary,
    )

    manifest_path = output_dir / "final_robustness_audit_manifest.json"
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
            "qiskit_version": _version_or_none("qiskit"),
            "qiskit_aer_available": _qiskit_aer_available(),
            "pennylane_version": _version_or_none("pennylane"),
            "pyqsp_version": _version_or_none("pyqsp"),
            "input_artifact_paths": _input_artifact_paths(resolved),
            "success_failure_skipped_counts": _status_rollup(final_summary),
        }
    )
    write_json(manifest_path, metadata)
    artifacts["final_robustness_manifest"] = manifest_path
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: path for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "phase": phase,
        "signed": signed,
        "noise": noise,
        "pq": pq,
        "oracle": oracle,
        "repeat": repeat,
        "alpha": alpha,
        "summary": final_summary,
        "artifacts": artifacts,
    }


def phase_synthesis_hard_case_audit(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting in config["phase_audit_settings"]:
        row = _phase_audit_one(setting, config)
        rows.append(row)
    return pd.DataFrame(rows, columns=PHASE_AUDIT_COLUMNS)


def _phase_audit_one(setting: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "difficulty": str(setting["difficulty"]),
        "case_name": str(setting["case_name"]),
        "subproblem_size": int(setting["subproblem_size"]),
        "alpha": float(setting["alpha"]),
        "epsilon_target": float(setting["epsilon_target"]),
        "requested_degree": int(setting["requested_degree"]),
        "used_degree": np.nan,
        "polynomial_parity": "odd",
        "target_bounded_status": "not_evaluated",
        "max_error_dense_grid": np.nan,
        "max_error_actual_singular_values": np.nan,
        "phase_synthesis_status": "not_started",
        "phase_count": 0,
        "synthesis_runtime_seconds": 0.0,
        "phase_residual_or_internal_error": np.nan,
        "failure_mode": "not_started",
        "notes": "",
    }
    try:
        subproblem = load_sweep_subproblem(
            {
                "case_name": setting["case_name"],
                "subproblem_size": setting["subproblem_size"],
                "selection_mode": setting.get("selection_mode", "high_leverage"),
            },
            seed=int(config["seed"]),
        )
        H = np.asarray(subproblem.H_tilde, dtype=np.float64)
        singular_values = np.linalg.svd(H, compute_uv=False)
        gamma = float(np.max(singular_values[singular_values > 1.0e-14]))
        degree, adjustment = qsvt_odd_degree(int(setting["requested_degree"]))
        cheb, _, C_alpha = fit_bounded_ridge_polynomial(
            alpha=float(setting["alpha"]),
            beta=gamma,
            degree=degree,
        )
        dense_grid = np.linspace(0.0, 1.0, int(config["dense_grid_size"]))
        dense_target = bounded_ridge_target(
            dense_grid,
            alpha=float(setting["alpha"]),
            beta=gamma,
            C_alpha=C_alpha,
        )
        dense_error = float(np.max(np.abs(cheb(dense_grid) - dense_target)))
        actual_grid = singular_values / gamma
        actual_target = bounded_ridge_target(
            actual_grid,
            alpha=float(setting["alpha"]),
            beta=gamma,
            C_alpha=C_alpha,
        )
        actual_error = float(np.max(np.abs(cheb(actual_grid) - actual_target)))
        unit_grid = np.linspace(-1.0, 1.0, max(4097, degree * 16 + 1))
        bounded = bool(np.max(np.abs(cheb(unit_grid))) <= 1.0 + float(config["bounded_tol"]))
        base.update(
            {
                "used_degree": degree,
                "target_bounded_status": "bounded" if bounded else "admissibility_failure",
                "max_error_dense_grid": dense_error,
                "max_error_actual_singular_values": actual_error,
                "notes": adjustment,
            }
        )
        if not bounded:
            base.update(
                {
                    "phase_synthesis_status": "skipped_admissibility_failure",
                    "failure_mode": "admissibility_failure",
                    "synthesis_runtime_seconds": time.perf_counter() - started,
                }
            )
            return base
        if degree > int(config["max_phase_synthesis_degree"]):
            base.update(
                {
                    "phase_synthesis_status": "skipped_by_budget",
                    "failure_mode": "skipped_by_budget",
                    "synthesis_runtime_seconds": time.perf_counter() - started,
                    "notes": (
                        f"{base['notes']} degree={degree} exceeds "
                        f"max_phase_synthesis_degree={config['max_phase_synthesis_degree']}"
                    ).strip(),
                }
            )
            return base
        monomial = cheb.convert(kind=Polynomial).coef
        monomial = _pad_odd_coefficients(monomial, degree)
        phase_result = synthesize_qsvt_phases(
            monomial,
            angle_solver=str(config["phase_angle_solver"]),
        )
        runtime = time.perf_counter() - started
        if phase_result.status == "completed":
            base.update(
                {
                    "phase_synthesis_status": "completed",
                    "phase_count": int(phase_result.phases.size),
                    "synthesis_runtime_seconds": runtime,
                    "phase_residual_or_internal_error": 0.0,
                    "failure_mode": "success",
                }
            )
        else:
            base.update(
                {
                    "phase_synthesis_status": "failed",
                    "phase_count": int(phase_result.phases.size),
                    "synthesis_runtime_seconds": runtime,
                    "failure_mode": "numerical_failure",
                    "notes": phase_result.failure_reason,
                }
            )
        return base
    except Exception as exc:
        base.update(
            {
                "phase_synthesis_status": "failed",
                "synthesis_runtime_seconds": time.perf_counter() - started,
                "failure_mode": "numerical_failure",
                "notes": f"{type(exc).__name__}: {exc}",
            }
        )
        return base


def signed_phase_aware_readout_diagnostic(config: dict[str, Any]) -> pd.DataFrame:
    try:
        components = load_update_components(config)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    **{column: np.nan for column in SIGNED_READOUT_COLUMNS},
                    "observable_name": "signed_readout_input_unavailable",
                    "observable_type": "failure_record",
                    "basis_sampling_accessible": False,
                    "phase_sign_access_required": True,
                    "simulation_status": "failed_input_unavailable",
                    "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
                }
            ],
            columns=SIGNED_READOUT_COLUMNS,
        )
    ridge = components.ridge
    qsvt = components.qsvt
    i = int(np.argmax(np.abs(ridge)))
    if ridge.size > 1:
        order = np.argsort(-np.abs(ridge))
        j = int(order[1])
    else:
        j = i
    specs = [
        ("signed_coordinate_value", "signed_coordinate", (i,)),
        ("signed_coordinate_contrast", "signed_coordinate_contrast", (i, j)),
        ("signed_branch_coordinate_proxy", "signed_branch_coordinate_proxy", (i, j)),
    ]
    rows = []
    for name, obs_type, indices in specs:
        ridge_value = _signed_value(ridge, indices)
        qsvt_value = _signed_value(qsvt, indices)
        error = abs(qsvt_value - ridge_value)
        rows.append(
            {
                "observable_name": name,
                "observable_type": obs_type,
                "coordinate_indices": " ".join(str(index) for index in indices),
                "metadata_label": _signed_metadata_label(indices, components.metadata, obs_type),
                "ridge_signed_value": ridge_value,
                "qsvt_statevector_signed_value": qsvt_value,
                "phase_aware_estimate": qsvt_value,
                "abs_error_vs_ridge": error,
                "rel_error_vs_ridge": error / max(abs(ridge_value), SMALL_TOL),
                "sign_access_model": (
                    "statevector phase-aware diagnostic from signed update amplitudes; "
                    "not ordinary computational-basis sampling"
                ),
                "basis_sampling_accessible": False,
                "phase_sign_access_required": True,
                "shot_count": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "simulation_status": "completed_statevector_phase_aware",
                "failure_or_skip_reason": "",
            }
        )
    return pd.DataFrame(rows, columns=SIGNED_READOUT_COLUMNS)


def noise_sensitivity_integrated_qsvt(config: dict[str, Any]) -> pd.DataFrame:
    try:
        components = load_update_components(config)
    except Exception as exc:
        return pd.DataFrame(
            [_noise_failure_row(f"input unavailable: {type(exc).__name__}: {exc}")],
            columns=NOISE_COLUMNS,
        )
    qsvt = components.qsvt
    probabilities = normalized_probabilities(qsvt)
    ideal_success = float(config["integrated_success_probability"])
    selected = int(np.argmax(probabilities))
    rows: list[dict[str, Any]] = []
    aer_available = _qiskit_aer_available()
    aer_context: dict[str, Any] | None = None
    aer_fallback_reason = ""
    if aer_available and bool(config["use_aer_if_available"]):
        try:
            aer_context = _build_aer_noise_context(config=config, qsvt_update=qsvt)
            selected = int(np.argmax(aer_context["ideal_conditional_distribution"]))
        except Exception as exc:  # pragma: no cover - qiskit-version dependent
            aer_fallback_reason = (
                f"aer circuit construction failed: {type(exc).__name__}: {exc}; "
                "used deterministic distribution-noise proxy"
            )
    for noise in config["noise_models"]:
        for shots in config["noise_shots"]:
            if aer_context is not None:
                try:
                    rows.append(
                        _aer_noise_row(
                            context=aer_context,
                            noise_spec=noise,
                            shots=int(shots),
                            seed=int(config["seed"]) + int(shots) + len(rows),
                        )
                    )
                    continue
                except Exception as exc:  # pragma: no cover - qiskit-version dependent
                    aer_fallback_reason = (
                        f"aer simulation failed: {type(exc).__name__}: {exc}; "
                        "used deterministic distribution-noise proxy"
                    )
            rows.append(
                _proxy_noise_row(
                    qsvt=qsvt,
                    probabilities=probabilities,
                    selected=selected,
                    ideal_success=ideal_success,
                    noise_spec=noise,
                    shots=int(shots),
                    seed=int(config["seed"]) + int(shots) + len(rows),
                    depth=int(config["integrated_raw_depth"]),
                    cx_count=int(config["integrated_cx_count"]),
                    failure_reason=_noise_proxy_reason(
                        aer_available=aer_available,
                        use_aer=bool(config["use_aer_if_available"]),
                        fallback_reason=aer_fallback_reason,
                    ),
                )
            )
    return pd.DataFrame(rows, columns=NOISE_COLUMNS)


def reactive_pq_row_composition_ablation(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_name in config["pq_cases"]:
        for row_set in config["pq_row_sets"]:
            try:
                rows.append(_pq_one_case(case_name, row_set, config))
            except Exception as exc:
                rows.append(_pq_failure_row(case_name, row_set, exc))
    return pd.DataFrame(rows, columns=PQ_COLUMNS)


def build_measurement_config_for_row_set(row_set: str) -> dict[str, Any]:
    base = {
        "include_voltage_magnitudes": True,
        "include_p_injections": False,
        "include_q_injections": False,
        "include_p_branch_flows": False,
        "include_q_branch_flows": False,
        "voltage_std": 0.01,
        "injection_p_std": 0.03,
        "injection_q_std": 0.03,
        "flow_p_std": 0.02,
        "flow_q_std": 0.02,
    }
    mapping = {
        "V_only": {},
        "V_plus_P_injection": {"include_p_injections": True},
        "V_plus_Q_injection": {"include_q_injections": True},
        "V_plus_PQ_injection": {"include_p_injections": True, "include_q_injections": True},
        "V_plus_P_branch_flow": {"include_p_branch_flows": True},
        "V_plus_Q_branch_flow": {"include_q_branch_flows": True},
        "V_plus_PQ_branch_flow": {
            "include_p_branch_flows": True,
            "include_q_branch_flows": True,
        },
        "full_AC": {
            "include_p_injections": True,
            "include_q_injections": True,
            "include_p_branch_flows": True,
            "include_q_branch_flows": True,
        },
        "full_AC_without_Q_rows": {
            "include_p_injections": True,
            "include_p_branch_flows": True,
        },
        "full_AC_without_branch_Q_flow_rows": {
            "include_p_injections": True,
            "include_q_injections": True,
            "include_p_branch_flows": True,
        },
        "full_AC_without_injection_Q_rows": {
            "include_p_injections": True,
            "include_p_branch_flows": True,
            "include_q_branch_flows": True,
        },
    }
    if row_set not in mapping:
        raise ValueError(f"unsupported row set: {row_set}")
    base.update(mapping[row_set])
    return base


def _pq_one_case(case_name: str, row_set: str, config: dict[str, Any]) -> dict[str, Any]:
    rng = make_rng(int(config["seed"]))
    system = build_ac_weighted_system(
        case_name=case_name,
        case_source=str(config["case_source"]),
        linearization_config=dict(config["linearization"]),
        measurement_config=build_measurement_config_for_row_set(row_set),
        rng=rng,
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H, compute_uv=False)
    positive = singular_values[singular_values > float(config["nonzero_tol"])]
    oracle = SparseJacobianOracle.from_matrix(H, nonzero_tol=float(config["nonzero_tol"]))
    pinv = PseudoinverseEstimator(rcond=1.0e-10).solve(system)
    ridge = RidgeEstimator(alpha=float(config["pq_ridge_alpha"])).solve(system)
    tsvd = TruncatedSVDEstimator(tau=float(config["pq_tsvd_tau"])).solve(system)
    huber_rmse = np.nan
    if bool(config["pq_run_huber"]):
        huber = HuberIRLSEstimator(delta=1.5, max_iterations=10, tolerance=1.0e-7).solve(system)
        huber_rmse = np.nan if huber.failed else system.rmse(huber.x_hat)
    return {
        "case_name": case_name,
        "row_set": row_set,
        "measurement_setting": "AC row-composition ablation",
        "rows": int(H.shape[0]),
        "states": int(H.shape[1]),
        "redundancy": float(H.shape[0] / max(H.shape[1], 1)),
        "rank": int(np.linalg.matrix_rank(H, tol=float(config["nonzero_tol"]))),
        "sigma_min_nonzero": float(np.min(positive)) if positive.size else 0.0,
        "sigma_max": float(np.max(singular_values)) if singular_values.size else 0.0,
        "condition_number": (
            float(np.max(singular_values) / np.min(positive)) if positive.size else np.inf
        ),
        "nnz": int(sum(oracle.row_nnz(i) for i in range(H.shape[0]))),
        "density": float(np.count_nonzero(np.abs(H) > float(config["nonzero_tol"])) / H.size),
        "max_row_sparsity": int(oracle.max_row_nnz()),
        "pinv_rmse": np.nan if pinv.failed else system.rmse(pinv.x_hat),
        "ridge_rmse": np.nan if ridge.failed else system.rmse(ridge.x_hat),
        "tsvd_rmse": np.nan if tsvd.failed else system.rmse(tsvd.x_hat),
        "huber_rmse": huber_rmse,
        "status": "rank_deficient" if positive.size < H.shape[1] else "completed",
        "failure_or_skip_reason": "",
    }


def tiny_reversible_sparse_oracle_lookup(config: dict[str, Any]) -> pd.DataFrame:
    matrix = np.asarray(config["tiny_oracle_matrix"], dtype=np.float64)
    try:
        row = tiny_oracle_lookup_row(matrix, matrix_label="tiny_sparse_fixture")
    except Exception as exc:
        row = {column: np.nan for column in TINY_ORACLE_COLUMNS}
        row.update(
            {
                "matrix_label": "tiny_sparse_fixture",
                "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                "circuit_constructed": False,
                "truth_table_passed": False,
                "simulation_status": "failed_or_skipped",
                "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return pd.DataFrame([row], columns=TINY_ORACLE_COLUMNS)


def tiny_oracle_lookup_row(matrix: np.ndarray, *, matrix_label: str) -> dict[str, Any]:
    oracle = SparseJacobianOracle.from_matrix(matrix)
    m, n = oracle.shape
    s = max(oracle.max_row_nnz(), 1)
    row_qubits = ceil_log2(m)
    ell_qubits = ceil_log2(s)
    col_qubits = ceil_log2(n)
    try:
        circuit, qubit_layout = build_tiny_index_oracle_circuit(oracle)
        passed = verify_tiny_index_oracle_truth_table(oracle, circuit, qubit_layout)
        unitarity = _operator_unitarity_error(circuit)
        transpile_meta = _transpile_circuit(circuit)
        return {
            "matrix_label": matrix_label,
            "matrix_shape": f"{m}x{n}",
            "row_qubits": row_qubits,
            "ell_qubits": ell_qubits,
            "column_qubits": col_qubits,
            "total_qubits": int(circuit.num_qubits),
            "num_truth_table_rows": int(m * s),
            "padding_encoded_column": 0,
            "circuit_constructed": True,
            "truth_table_passed": bool(passed),
            "unitarity_error_fro": unitarity,
            **transpile_meta,
            "simulation_status": "completed" if passed else "failed_truth_table",
            "failure_or_skip_reason": "",
        }
    except ImportError as exc:
        return {
            "matrix_label": matrix_label,
            "matrix_shape": f"{m}x{n}",
            "row_qubits": row_qubits,
            "ell_qubits": ell_qubits,
            "column_qubits": col_qubits,
            "total_qubits": row_qubits + ell_qubits + col_qubits,
            "num_truth_table_rows": int(m * s),
            "padding_encoded_column": 0,
            "circuit_constructed": False,
            "truth_table_passed": False,
            "unitarity_error_fro": np.nan,
            "transpilation_status": "skipped_qiskit_unavailable",
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
            "simulation_status": "skipped_qiskit_unavailable",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
        }


def build_tiny_index_oracle_circuit(
    oracle: SparseJacobianOracle,
) -> tuple[Any, dict[str, list[int]]]:
    from qiskit import QuantumCircuit  # type: ignore[import-not-found]

    m, n = oracle.shape
    s = max(oracle.max_row_nnz(), 1)
    row_qubits = ceil_log2(m)
    ell_qubits = ceil_log2(s)
    col_qubits = ceil_log2(n)
    total = row_qubits + ell_qubits + col_qubits
    circuit = QuantumCircuit(total, name="tiny_index_oracle")
    row_bits = list(range(row_qubits))
    ell_bits = list(range(row_qubits, row_qubits + ell_qubits))
    col_bits = list(range(row_qubits + ell_qubits, total))
    controls = row_bits + ell_bits
    for i in range(m):
        for ell in range(s):
            column = oracle.index_oracle(i, ell)
            encoded = 0 if column == oracle.padding_column else int(column)
            if encoded == 0:
                continue
            pattern = _bits_little_endian(i, row_qubits) + _bits_little_endian(ell, ell_qubits)
            for qubit, bit in zip(controls, pattern, strict=True):
                if bit == 0:
                    circuit.x(qubit)
            for bit_index, bit in enumerate(_bits_little_endian(encoded, col_qubits)):
                if bit:
                    if controls:
                        circuit.mcx(controls, col_bits[bit_index])
                    else:
                        circuit.x(col_bits[bit_index])
            for qubit, bit in reversed(list(zip(controls, pattern, strict=True))):
                if bit == 0:
                    circuit.x(qubit)
    return circuit, {"row": row_bits, "ell": ell_bits, "column": col_bits}


def verify_tiny_index_oracle_truth_table(
    oracle: SparseJacobianOracle,
    circuit: Any,
    qubit_layout: dict[str, list[int]],
) -> bool:
    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    row_bits = qubit_layout["row"]
    ell_bits = qubit_layout["ell"]
    col_bits = qubit_layout["column"]
    total_qubits = int(circuit.num_qubits)
    s = max(oracle.max_row_nnz(), 1)
    for i in range(oracle.shape[0]):
        for ell in range(s):
            initial_index = _basis_index(
                total_qubits,
                row_bits=row_bits,
                row_value=i,
                ell_bits=ell_bits,
                ell_value=ell,
                col_bits=col_bits,
                col_value=0,
            )
            evolved = Statevector.from_int(initial_index, 2**total_qubits).evolve(circuit)
            output_index = int(np.argmax(np.abs(evolved.data)))
            column = oracle.index_oracle(i, ell)
            expected = 0 if column == oracle.padding_column else int(column)
            expected_index = _basis_index(
                total_qubits,
                row_bits=row_bits,
                row_value=i,
                ell_bits=ell_bits,
                ell_value=ell,
                col_bits=col_bits,
                col_value=expected,
            )
            if output_index != expected_index:
                return False
    return True


def integrated_qsvt_repeat_case(config: dict[str, Any]) -> pd.DataFrame:
    if bool(config["force_repeat_case_failure"]):
        return pd.DataFrame(
            [
                {
                    **{column: np.nan for column in REPEAT_COLUMNS},
                    "case_name": str(config["repeat_case_spec"].get("case_name", "unknown")),
                    "subproblem_size": int(config["repeat_case_spec"].get("subproblem_size", 0)),
                    "alpha": float(config["repeat_alpha"]),
                    "epsilon_target": float(config["repeat_epsilon"]),
                    "simulation_status": "skipped_forced_failure",
                    "failure_or_skip_reason": "forced repeat-case failure for audit test",
                }
            ],
            columns=REPEAT_COLUMNS,
        )
    try:
        evaluation = run_ieee_selected_block(
            {
                "seed": int(config["seed"]),
                "subproblem_spec": dict(config["repeat_case_spec"]),
                "alpha": float(config["repeat_alpha"]),
                "epsilon_target": float(config["repeat_epsilon"]),
                "degree": int(config["repeat_degree"]),
                "angle_solver": str(config["phase_angle_solver"]),
                "basis_gates": list(config["basis_gates"]),
                "transpile_qubit_limit": int(config["repeat_transpile_qubit_limit"]),
                "transpile_optimization_level": int(config["transpile_optimization_level"]),
                "block_results_path": str(config["block_results_path"]),
                "block_matrices_dir": str(config["block_matrices_dir"]),
                "end_to_end_results_path": str(config["end_to_end_results_path"]),
                "artifact_match_rtol": 1.0e-9,
                "artifact_match_atol": 1.0e-8,
            }
        )
        row = evaluation.row
        out = {
            "case_name": row["case_name"],
            "subproblem_size": row["subproblem_size"],
            "alpha": row["alpha"],
            "epsilon_target": row["epsilon_target"],
            "degree": row["degree"],
            "phase_synthesis_status": row["phase_synthesis_status"],
            "qsvt_sequence_status": row["qsvt_sequence_status"],
            "simulation_status": row["simulation_status"],
            "transform_block_error_fro": row["transform_block_error_fro"],
            "circuit_vs_polynomial_fro_error": row["circuit_vs_polynomial_fro_error"],
            "relative_update_error": row["relative_update_error"],
            "residual_gap": row["residual_gap"],
            "success_probability": row["success_probability_residual_state"],
            "raw_circuit_depth": row["raw_circuit_depth"],
            "transpilation_status": row["transpilation_status"],
            "transpiled_depth": row["transpiled_depth"],
            "transpiled_cx_count": row["transpiled_cx_count"],
            "failure_or_skip_reason": row["failure_or_skip_reason"],
        }
    except Exception as exc:
        out = {
            **{column: np.nan for column in REPEAT_COLUMNS},
            "case_name": str(config["repeat_case_spec"].get("case_name", "unknown")),
            "subproblem_size": int(config["repeat_case_spec"].get("subproblem_size", 0)),
            "alpha": float(config["repeat_alpha"]),
            "epsilon_target": float(config["repeat_epsilon"]),
            "degree": int(config["repeat_degree"]),
            "phase_synthesis_status": "failed_or_not_attempted",
            "qsvt_sequence_status": "failed",
            "simulation_status": "failed",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
        }
    return pd.DataFrame([out], columns=REPEAT_COLUMNS)


def alpha_selection_diagnostic(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_name in config["alpha_cases"]:
        try:
            case_rows = _alpha_case_rows(case_name, config)
            rows.extend(case_rows)
        except Exception as exc:
            rows.append(
                {
                    **{column: np.nan for column in ALPHA_COLUMNS},
                    "case_name": case_name,
                    "stress_setting": "clean_linearized",
                    "rule_status": "failed",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows, columns=ALPHA_COLUMNS)


def alpha_tradeoff_rows(
    *,
    case_name: str,
    H: np.ndarray,
    r: np.ndarray,
    x_true: np.ndarray | None,
    alpha_grid: list[float],
    epsilon_targets: list[float],
    degree_grid: list[int],
) -> list[dict[str, Any]]:
    singular_values = np.linalg.svd(H, compute_uv=False)
    condition = _condition_from_singular_values(singular_values)
    per_alpha: dict[float, dict[str, float]] = {}
    for alpha in alpha_grid:
        update = ridge_update_svd(H, r, alpha=float(alpha))
        residual = float(np.linalg.norm(H @ update - r))
        rmse = float(np.sqrt(np.mean((update - x_true) ** 2))) if x_true is not None else np.nan
        per_alpha[float(alpha)] = {
            "rmse": rmse,
            "residual": residual,
            "update_norm": float(np.linalg.norm(update)),
        }
    oracle_alpha = min(
        per_alpha,
        key=lambda value: (
            per_alpha[value]["rmse"] if np.isfinite(per_alpha[value]["rmse"]) else np.inf
        ),
    )
    lcurve_alpha = choose_lcurve_alpha(per_alpha)
    rows = []
    for alpha in alpha_grid:
        degree_diag = degree_feasibility_by_epsilon(
            singular_values,
            alpha=float(alpha),
            epsilon_targets=epsilon_targets,
            degree_grid=degree_grid,
            dense_grid_size=257,
        )
        for epsilon in epsilon_targets:
            for rule_name, chosen in [
                ("oracle_best_alpha_using_x_true_diagnostic_only", oracle_alpha),
                ("discrete_lcurve_corner_diagnostic", lcurve_alpha),
            ]:
                rows.append(
                    {
                        "case_name": case_name,
                        "stress_setting": "clean_linearized",
                        "alpha": float(alpha),
                        "rmse": per_alpha[float(alpha)]["rmse"],
                        "residual_norm": per_alpha[float(alpha)]["residual"],
                        "update_norm": per_alpha[float(alpha)]["update_norm"],
                        "condition_number": condition,
                        "epsilon_target": float(epsilon),
                        "required_degree": degree_diag[float(epsilon)]["required_degree"],
                        "target_met": degree_diag[float(epsilon)]["target_met"],
                        "chosen_alpha_by_rule": float(chosen),
                        "rule_name": rule_name,
                        "rule_status": "completed",
                        "notes": (
                            "alpha diagnostic only; not a field-calibrated operational tuning rule"
                        ),
                    }
                )
    return rows


def choose_lcurve_alpha(per_alpha: dict[float, dict[str, float]]) -> float:
    ordered = sorted(per_alpha)
    if len(ordered) < 3:
        return ordered[len(ordered) // 2]
    points = np.array(
        [
            [
                math.log10(max(per_alpha[alpha]["residual"], SMALL_TOL)),
                math.log10(max(per_alpha[alpha]["update_norm"], SMALL_TOL)),
            ]
            for alpha in ordered
        ],
        dtype=np.float64,
    )
    scores = []
    for idx in range(1, len(points) - 1):
        a = points[idx] - points[idx - 1]
        b = points[idx + 1] - points[idx]
        denom = max(np.linalg.norm(a) * np.linalg.norm(b), SMALL_TOL)
        cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        scores.append((math.acos(cosine), ordered[idx]))
    return max(scores, key=lambda item: item[0])[1]


def _alpha_case_rows(case_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = make_rng(int(config["seed"]))
    system = build_ac_weighted_system(
        case_name=case_name,
        case_source=str(config["case_source"]),
        linearization_config=dict(config["linearization"]),
        measurement_config=build_measurement_config_for_row_set("full_AC"),
        rng=rng,
    )
    return alpha_tradeoff_rows(
        case_name=case_name,
        H=np.asarray(system.H_tilde, dtype=np.float64),
        r=np.asarray(system.r_tilde, dtype=np.float64),
        x_true=None if system.x_true is None else np.asarray(system.x_true, dtype=np.float64),
        alpha_grid=[float(value) for value in config["alpha_grid"]],
        epsilon_targets=[float(value) for value in config["epsilon_targets"]],
        degree_grid=[int(value) for value in config["degree_grid"]],
    )


def final_robustness_summary(
    *,
    phase: pd.DataFrame,
    signed: pd.DataFrame,
    noise: pd.DataFrame,
    pq: pd.DataFrame,
    oracle: pd.DataFrame,
    repeat: pd.DataFrame,
    alpha: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        ("A_phase_synthesis_hard_case_audit", phase, "phase_synthesis_status"),
        ("B_signed_phase_aware_readout", signed, "simulation_status"),
        ("C_noise_sensitivity_integrated_qsvt", noise, "simulation_status"),
        ("D_reactive_pq_row_composition_ablation", pq, "status"),
        ("E_tiny_reversible_sparse_oracle_lookup", oracle, "simulation_status"),
        ("F_integrated_qsvt_repeat_case", repeat, "simulation_status"),
        ("G_alpha_selection_diagnostic", alpha, "rule_status"),
    ]
    rows = []
    for name, frame, status_col in specs:
        statuses = (
            frame[status_col].astype(str) if status_col in frame else pd.Series([], dtype=str)
        )
        completed = int(statuses.str.contains("completed|success|rank_deficient", case=False).sum())
        skipped = int(statuses.str.contains("skipped|unavailable", case=False).sum())
        failed = int(statuses.str.contains("failed|failure", case=False).sum())
        status = "completed" if completed and failed == 0 else "completed_with_boundaries"
        if completed == 0 and (failed or skipped):
            status = "skipped_or_failed"
        rows.append(
            {
                "sub_experiment": name,
                "status": status,
                "rows": len(frame),
                "completed_rows": completed,
                "failed_rows": failed,
                "skipped_rows": skipped,
                "key_metric": _key_metric_for_summary(name, frame),
                "main_paper_recommendation": _main_paper_recommendation(name),
                "supplement_recommendation": _supplement_recommendation(name),
                "claim_boundary_note": (
                    "audit/diagnostic only; no speedup, hardware, full-scale, "
                    "or QSVT-over-Ridge claim"
                ),
            }
        )
    return pd.DataFrame(rows, columns=FINAL_SUMMARY_COLUMNS)


def load_update_components(config: dict[str, Any]) -> UpdateComponents:
    if "ridge_update" in config and "qsvt_update" in config:
        return UpdateComponents(
            ridge=np.asarray(config["ridge_update"], dtype=np.float64),
            qsvt=np.asarray(config["qsvt_update"], dtype=np.float64),
            metadata=dict(config.get("metadata", {})),
        )
    path = Path(config["update_components_path"])
    frame = pd.read_csv(path)
    subset = frame[
        (frame["case_name"] == str(config["readout_case_name"]))
        & (frame["subproblem_size"].astype(int) == int(config["readout_subproblem_size"]))
        & np.isclose(frame["alpha"].astype(float), float(config["readout_alpha"]))
        & (frame["degree"].astype(int) == int(config["readout_degree"]))
    ].sort_values("component_index")
    if subset.empty:
        raise ValueError("matching update components not found")
    subset = subset.drop_duplicates("component_index", keep="first")
    return UpdateComponents(
        ridge=subset["ridge_update_component"].to_numpy(dtype=np.float64),
        qsvt=subset["qsvt_poly_update_component"].to_numpy(dtype=np.float64),
        metadata={
            "case_name": config["readout_case_name"],
            "subproblem_size": config["readout_subproblem_size"],
            "source": str(path),
        },
    )


def normalized_probabilities(update: np.ndarray) -> np.ndarray:
    values = np.asarray(update, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= SMALL_TOL:
        raise ValueError("update norm is too small")
    p = values**2 / norm**2
    return p / float(np.sum(p))


def _write_all_outputs(
    *,
    config: dict[str, Any],
    output_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
    phase: pd.DataFrame,
    signed: pd.DataFrame,
    noise: pd.DataFrame,
    pq: pd.DataFrame,
    oracle: pd.DataFrame,
    repeat: pd.DataFrame,
    alpha: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> dict[str, Path]:
    artifacts = {
        "phase_audit_csv": output_dir / "phase_synthesis_hard_case_audit.csv",
        "phase_audit_table": tables_dir / "table_phase_synthesis_hard_case_audit.csv",
        "phase_audit_figure": figures_dir / "figure_phase_synthesis_success_by_degree.png",
        "phase_audit_report": reports_dir / "phase_synthesis_hard_case_audit_report.md",
        "signed_readout_csv": output_dir / "signed_phase_aware_readout_results.csv",
        "signed_readout_table": tables_dir / "table_signed_phase_aware_readout_summary.csv",
        "signed_readout_figure": figures_dir / "figure_signed_readout_error.png",
        "signed_readout_report": reports_dir / "signed_phase_aware_readout_report.md",
        "noise_csv": output_dir / "noise_sensitivity_integrated_qsvt.csv",
        "noise_table": tables_dir / "table_noise_sensitivity_integrated_qsvt.csv",
        "noise_error_figure": figures_dir / "figure_noise_sensitivity_observable_error.png",
        "noise_success_figure": figures_dir / "figure_noise_sensitivity_success_probability.png",
        "noise_report": reports_dir / "noise_sensitivity_integrated_qsvt_report.md",
        "pq_csv": output_dir / "reactive_pq_row_composition_ablation.csv",
        "pq_table": tables_dir / "table_reactive_pq_row_composition_summary.csv",
        "pq_condition_figure": figures_dir / "figure_reactive_pq_conditioning.png",
        "pq_rank_figure": figures_dir / "figure_reactive_pq_rank_redundancy.png",
        "pq_report": reports_dir / "reactive_pq_row_composition_ablation_report.md",
        "tiny_oracle_csv": output_dir / "tiny_reversible_sparse_oracle_lookup.csv",
        "tiny_oracle_table": tables_dir / "table_tiny_reversible_sparse_oracle_lookup.csv",
        "tiny_oracle_figure": figures_dir / "figure_tiny_oracle_circuit_resources.png",
        "tiny_oracle_report": reports_dir / "tiny_reversible_sparse_oracle_lookup_report.md",
        "repeat_csv": output_dir / "integrated_qsvt_repeat_case.csv",
        "repeat_table": tables_dir / "table_integrated_qsvt_repeat_case.csv",
        "repeat_figure": figures_dir / "figure_integrated_qsvt_repeat_case_errors.png",
        "repeat_report": reports_dir / "integrated_qsvt_repeat_case_report.md",
        "alpha_csv": output_dir / "alpha_selection_diagnostic.csv",
        "alpha_table": tables_dir / "table_alpha_selection_diagnostic_summary.csv",
        "alpha_tradeoff_figure": figures_dir / "figure_alpha_selection_rmse_degree_tradeoff.png",
        "alpha_lcurve_figure": figures_dir / "figure_alpha_selection_lcurve.png",
        "alpha_report": reports_dir / "alpha_selection_diagnostic_report.md",
        "final_summary_table": tables_dir / "table_final_robustness_audit_summary.csv",
        "final_report": reports_dir / "final_robustness_audits_report.md",
    }
    phase.to_csv(artifacts["phase_audit_csv"], index=False)
    phase.to_csv(artifacts["phase_audit_table"], index=False)
    signed.to_csv(artifacts["signed_readout_csv"], index=False)
    signed.to_csv(artifacts["signed_readout_table"], index=False)
    noise.to_csv(artifacts["noise_csv"], index=False)
    noise.to_csv(artifacts["noise_table"], index=False)
    pq.to_csv(artifacts["pq_csv"], index=False)
    pq.to_csv(artifacts["pq_table"], index=False)
    oracle.to_csv(artifacts["tiny_oracle_csv"], index=False)
    oracle.to_csv(artifacts["tiny_oracle_table"], index=False)
    repeat.to_csv(artifacts["repeat_csv"], index=False)
    repeat.to_csv(artifacts["repeat_table"], index=False)
    alpha.to_csv(artifacts["alpha_csv"], index=False)
    alpha.to_csv(artifacts["alpha_table"], index=False)
    final_summary.to_csv(artifacts["final_summary_table"], index=False)
    _plot_phase(phase, artifacts["phase_audit_figure"])
    _plot_signed(signed, artifacts["signed_readout_figure"])
    _plot_noise_error(noise, artifacts["noise_error_figure"])
    _plot_noise_success(noise, artifacts["noise_success_figure"])
    _plot_pq_condition(pq, artifacts["pq_condition_figure"])
    _plot_pq_rank(pq, artifacts["pq_rank_figure"])
    _plot_tiny_oracle(oracle, artifacts["tiny_oracle_figure"])
    _plot_repeat(repeat, artifacts["repeat_figure"])
    _plot_alpha_tradeoff(alpha, artifacts["alpha_tradeoff_figure"])
    _plot_alpha_lcurve(alpha, artifacts["alpha_lcurve_figure"])
    _write_reports(
        config=config,
        artifacts=artifacts,
        phase=phase,
        signed=signed,
        noise=noise,
        pq=pq,
        oracle=oracle,
        repeat=repeat,
        alpha=alpha,
        final_summary=final_summary,
    )
    return artifacts


def _write_reports(
    *,
    config: dict[str, Any],
    artifacts: dict[str, Path],
    phase: pd.DataFrame,
    signed: pd.DataFrame,
    noise: pd.DataFrame,
    pq: pd.DataFrame,
    oracle: pd.DataFrame,
    repeat: pd.DataFrame,
    alpha: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> None:
    report_specs = [
        (
            "phase_audit_report",
            "Phase-Synthesis Hard-Case Audit",
            "This audit identifies which representative degree/alpha/precision settings can be "
            "phase-synthesized under the configured numerical budget. Failed or skipped hard "
            "cases are reported as feasibility boundaries, not hidden.",
            phase,
        ),
        (
            "signed_readout_report",
            "Signed Phase-Aware Readout Diagnostic",
            "Signed update quantities require phase/sign-aware access. This diagnostic separates "
            "statevector-accessible signed information from energy-style observables accessible "
            "through computational-basis sampling.",
            signed,
        ),
        (
            "noise_report",
            "Noise Sensitivity of Integrated Small QSVT Circuit",
            "The dense proof-of-concept circuit is not optimized for hardware. Noise simulation "
            "is used only to characterize sensitivity and motivate resource-aware or "
            "fault-tolerant implementations.",
            noise,
        ),
        (
            "pq_report",
            "Reactive P/Q Measurement-Row Composition Ablation",
            "This ablation separates active/reactive and injection/branch-flow row effects on "
            "weighted-Jacobian conditioning. It supports the PSSE measurement-composition "
            "interpretation rather than a QSVT-over-Ridge claim.",
            pq,
        ),
        (
            "tiny_oracle_report",
            "Tiny Reversible Sparse-Oracle Lookup Circuit",
            "This is a tiny reversible lookup prototype for the sparse index oracle. It is not "
            "a compiled scalable oracle for full IEEE-scale matrices.",
            oracle,
        ),
        (
            "repeat_report",
            "Integrated QSVT Repeat-Case Diagnostic",
            "This repeat-case diagnostic tests whether the integrated small-circuit construction "
            "transfers to one additional selected subproblem. It remains selected-subproblem "
            "evidence.",
            repeat,
        ),
        (
            "alpha_report",
            "Alpha-Selection Diagnostic",
            "The alpha-selection diagnostic is used to understand stability and degree-cost "
            "tradeoffs. It is not presented as a field-calibrated operational tuning rule.",
            alpha,
        ),
    ]
    for key, title, interpretation, frame in report_specs:
        artifacts[key].write_text(
            _simple_report(title, interpretation, frame, config),
            encoding="utf-8",
        )
    artifacts["final_report"].write_text(
        _final_report_markdown(final_summary, artifacts),
        encoding="utf-8",
    )


def _simple_report(
    title: str,
    interpretation: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    status_columns = [
        column
        for column in ("phase_synthesis_status", "simulation_status", "status", "rule_status")
        if column in frame
    ]
    status_lines = []
    for column in status_columns:
        status_lines.append(f"- {column}: {frame[column].value_counts(dropna=False).to_dict()}")
    if not status_lines:
        status_lines = ["- No status column available."]
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Goal",
            "",
            interpretation,
            "",
            "## Configuration",
            "",
            f"- Seed: {config['seed']}",
            f"- Alpha grid: {config['alpha_grid']}",
            f"- Epsilon targets: {config['epsilon_targets']}",
            f"- Degree grid: {config['degree_grid']}",
            "",
            "## Status",
            "",
            f"- Rows: {len(frame)}",
            *status_lines,
            "",
            "## Claim Boundary",
            "",
            "This audit is claim-boundary preserving. It does not demonstrate quantum "
            "speedup, hardware execution, full IEEE-scale QSVT execution, full-vector "
            "readout, scalable sparse-oracle construction, or QSVT superiority over "
            "Ridge/Tikhonov.",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _final_report_markdown(summary: pd.DataFrame, artifacts: dict[str, Path]) -> str:
    rows = [
        f"- {row.sub_experiment}: {row.status}; rows={row.rows}; key={row.key_metric}"
        for row in summary.itertuples(index=False)
    ]
    artifact_lines = [f"- `{path}`" for path in artifacts.values()]
    return "\n".join(
        [
            "# Final Robustness Audits Report",
            "",
            "## Summary",
            "",
            *rows,
            "",
            "## Main-Paper vs Supplement Recommendation",
            "",
            "- Main paper: use the consolidated summary table, the phase-synthesis boundary "
            "statement, the signed-readout limitation statement, and one compact alpha/degree "
            "tradeoff figure if space allows.",
            "- Supplement: place detailed phase audit rows, noise sensitivity, P/Q row-composition "
            "tables, tiny oracle truth-table resources, and repeat-case details.",
            "",
            "## Claim Boundary",
            "",
            "These robustness audits strengthen the manuscript by documenting phase-synthesis "
            "boundaries, phase-aware readout requirements, noise sensitivity, measurement-row "
            "effects, tiny oracle-circuit feasibility, repeatability of the integrated "
            "small-circuit result, and alpha/degree tradeoffs.",
            "",
            "They do not prove scalable QSVT PSSE, quantum speedup, full-vector readout, "
            "hardware execution, QSVT-over-Ridge superiority, or nonlinear QSVT-in-the-loop "
            "state estimation.",
            "",
            "## Artifacts",
            "",
            *artifact_lines,
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 123,
        "case_source": "pypower",
        "dense_grid_size": 1025,
        "bounded_tol": 1.0e-7,
        "max_phase_synthesis_degree": 35,
        "phase_angle_solver": "root-finding",
        "alpha_grid": [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
        "epsilon_targets": [1.0e-2, 1.0e-3, 1.0e-4],
        "degree_grid": [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 201],
        "phase_audit_settings": [
            {
                "difficulty": "easy",
                "case_name": "ieee14",
                "subproblem_size": 4,
                "alpha": 1.0e-2,
                "epsilon_target": 1.0e-2,
                "requested_degree": 11,
            },
            {
                "difficulty": "medium",
                "case_name": "ieee30",
                "subproblem_size": 8,
                "alpha": 1.0e-3,
                "epsilon_target": 1.0e-3,
                "requested_degree": 35,
            },
            {
                "difficulty": "hard",
                "case_name": "ieee57",
                "subproblem_size": 16,
                "alpha": 1.0e-4,
                "epsilon_target": 1.0e-4,
                "requested_degree": 75,
            },
            {
                "difficulty": "very_hard",
                "case_name": "ieee118",
                "subproblem_size": 16,
                "alpha": 1.0e-6,
                "epsilon_target": 1.0e-4,
                "requested_degree": 201,
            },
        ],
        "readout_case_name": "ieee14",
        "readout_subproblem_size": 4,
        "readout_alpha": 1.0e-2,
        "readout_degree": 11,
        "update_components_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_update_components.csv"
        ),
        "integrated_success_probability": 0.37965004483970727,
        "integrated_raw_depth": 23,
        "integrated_cx_count": 246,
        "use_aer_if_available": True,
        "noise_models": [
            {"noise_model": "ideal", "p1": 0.0, "p2": 0.0, "readout_error": 0.0},
            {"noise_model": "depolarizing_low", "p1": 1.0e-4, "p2": 1.0e-4, "readout_error": 0.0},
            {
                "noise_model": "depolarizing_medium",
                "p1": 1.0e-3,
                "p2": 1.0e-3,
                "readout_error": 1.0e-3,
            },
            {
                "noise_model": "depolarizing_high",
                "p1": 1.0e-2,
                "p2": 1.0e-2,
                "readout_error": 1.0e-2,
            },
        ],
        "noise_shots": [1000, 10000],
        "pq_cases": ["ieee14", "ieee30", "ieee57", "ieee118"],
        "pq_row_sets": [
            "V_only",
            "V_plus_P_injection",
            "V_plus_Q_injection",
            "V_plus_PQ_injection",
            "V_plus_P_branch_flow",
            "V_plus_Q_branch_flow",
            "V_plus_PQ_branch_flow",
            "full_AC",
            "full_AC_without_Q_rows",
            "full_AC_without_branch_Q_flow_rows",
            "full_AC_without_injection_Q_rows",
        ],
        "pq_ridge_alpha": 1.0e-4,
        "pq_tsvd_tau": 1.0e-5,
        "pq_run_huber": False,
        "linearization": {
            "angle_perturbation_std": 0.005,
            "voltage_perturbation_std": 0.005,
            "min_voltage_magnitude": 0.5,
        },
        "nonzero_tol": 1.0e-12,
        "tiny_oracle_matrix": [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
        "repeat_case_spec": {
            "case_name": "ieee30",
            "subproblem_size": 4,
            "selection_mode": "high_leverage",
        },
        "repeat_alpha": 1.0e-2,
        "repeat_epsilon": 1.0e-2,
        "repeat_degree": 11,
        "force_repeat_case_failure": False,
        "repeat_transpile_qubit_limit": 4,
        "transpile_optimization_level": 1,
        "basis_gates": DEFAULT_BASIS_GATES,
        "block_results_path": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "block_matrices_dir": str(root / "explicit_block_encoding_demo" / "matrices"),
        "end_to_end_results_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_results.csv"
        ),
        "alpha_cases": ["ieee14", "ieee57"],
    }
    if config:
        resolved.update(config)
    resolved["alpha_grid"] = [float(value) for value in resolved["alpha_grid"]]
    resolved["epsilon_targets"] = [float(value) for value in resolved["epsilon_targets"]]
    resolved["degree_grid"] = [int(value) for value in resolved["degree_grid"]]
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    return resolved


def _pad_odd_coefficients(coefficients: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.size < int(degree) + 1:
        values = np.pad(values, (0, int(degree) + 1 - values.size))
    values = values[: int(degree) + 1].copy()
    values[0::2] = 0.0
    values[np.abs(values) < 1.0e-14] = 0.0
    return values


def _signed_value(vector: np.ndarray, indices: tuple[int, ...]) -> float:
    if len(indices) == 1:
        return float(vector[indices[0]])
    return float(vector[indices[0]] - vector[indices[1]])


def _signed_metadata_label(
    indices: tuple[int, ...],
    metadata: dict[str, Any],
    observable_type: str,
) -> str:
    source = metadata.get("source", "reconstructed update components")
    if len(indices) == 1:
        return f"selected state coordinate {indices[0]}; source={source}"
    return (
        f"coordinate-pair proxy {indices[0]}-{indices[1]}; no physical branch label asserted; "
        f"type={observable_type}; source={source}"
    )


def _noise_strength_proxy(
    *,
    p1: float,
    p2: float,
    readout_error: float,
    depth: int,
    cx_count: int,
) -> float:
    strength = 1.0 - (1.0 - p1) ** max(depth, 0) * (1.0 - p2) ** max(cx_count, 0)
    strength = 1.0 - (1.0 - strength) * (1.0 - readout_error)
    return float(np.clip(strength, 0.0, 1.0))


def _proxy_noise_row(
    *,
    qsvt: np.ndarray,
    probabilities: np.ndarray,
    selected: int,
    ideal_success: float,
    noise_spec: dict[str, Any],
    shots: int,
    seed: int,
    depth: int,
    cx_count: int,
    failure_reason: str,
) -> dict[str, Any]:
    p1 = float(noise_spec.get("p1", 0.0))
    p2 = float(noise_spec.get("p2", 0.0))
    readout = float(noise_spec.get("readout_error", 0.0))
    strength = _noise_strength_proxy(
        p1=p1,
        p2=p2,
        readout_error=readout,
        depth=depth,
        cx_count=cx_count,
    )
    mixed = (1.0 - strength) * probabilities + strength / probabilities.size
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(int(shots), mixed)
    p_hat = counts[selected] / float(shots)
    ideal_energy = float(np.linalg.norm(qsvt) ** 2 * probabilities[selected])
    noisy_energy = float(np.linalg.norm(qsvt) ** 2 * p_hat)
    return {
        "noise_model": str(noise_spec["noise_model"]),
        "p1": p1,
        "p2": p2,
        "readout_error": readout,
        "shots": int(shots),
        "success_probability_estimate": ideal_success * (1.0 - strength),
        "observable_error_vs_ideal": abs(noisy_energy - ideal_energy),
        "update_error_proxy_vs_ideal": float(np.linalg.norm(mixed - probabilities)),
        "residual_gap_proxy": abs(strength),
        "distribution_total_variation_distance": 0.5 * float(np.sum(np.abs(mixed - probabilities))),
        "noisy_counts_available": True,
        "simulation_status": "completed_statevector_distribution_noise_proxy",
        "failure_or_skip_reason": failure_reason,
    }


def _noise_proxy_reason(*, aer_available: bool, use_aer: bool, fallback_reason: str) -> str:
    if fallback_reason:
        return fallback_reason
    if not use_aer:
        return "Aer not requested for this run; used deterministic distribution-noise proxy"
    if not aer_available:
        return "qiskit_aer_unavailable; used deterministic distribution-noise proxy"
    return "used deterministic distribution-noise proxy"


def _build_aer_noise_context(
    *,
    config: dict[str, Any],
    qsvt_update: np.ndarray,
) -> dict[str, Any]:
    from qiskit import QuantumCircuit, transpile  # type: ignore[import-not-found]
    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    spec = {
        "case_name": str(config["readout_case_name"]),
        "subproblem_size": int(config["readout_subproblem_size"]),
        "selection_mode": "high_leverage",
    }
    subproblem = load_sweep_subproblem(spec, seed=int(config["seed"]))
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    b = np.asarray(subproblem.r_tilde, dtype=np.float64)
    gamma = float(np.linalg.svd(A, compute_uv=False)[0])
    encoding = construct_padded_block_encoding(A, gamma=gamma)
    degree = int(config["readout_degree"])
    cheb, _ = fit_actual_singular_interpolating_polynomial(
        alpha=float(config["readout_alpha"]),
        gamma=gamma,
        singular_values=np.linalg.svd(A, compute_uv=False),
        degree=degree,
    )
    coefficients = cheb.convert(kind=Polynomial).coef
    coefficients = _pad_odd_coefficients(coefficients, degree)
    phase_result = synthesize_qsvt_phases(
        coefficients,
        angle_solver=str(config["phase_angle_solver"]),
    )
    if phase_result.status != "completed":
        raise RuntimeError(f"phase synthesis unavailable for Aer noise path: {phase_result.status}")
    bundle = build_structured_qsvt_operator_circuit(
        np.asarray(encoding.U, dtype=np.complex128),
        np.asarray(phase_result.phases, dtype=np.float64),
        encoded_dimension=int(encoding.A_bar_padded.shape[0]),
    )
    total_dimension = int(encoding.U.shape[0])
    residual = np.zeros(total_dimension, dtype=np.complex128)
    residual[: b.size] = b / max(float(np.linalg.norm(b)), SMALL_TOL)
    prep = QuantumCircuit(bundle.n_qubits, name="residual_state_prep")
    prep.initialize(residual, list(range(bundle.n_qubits)))
    circuit = prep.compose(bundle.qsvt_operator_circuit.inverse())
    measured = circuit.copy()
    measured.measure_all()
    transpiled = transpile(
        measured,
        basis_gates=list(config["basis_gates"]),
        optimization_level=0,
        seed_transpiler=int(config["seed"]),
    )
    ideal_probs = np.asarray(
        Statevector.from_instruction(circuit).probabilities(),
        dtype=np.float64,
    )
    update_dimension = int(qsvt_update.size)
    postselected = ideal_probs[:update_dimension]
    ideal_success = float(np.sum(postselected))
    if ideal_success <= SMALL_TOL:
        raise RuntimeError("ideal postselection probability is numerically zero")
    return {
        "transpiled_measured_circuit": transpiled,
        "ideal_conditional_distribution": postselected / ideal_success,
        "ideal_success_probability": ideal_success,
        "qsvt_update_norm": float(np.linalg.norm(qsvt_update)),
        "update_dimension": update_dimension,
    }


def _aer_noise_row(
    *,
    context: dict[str, Any],
    noise_spec: dict[str, Any],
    shots: int,
    seed: int,
) -> dict[str, Any]:
    from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

    p1 = float(noise_spec.get("p1", 0.0))
    p2 = float(noise_spec.get("p2", 0.0))
    readout = float(noise_spec.get("readout_error", 0.0))
    noise_model = _build_aer_noise_model(p1=p1, p2=p2, readout_error=readout)
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    job = simulator.run(
        context["transpiled_measured_circuit"],
        shots=int(shots),
        seed_simulator=seed,
    )
    counts = job.result().get_counts()
    update_dimension = int(context["update_dimension"])
    observed = np.zeros(update_dimension, dtype=np.float64)
    success_count = 0
    for bitstring, count in counts.items():
        index = _counts_key_to_basis_index(str(bitstring))
        if 0 <= index < update_dimension:
            observed[index] += int(count)
            success_count += int(count)
    if success_count == 0:
        conditional = np.zeros(update_dimension, dtype=np.float64)
    else:
        conditional = observed / float(success_count)
    ideal = np.asarray(context["ideal_conditional_distribution"], dtype=np.float64)
    selected = int(np.argmax(ideal))
    qsvt_norm = float(context["qsvt_update_norm"])
    ideal_energy = qsvt_norm**2 * float(ideal[selected])
    observed_energy = qsvt_norm**2 * float(conditional[selected])
    success_probability = float(success_count / max(int(shots), 1))
    return {
        "noise_model": str(noise_spec["noise_model"]),
        "p1": p1,
        "p2": p2,
        "readout_error": readout,
        "shots": int(shots),
        "success_probability_estimate": success_probability,
        "observable_error_vs_ideal": abs(observed_energy - ideal_energy),
        "update_error_proxy_vs_ideal": float(np.linalg.norm(conditional - ideal)),
        "residual_gap_proxy": abs(
            success_probability - float(context["ideal_success_probability"])
        ),
        "distribution_total_variation_distance": 0.5 * float(np.sum(np.abs(conditional - ideal))),
        "noisy_counts_available": True,
        "simulation_status": "completed_aer_noise_model",
        "failure_or_skip_reason": "",
    }


def _build_aer_noise_model(*, p1: float, p2: float, readout_error: float) -> Any:
    if p1 <= 0.0 and p2 <= 0.0 and readout_error <= 0.0:
        return None
    from qiskit_aer.noise import (  # type: ignore[import-not-found]
        NoiseModel,
        ReadoutError,
        depolarizing_error,
    )

    model = NoiseModel()
    if p1 > 0.0:
        model.add_all_qubit_quantum_error(depolarizing_error(float(p1), 1), ["rz", "sx", "x"])
    if p2 > 0.0:
        model.add_all_qubit_quantum_error(depolarizing_error(float(p2), 2), ["cx"])
    if readout_error > 0.0:
        error = ReadoutError(
            [
                [1.0 - float(readout_error), float(readout_error)],
                [float(readout_error), 1.0 - float(readout_error)],
            ]
        )
        model.add_all_qubit_readout_error(error)
    return model


def _counts_key_to_basis_index(key: str) -> int:
    bitstring = key.replace(" ", "")
    if not bitstring:
        return 0
    return int(bitstring, 2)


def _noise_failure_row(reason: str) -> dict[str, Any]:
    return {
        "noise_model": "unavailable",
        "p1": np.nan,
        "p2": np.nan,
        "readout_error": np.nan,
        "shots": np.nan,
        "success_probability_estimate": np.nan,
        "observable_error_vs_ideal": np.nan,
        "update_error_proxy_vs_ideal": np.nan,
        "residual_gap_proxy": np.nan,
        "distribution_total_variation_distance": np.nan,
        "noisy_counts_available": False,
        "simulation_status": "failed_input_unavailable",
        "failure_or_skip_reason": reason,
    }


def _pq_failure_row(case_name: str, row_set: str, exc: Exception) -> dict[str, Any]:
    return {
        **{column: np.nan for column in PQ_COLUMNS},
        "case_name": case_name,
        "row_set": row_set,
        "measurement_setting": "AC row-composition ablation",
        "status": "failed",
        "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
    }


def _condition_from_singular_values(singular_values: np.ndarray) -> float:
    positive = singular_values[singular_values > 1.0e-12]
    return float(np.max(positive) / np.min(positive)) if positive.size else np.inf


def _bits_little_endian(value: int, width: int) -> list[int]:
    return [(int(value) >> index) & 1 for index in range(int(width))]


def _basis_index(
    total_qubits: int,
    *,
    row_bits: list[int],
    row_value: int,
    ell_bits: list[int],
    ell_value: int,
    col_bits: list[int],
    col_value: int,
) -> int:
    bits = [0] * int(total_qubits)
    for qubit, bit in zip(row_bits, _bits_little_endian(row_value, len(row_bits)), strict=True):
        bits[qubit] = bit
    for qubit, bit in zip(ell_bits, _bits_little_endian(ell_value, len(ell_bits)), strict=True):
        bits[qubit] = bit
    for qubit, bit in zip(col_bits, _bits_little_endian(col_value, len(col_bits)), strict=True):
        bits[qubit] = bit
    return int(sum(bit << index for index, bit in enumerate(bits)))


def _operator_unitarity_error(circuit: Any) -> float:
    from qiskit.quantum_info import Operator  # type: ignore[import-not-found]

    matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0]), ord="fro"))


def _transpile_circuit(circuit: Any) -> dict[str, Any]:
    try:
        from qiskit import transpile  # type: ignore[import-not-found]

        transpiled = transpile(circuit, basis_gates=DEFAULT_BASIS_GATES, optimization_level=1)
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        return {
            "transpilation_status": "completed",
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_cx_count": int(counts.get("cx", 0)),
            "transpiled_total_ops": int(sum(counts.values())),
        }
    except Exception as exc:  # pragma: no cover - qiskit-version dependent
        return {
            "transpilation_status": "failed",
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
        }


def _version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _qiskit_aer_available() -> bool:
    try:
        import qiskit_aer  # noqa: F401  # type: ignore[import-not-found]

        return True
    except Exception:
        return False


def _input_artifact_paths(config: dict[str, Any]) -> dict[str, str]:
    keys = [
        "update_components_path",
        "block_results_path",
        "block_matrices_dir",
        "end_to_end_results_path",
    ]
    return {key: str(config[key]) for key in keys if key in config}


def _status_rollup(summary: pd.DataFrame) -> dict[str, int]:
    return {
        "completed_rows": int(summary["completed_rows"].sum()),
        "failed_rows": int(summary["failed_rows"].sum()),
        "skipped_rows": int(summary["skipped_rows"].sum()),
    }


def _key_metric_for_summary(name: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return "no rows"
    if name.startswith("A_"):
        return f"phase statuses={frame['phase_synthesis_status'].value_counts().to_dict()}"
    if name.startswith("B_"):
        signed_error = pd.to_numeric(frame["abs_error_vs_ridge"], errors="coerce").max()
        return f"max signed error={signed_error:.3e}"
    if name.startswith("C_"):
        tv_distance = pd.to_numeric(
            frame["distribution_total_variation_distance"],
            errors="coerce",
        ).max()
        return f"max TV distance={tv_distance:.3e}"
    if name.startswith("D_"):
        return (
            "condition range="
            f"{pd.to_numeric(frame['condition_number'], errors='coerce').min():.3e}-"
            f"{pd.to_numeric(frame['condition_number'], errors='coerce').max():.3e}"
        )
    if name.startswith("E_"):
        return f"truth_table_passed={bool(frame['truth_table_passed'].iloc[0])}"
    if name.startswith("F_"):
        return f"repeat status={frame['simulation_status'].iloc[0]}"
    if name.startswith("G_"):
        return f"rules={frame['rule_name'].nunique() if 'rule_name' in frame else 0}"
    return "generated"


def _main_paper_recommendation(name: str) -> str:
    if name.startswith(("A_", "B_", "G_")):
        return "main-paper worthy as compact boundary/diagnostic statement"
    return "optional main-paper mention only if space allows"


def _supplement_recommendation(name: str) -> str:
    return "include detailed rows/figures in supplement or reproducibility package"


def _plot_phase(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if frame.empty:
        ax.text(0.5, 0.5, "No phase audit rows", ha="center", va="center")
    else:
        colors = [
            "#4c78a8" if status == "completed" else "#f58518"
            for status in frame["phase_synthesis_status"]
        ]
        ax.scatter(frame["used_degree"], np.arange(len(frame)), c=colors, s=70)
        ax.set_yticks(np.arange(len(frame)))
        ax.set_yticklabels(frame["difficulty"].astype(str))
        ax.set_xlabel("used polynomial degree")
        ax.set_title("Phase Synthesis Audit Status by Degree")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_signed(frame: pd.DataFrame, path: Path) -> None:
    _bar_plot(
        frame,
        path,
        x_col="observable_name",
        y_col="abs_error_vs_ridge",
        title="Signed Phase-Aware Readout Error",
        ylabel="absolute error vs Ridge",
        log=True,
    )


def _plot_noise_error(frame: pd.DataFrame, path: Path) -> None:
    _line_plot(frame, path, "shots", "observable_error_vs_ideal", "noise_model", "Noise Error")


def _plot_noise_success(frame: pd.DataFrame, path: Path) -> None:
    _line_plot(
        frame,
        path,
        "shots",
        "success_probability_estimate",
        "noise_model",
        "Noise Success Probability",
    )


def _plot_pq_condition(frame: pd.DataFrame, path: Path) -> None:
    _bar_plot(
        frame,
        path,
        x_col="row_set",
        y_col="condition_number",
        title="P/Q Row-Composition Conditioning",
        ylabel="condition number",
        log=True,
    )


def _plot_pq_rank(frame: pd.DataFrame, path: Path) -> None:
    _bar_plot(
        frame,
        path,
        x_col="row_set",
        y_col="redundancy",
        title="P/Q Row-Composition Redundancy",
        ylabel="rows / states",
        log=False,
    )


def _plot_tiny_oracle(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if frame.empty:
        ax.text(0.5, 0.5, "No oracle rows", ha="center", va="center")
    else:
        values = [
            float(frame["transpiled_depth"].fillna(0).iloc[0]),
            float(frame["transpiled_cx_count"].fillna(0).iloc[0]),
            float(frame["total_qubits"].fillna(0).iloc[0]),
        ]
        ax.bar(["depth", "CX", "qubits"], values)
        ax.set_title("Tiny Sparse Oracle Circuit Resources")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_repeat(frame: pd.DataFrame, path: Path) -> None:
    _bar_plot(
        frame,
        path,
        x_col="case_name",
        y_col="relative_update_error",
        title="Integrated QSVT Repeat-Case Error",
        ylabel="relative update error",
        log=True,
    )


def _plot_alpha_tradeoff(frame: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
    if frame.empty:
        ax1.text(0.5, 0.5, "No alpha rows", ha="center", va="center")
    else:
        subset = frame[frame["rule_name"].astype(str).str.contains("oracle")].copy()
        grouped = subset.groupby("alpha", dropna=False).agg(
            rmse=("rmse", "median"),
            degree=("required_degree", "median"),
        )
        ax1.plot(grouped.index, grouped["rmse"], marker="o", label="RMSE")
        ax1.set_xscale("log")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("median RMSE")
        ax2 = ax1.twinx()
        ax2.plot(grouped.index, grouped["degree"], marker="s", color="#f58518", label="degree")
        ax2.set_ylabel("median required degree")
        ax1.set_title("Alpha RMSE / Degree Tradeoff")
        ax1.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_alpha_lcurve(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if frame.empty:
        ax.text(0.5, 0.5, "No alpha rows", ha="center", va="center")
    else:
        subset = frame[frame["rule_name"].astype(str).str.contains("lcurve")].drop_duplicates(
            ["case_name", "alpha"]
        )
        ax.scatter(subset["residual_norm"], subset["update_norm"], c=np.log10(subset["alpha"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("residual norm")
        ax.set_ylabel("update norm")
        ax.set_title("Discrete L-Curve Diagnostic")
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _bar_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    log: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    values = pd.to_numeric(frame[y_col], errors="coerce") if y_col in frame else pd.Series([])
    if frame.empty or values.dropna().empty:
        ax.text(0.5, 0.5, f"No {ylabel} rows", ha="center", va="center")
    else:
        labels = frame[x_col].astype(str).tolist()
        y = values.fillna(0.0).to_numpy()
        if log:
            y = np.maximum(y, 1.0e-16)
            ax.set_yscale("log")
        ax.bar(np.arange(len(y)), y)
        ax.set_xticks(np.arange(len(y)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _line_plot(
    frame: pd.DataFrame,
    path: Path,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if frame.empty:
        ax.text(0.5, 0.5, "No noise rows", ha="center", va="center")
    else:
        for label, group in frame.groupby(group_col, dropna=False):
            ax.plot(group[x_col], group[y_col], marker="o", label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run final TQE robustness audit package.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--max-phase-synthesis-degree", type=int, default=35)
    args = parser.parse_args(argv)
    run = run_final_robustness_audits(
        {
            "output_root": args.output_root,
            "max_phase_synthesis_degree": args.max_phase_synthesis_degree,
        }
    )
    print(f"Wrote final robustness audits to {run['output_dir']}")


if __name__ == "__main__":
    main()
