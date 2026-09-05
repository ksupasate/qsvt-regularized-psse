from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import bernoulli_amplitude_estimate
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.codesigned_bounded_targets import (
    CodesignedSolution,
    build_codesigned_solution,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.gate_state_preparation import normalize_and_pad_for_gate_preparation
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.utils.io import ensure_directory

CODESIGNED_GATE_CLAIM = (
    "Dense gate-level validation of co-designed QSVT-safe bounded targets on a "
    "selected IEEE14-derived subproblem. Phases are synthesized directly from the "
    "co-designed bounded coefficients and the structured QSVT operator circuit is "
    "simulated exactly; the success amplitude is estimated with the implemented "
    "small-circuit routine and used to recover the update scale. This is "
    "selected-subproblem dense simulator evidence, not full IEEE-scale hardware "
    "execution, and claims no QSVT superiority over Ridge/Tikhonov, quantum speedup, "
    "or quantum advantage."
)

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
MAX_MANAGEABLE_DEGREE = 101

GATE_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "deployability_class",
    "qsvt_safe",
    "gate_status",
    "phase_synthesis_status",
    "gate_residual_raw",
    "gate_residual_scaled",
    "ridge_residual",
    "residual_ratio_vs_no_update",
    "state_error_gate_vs_polynomial",
    "state_error_gate_vs_ridge",
    "direction_error_gate_vs_ridge",
    "success_probability_exact",
    "success_probability_estimated",
    "circuit_depth",
    "two_qubit_gates",
    "residual_feasible_after_gate",
    "dominant_limitation",
]


def run_qsvt_codesigned_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_residual_feasible_codesigned_search/"
        "deployable_residual_feasible_configs.csv",
        "fallback_input": "outputs/qsvt_residual_feasible_codesigned_search/"
        "diagnostic_feasible_configs.csv",
        "max_configs": 3,
        "shots": 1000,
        "seed": 123,
        "case_source": "pypower",
        "submatrix_size": 4,
        "grid_size": 4096,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_codesigned_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    feasible_frame = _read_csv(Path(resolved["input"]))
    fallback_frame = _read_csv(Path(resolved["fallback_input"]))
    selected, source = select_codesigned_gate_configs(
        feasible_frame, fallback_frame, max_configs=int(resolved["max_configs"])
    )

    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        rows.append(
            validate_single_codesigned_gate_config(
                row=record,
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                case_source=str(resolved["case_source"]),
                default_submatrix_size=int(resolved["submatrix_size"]),
                grid_size=int(resolved["grid_size"]),
                phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
                source=source,
            )
        )

    artifacts = write_codesigned_gate_outputs(output_dir, resolved, selected, rows, source=source)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "source": source,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_codesigned_gate_configs(
    feasible_frame: pd.DataFrame,
    fallback_frame: pd.DataFrame,
    *,
    max_configs: int,
    max_manageable_degree: int = MAX_MANAGEABLE_DEGREE,
) -> tuple[pd.DataFrame, str]:
    for frame, source in (
        (feasible_frame, "deployable_residual_feasible"),
        (fallback_frame, "diagnostic_feasible"),
    ):
        if frame is None or frame.empty:
            continue
        eligible = _eligible(frame, max_manageable_degree)
        if not eligible.empty:
            return _rank(eligible).head(int(max_configs)).reset_index(drop=True), source
    columns = (
        list(feasible_frame.columns)
        if feasible_frame is not None and len(feasible_frame.columns)
        else (list(fallback_frame.columns) if fallback_frame is not None else [])
    )
    return pd.DataFrame(columns=columns), "none"


def validate_single_codesigned_gate_config(
    *,
    row: dict[str, Any],
    shots: int,
    seed: int,
    case_source: str = "pypower",
    default_submatrix_size: int = 4,
    grid_size: int = 4096,
    phase_timeout_seconds: int = 40,
    source: str = "deployable_residual_feasible",
) -> dict[str, Any]:
    base = _base_row(row)
    try:
        case = str(row.get("case", "ieee14"))
        model = str(row.get("model", "ac_linearized"))
        alpha = float(row["alpha"])
        degree = int(float(row.get("degree", row.get("requested_degree", 35))))
        target_family = str(row.get("target_family", "weighted_support_ls"))
        submatrix_size = int(default_submatrix_size)

        subproblem = extract_state_estimation_subproblem(
            case=case,
            model=model,
            submatrix_size=submatrix_size,
            seed=int(seed),
            case_source=str(case_source),
        )
        H = np.asarray(subproblem.H_tilde, dtype=np.float64)
        r = np.asarray(subproblem.r_tilde, dtype=np.float64)
        no_update = float(np.linalg.norm(r))
        ridge_update = ridge_tikhonov_update(H, r, alpha=alpha)
        ridge_residual = _residual(H, ridge_update, r)

        solution = build_codesigned_solution(
            subproblem, alpha=alpha, degree=degree, target_family=target_family, grid_size=grid_size
        )
        if solution is None:
            base.update(
                {
                    "gate_status": "failed",
                    "phase_synthesis_status": "polynomial_construction_failed",
                    "dominant_limitation": "polynomial_construction",
                }
            )
            return base

        gate = _build_codesigned_gate_circuit(
            H=H,
            r=r,
            alpha=alpha,
            solution=solution,
            phase_timeout_seconds=int(phase_timeout_seconds),
        )
        if gate.get("status") != "synthesized":
            base.update(
                {
                    "ridge_residual": float(ridge_residual),
                    "gate_status": "not_run",
                    "phase_synthesis_status": gate.get("status", "runtime_limited"),
                    "success_probability_exact": float(solution.success_probability_proxy),
                    "deployability_class": solution.deployability_class,
                    "qsvt_safe": bool(solution.qsvt_safe),
                    "dominant_limitation": "phase_synthesis_runtime_limited",
                }
            )
            return base

        gate_update = gate["gate_update"]
        p_exact = float(gate["success_probability_exact"])
        gate_residual_raw = _residual(H, gate_update, r)

        estimate = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed))
        scale_factor = (
            math.sqrt(max(float(estimate.estimate), 0.0) / p_exact) if p_exact > 1.0e-15 else 1.0
        )
        scaled_update = scale_factor * gate_update
        gate_residual_scaled = _residual(H, scaled_update, r)
        ratio = gate_residual_scaled / no_update if no_update > 0.0 else float("nan")

        state_error_poly = _relative_error(gate_update, solution.known_c_update)
        state_error_ridge = _relative_error(scaled_update, ridge_update)
        direction_error = _direction_error(scaled_update, ridge_update)

        deployable = solution.deployability_class in DEPLOYABLE_CLASSES
        feasible_after_gate = (
            math.isfinite(ratio) and ratio <= RESIDUAL_RATIO_FEASIBLE_MAX and deployable
        )
        dominant = _dominant_limitation(
            ratio=ratio,
            direction_error=direction_error,
            deployable=deployable,
            source=source,
        )

        base.update(
            {
                "target_family": target_family,
                "weighting_scheme": solution.weighting_scheme,
                "deployability_class": solution.deployability_class,
                "qsvt_safe": bool(solution.qsvt_safe),
                "gate_status": "completed",
                "phase_synthesis_status": "synthesized",
                "gate_residual_raw": float(gate_residual_raw),
                "gate_residual_scaled": float(gate_residual_scaled),
                "ridge_residual": float(ridge_residual),
                "residual_ratio_vs_no_update": float(ratio),
                "state_error_gate_vs_polynomial": float(state_error_poly),
                "state_error_gate_vs_ridge": float(state_error_ridge),
                "direction_error_gate_vs_ridge": float(direction_error),
                "success_probability_exact": float(p_exact),
                "success_probability_estimated": float(estimate.estimate),
                "circuit_depth": int(gate["circuit_depth"]),
                "two_qubit_gates": int(gate["two_qubit_gates"]),
                "residual_feasible_after_gate": bool(feasible_after_gate),
                "dominant_limitation": dominant,
            }
        )
        return base
    except Exception as exc:
        base.update(
            {
                "gate_status": "failed",
                "phase_synthesis_status": "error",
                "dominant_limitation": f"error:{type(exc).__name__}",
            }
        )
        return base


def _build_codesigned_gate_circuit(
    *,
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    solution: CodesignedSolution,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    try:
        import pennylane as qml  # type: ignore[import-not-found]
        from qiskit.quantum_info import Operator
    except Exception as exc:
        return {"status": f"runtime_limited:{type(exc).__name__}"}

    coefficients = np.asarray(solution.bounded_coefficients, dtype=np.float64)
    try:
        validate_qsvt_polynomial(coefficients, parity="odd", grid_size=8193, bound_tolerance=1.0e-4)
    except Exception as exc:
        return {"status": f"validation_failed:{type(exc).__name__}"}

    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    try:
        from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import _phase_timeout

        with _phase_timeout(int(phase_timeout_seconds)):
            phases = np.asarray(
                qml.poly_to_angles(coefficients, "QSVT", angle_solver="iterative"),
                dtype=np.float64,
            )
    except Exception as exc:
        return {"status": f"runtime_limited:{type(exc).__name__}"}
    if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
        return {"status": "runtime_limited:invalid_phases"}

    operator = build_structured_qsvt_operator_circuit(
        block_encoding.unitary, phases, encoded_dimension=A.shape[0]
    )
    qsvt_unitary = np.asarray(Operator(operator.qsvt_operator_circuit).data)
    state_preparation = normalize_and_pad_for_gate_preparation(
        r, target_dimension=block_encoding.unitary.shape[0]
    )
    output = qsvt_unitary @ state_preparation.padded_state
    dimension = int(A.shape[0])
    prefix = output[:dimension]
    success_probability = float(np.sum(np.abs(prefix) ** 2))
    r_norm = float(np.linalg.norm(r))
    gate_update = (solution.c_scale / beta) * r_norm * np.real(prefix)
    return {
        "status": "synthesized",
        "gate_update": gate_update,
        "success_probability_exact": success_probability,
        "circuit_depth": int(operator.qsvt_operator_circuit.depth()),
        "two_qubit_gates": int(operator.qsvt_operator_circuit.num_nonlocal_gates()),
    }


def _dominant_limitation(
    *, ratio: float, direction_error: float, deployable: bool, source: str
) -> str:
    if math.isfinite(ratio) and ratio <= RESIDUAL_RATIO_FEASIBLE_MAX:
        if not deployable:
            return "diagnostic_method_not_deployable"
        return "none"
    if not math.isfinite(direction_error) or direction_error > 0.25:
        return "gate_extraction_direction"
    return "polynomial_target_or_readout"


def write_codesigned_gate_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    selected: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Path]:
    results = pd.DataFrame(rows)
    for column in GATE_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results = results[GATE_COLUMNS]

    selected_path = output_dir / "selected_codesigned_gate_configs.csv"
    results_path = output_dir / "codesigned_gate_validation_results.csv"
    interpretation_path = output_dir / "codesigned_gate_validation_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(
        codesigned_gate_interpretation(results, source=source), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "selected_codesigned_gate_configs": str(selected_path),
            "codesigned_gate_validation_results": str(results_path),
            "codesigned_gate_validation_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CODESIGNED_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "selected_codesigned_gate_configs": selected_path,
        "codesigned_gate_validation_results": results_path,
        "codesigned_gate_validation_interpretation": interpretation_path,
    }


def codesigned_gate_interpretation(frame: pd.DataFrame, *, source: str) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# Co-Designed Gate-Level Validation",
                "",
                CODESIGNED_GATE_CLAIM,
                "",
                "- No configurations were available for gate validation.",
                "",
            ]
        )
    completed = frame[frame["gate_status"] == "completed"]
    feasible = completed[completed["residual_feasible_after_gate"] == True]  # noqa: E712
    best_residual = (
        float(pd.to_numeric(completed["gate_residual_scaled"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )
    best_ratio = (
        float(pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )

    if source == "deployable_residual_feasible" and not feasible.empty:
        headline = (
            "A deployable co-designed configuration is residual-feasible after dense gate-level "
            "validation: the work may be classified as "
            "solver_prototype_ready_selected_subproblems."
        )
    elif source in {"deployable_residual_feasible", "diagnostic_feasible"} and feasible.empty:
        headline = (
            "No configuration remained residual-feasible after gate validation: the co-designed "
            "target family did not close the residual-feasibility gap under current constraints "
            "at the gate level."
        )
    else:
        headline = (
            "Only fallback/diagnostic configurations were validated; the work remains an "
            "engineering framework with diagnostic solver evidence."
        )

    return "\n".join(
        [
            "# Co-Designed Gate-Level Validation",
            "",
            CODESIGNED_GATE_CLAIM,
            "",
            f"Selection source: {source}.",
            "",
            "## Counts",
            f"- Configurations validated: {len(frame)}",
            f"- Completed gate runs: {len(completed)}",
            f"- Residual-feasible after gate: {len(feasible)}",
            "",
            "## Best Gate Result",
            f"- Best gate residual (scaled): {best_residual:.6g}",
            f"- Best residual_ratio_vs_no_update: {best_ratio:.6g}",
            "",
            "## Headline",
            f"- {headline}",
            "",
        ]
    )


def _eligible(frame: pd.DataFrame, max_manageable_degree: int) -> pd.DataFrame:
    if "degree" not in frame.columns:
        return frame.copy()
    degree = pd.to_numeric(frame["degree"], errors="coerce")
    return frame[degree.notna() & (degree <= max_manageable_degree)].copy()


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ("residual_ratio_vs_no_update", "direction_error_vs_ridge", "degree")
        if column in frame.columns
    ]
    ranked = frame.copy()
    for column in sort_columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    if sort_columns:
        ranked = ranked.sort_values(by=sort_columns, kind="stable")
    dedup_columns = [
        column
        for column in ("subproblem_id", "alpha", "degree", "target_family")
        if column in ranked.columns
    ]
    if dedup_columns:
        ranked = ranked.drop_duplicates(subset=dedup_columns)
    return ranked


def _base_row(row: dict[str, Any]) -> dict[str, Any]:
    base = {column: np.nan for column in GATE_COLUMNS}
    base.update(
        {
            "case": row.get("case", "ieee14"),
            "model": row.get("model", "ac_linearized"),
            "subproblem_id": row.get("subproblem_id", "selected_subproblem"),
            "alpha": _as_float(row.get("alpha")),
            "degree": _as_int(row.get("degree", row.get("requested_degree", 0))),
            "target_family": row.get("target_family", "weighted_support_ls"),
            "weighting_scheme": row.get("weighting_scheme", ""),
            "deployability_class": row.get("deployability_class", ""),
            "qsvt_safe": _as_bool(row.get("qsvt_safe")),
            "gate_status": "not_run",
            "phase_synthesis_status": "not_run",
            "residual_feasible_after_gate": False,
            "dominant_limitation": "not_run",
        }
    )
    return base


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    if not np.all(np.isfinite(update)):
        return float("nan")
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate - reference) / max(float(np.linalg.norm(reference)), 1.0e-15)
    )


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - np.clip(cosine, -1.0, 1.0)))))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}
