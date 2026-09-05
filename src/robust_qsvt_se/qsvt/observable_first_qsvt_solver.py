from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import bernoulli_amplitude_estimate
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.partial_observable_readout import estimate_overlap_from_hadamard_proxy
from robust_qsvt_se.qsvt.power_observable_protocols import (
    FULL_VECTOR_REQUIRED,
    NORM_SCALED_OBSERVABLE,
    PROBABILITY_READOUT,
    build_observable_protocols,
    polynomial_action_update,
)
from robust_qsvt_se.qsvt.scale_recovery_protocols import compute_bounded_qsvt_action
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.utils.io import ensure_directory

CLAIM = (
    "Observable-first QSVT state-estimation solver prototype: it estimates selected "
    "power-system observables from the QSVT output instead of reconstructing the full update "
    "vector, using amplitude-based norm recovery and small-circuit readout on a selected "
    "IEEE14-derived subproblem. Ridge/Tikhonov remains the reference; no QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, quantum advantage, hardware execution, or solved readout "
    "is claimed."
)

CLAIM_ALLOWED = (
    "readout-aware QSVT state-estimation pathway on a selected IEEE14-derived subproblem"
)
CLAIM_DISALLOWED = "no quantum speedup, QSVT-over-Ridge superiority, or solved readout bottleneck"

SUMMARY_COLUMNS = [
    "observable_name",
    "physical_meaning",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "norm_recovery_method",
    "ridge_value",
    "qsvt_polynomial_value",
    "qsvt_gate_value_if_available",
    "estimated_readout_value",
    "absolute_error",
    "relative_error",
    "target_tolerance",
    "shots",
    "readout_query_cost",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_full_vector_readout",
    "practical_for_observable_first_solver",
    "claim_allowed",
    "claim_disallowed",
]


def run_observable_first_solver(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-5, 1.0e-4, 1.0e-3],
        "degrees": [35, 51, 75],
        "shots": [1000, 10000],
        "target_tolerances": [1.0e-1, 5.0e-2, 1.0e-2],
        "topk": 2,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "target_design": "current_global",
        "output_dir": "outputs/qsvt_observable_first_solver",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    rows = evaluate_observable_first_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        shot_levels=[int(value) for value in resolved["shots"]],
        target_tolerances=[float(value) for value in resolved["target_tolerances"]],
        topk=int(resolved["topk"]),
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        target_design=str(resolved["target_design"]),
        seed=int(resolved["seed"]),
    )
    artifacts = write_observable_first_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_observable_first_grid(
    *,
    subproblem: Any,
    alphas: list[float],
    degrees: list[int],
    shot_levels: list[int],
    target_tolerances: list[float],
    topk: int = 2,
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    target_design: str = "current_global",
    seed: int = 123,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
        protocols = build_observable_protocols(
            H_tilde=H, ridge_update=ridge_update, metadata=subproblem.metadata, topk=int(topk)
        )
        for degree in degrees:
            qsvt_update = polynomial_action_update(H, r, alpha=float(alpha), degree=int(degree))
            action = compute_bounded_qsvt_action(H, r, alpha=float(alpha), degree=int(degree))
            p_exact = float(action.success_probability_proxy)
            for shots in shot_levels:
                rng = np.random.default_rng(int(seed) + int(shots) + int(degree))
                recovered = _amplitude_recovered_update(
                    qsvt_update=qsvt_update,
                    p_exact=p_exact,
                    shots=int(shots),
                    seed=int(seed) + int(shots) + int(degree),
                )
                for protocol in protocols:
                    estimate, method = _readout_estimate(
                        protocol=protocol,
                        recovered_update=recovered,
                        shots=int(shots),
                        rng=rng,
                    )
                    ridge_value = _observable_value(protocol, ridge_update)
                    qsvt_value = _observable_value(protocol, qsvt_update)
                    for tolerance in target_tolerances:
                        rows.append(
                            _row(
                                protocol=protocol,
                                ridge_value=ridge_value,
                                qsvt_value=qsvt_value,
                                estimated_value=estimate,
                                method=method,
                                tolerance=float(tolerance),
                                shots=int(shots),
                                alpha=float(alpha),
                                degree=int(degree),
                                case=case,
                                model=model,
                                subproblem_id=subproblem_id,
                                target_design=target_design,
                            )
                        )
    return rows


def _amplitude_recovered_update(
    *,
    qsvt_update: np.ndarray,
    p_exact: float,
    shots: int,
    seed: int,
) -> np.ndarray:
    """Scale the known-C QSVT update by the Bernoulli amplitude estimate sqrt(p_hat / p_exact)."""

    if p_exact <= 1.0e-15:
        return np.asarray(qsvt_update, dtype=np.float64)
    estimate = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed))
    factor = math.sqrt(max(estimate.estimate, 0.0) / p_exact)
    return factor * np.asarray(qsvt_update, dtype=np.float64)


def _readout_estimate(
    *,
    protocol: Any,
    recovered_update: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, str]:
    update = np.asarray(recovered_update, dtype=np.float64)
    update_norm = float(np.linalg.norm(update))
    if protocol.protocol_type == FULL_VECTOR_REQUIRED:
        return float("nan"), "full_vector_not_supported"
    if protocol.kind == "topk_identification":
        identified = _topk_indices(update, len(protocol.state_indices))
        return _jaccard(identified, protocol.state_indices), "none"
    if protocol.protocol_type == PROBABILITY_READOUT:
        # generic probability readout: component magnitude probability
        index = protocol.state_indices[0] if protocol.state_indices else 0
        state = update / max(update_norm, 1.0e-15)
        probability = float(state[index] ** 2)
        successes = int(rng.binomial(int(shots), float(np.clip(probability, 0.0, 1.0))))
        return (successes / float(shots)) * update_norm**2, "bernoulli_success_amplitude"
    if protocol.protocol_type == NORM_SCALED_OBSERVABLE:
        state = update / max(update_norm, 1.0e-15)
        probability = float(np.sum(state[list(protocol.state_indices)] ** 2))
        successes = int(rng.binomial(int(shots), float(np.clip(probability, 0.0, 1.0))))
        return (successes / float(shots)) * update_norm**2, "bernoulli_success_amplitude"
    # Signed linear functional (signed_overlap / hadamard_test_proxy).
    coefficients = np.asarray(protocol.coefficients, dtype=np.float64)
    coefficient_norm = float(np.linalg.norm(coefficients))
    if coefficient_norm <= 1.0e-15 or update_norm <= 1.0e-15:
        return 0.0, "bernoulli_success_amplitude"
    overlap = float(np.dot(coefficients / coefficient_norm, update / update_norm))
    estimate, _ = estimate_overlap_from_hadamard_proxy(complex(overlap), int(shots), rng)
    return float(estimate) * update_norm * coefficient_norm, "bernoulli_success_amplitude"


def _row(
    *,
    protocol: Any,
    ridge_value: float,
    qsvt_value: float,
    estimated_value: float,
    method: str,
    tolerance: float,
    shots: int,
    alpha: float,
    degree: int,
    case: str,
    model: str,
    subproblem_id: str,
    target_design: str,
) -> dict[str, Any]:
    if math.isfinite(estimated_value):
        absolute = abs(estimated_value - ridge_value)
        relative = absolute / abs(ridge_value) if abs(ridge_value) > 1.0e-15 else float("nan")
    else:
        absolute = float("nan")
        relative = float("nan")
    return {
        "observable_name": protocol.observable_name,
        "physical_meaning": protocol.physical_meaning,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_design": target_design,
        "norm_recovery_method": method,
        "ridge_value": float(ridge_value),
        "qsvt_polynomial_value": float(qsvt_value),
        "qsvt_gate_value_if_available": float("nan"),
        "estimated_readout_value": float(estimated_value),
        "absolute_error": float(absolute),
        "relative_error": float(relative),
        "target_tolerance": float(tolerance),
        "shots": int(shots),
        "readout_query_cost": int(required_shots_for_additive_error(float(tolerance))),
        "requires_norm_recovery": bool(protocol.requires_norm_recovery),
        "requires_signed_overlap": bool(protocol.requires_signed_overlap),
        "requires_full_vector_readout": bool(protocol.requires_full_vector_readout),
        "practical_for_observable_first_solver": bool(not protocol.requires_full_vector_readout),
        "claim_allowed": CLAIM_ALLOWED,
        "claim_disallowed": CLAIM_DISALLOWED,
    }


def write_observable_first_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    summary_path = output_dir / "observable_first_solver_summary.csv"
    accuracy_path = output_dir / "observable_first_accuracy_cost.csv"
    interpretation_path = output_dir / "observable_first_interpretation.md"

    frame.to_csv(summary_path, index=False)
    frame[
        [
            "observable_name",
            "alpha",
            "degree",
            "norm_recovery_method",
            "absolute_error",
            "relative_error",
            "target_tolerance",
            "shots",
            "readout_query_cost",
            "requires_norm_recovery",
            "requires_full_vector_readout",
            "practical_for_observable_first_solver",
        ]
    ].to_csv(accuracy_path, index=False)
    interpretation_path.write_text(observable_first_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "observable_first_solver_summary": str(summary_path),
            "observable_first_accuracy_cost": str(accuracy_path),
            "observable_first_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "observable_first_solver_summary": summary_path,
        "observable_first_accuracy_cost": accuracy_path,
        "observable_first_interpretation": interpretation_path,
    }


def observable_first_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# Observable-First QSVT Solver", "", CLAIM])
    by_name: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        by_name.setdefault(row["observable_name"], row)
    practical = [name for name, row in by_name.items() if not row["requires_full_vector_readout"]]
    full_vector = [name for name, row in by_name.items() if row["requires_full_vector_readout"]]
    norm_dominated = [name for name, row in by_name.items() if row["requires_norm_recovery"]]
    practical_frame = frame[frame["practical_for_observable_first_solver"] == True]  # noqa: E712
    best_relative = (
        float(practical_frame["relative_error"].astype(float).replace([np.inf], np.nan).min())
        if not practical_frame.empty
        else float("nan")
    )
    return "\n".join(
        [
            "# Observable-First QSVT Solver",
            "",
            CLAIM,
            "",
            "## 1. Observables estimable without full-vector recovery",
            *[f"- `{name}`" for name in practical],
            "",
            "## 2. Observables that remain dominated by norm recovery",
            *[f"- `{name}`" for name in norm_dominated],
            "",
            "## 3. Observables requiring full-vector readout (not emphasized)",
            *[f"- `{name}`" for name in full_vector],
            "",
            "## Required Answers",
            f"1. Observables estimable without full-vector recovery: {len(practical)} of "
            f"{len(by_name)}.",
            f"2. Norm-recovery-dominated observables: {len(norm_dominated)} (signed functionals "
            "and selected-area energy require recovering the update norm/scale).",
            "3. Is the observable-first route more feasible than full update-vector recovery? "
            "Yes for selected observables: probability/top-k readouts avoid both norm recovery and "
            "signed overlap, while signed/energy observables need only a single scalar norm.",
            "4. Power quantities to emphasize: bus-angle/voltage updates, branch "
            "angle-difference proxies, selected-area energy, and top-k identification "
            f"(best practical relative error {best_relative:.3g}); full-vector reconstruction "
            "should not be emphasized.",
            "",
        ]
    )


def _observable_value(protocol: Any, update: np.ndarray) -> float:
    vector = np.asarray(update, dtype=np.float64)
    if protocol.kind == "subset_energy":
        return float(np.sum(vector[list(protocol.state_indices)] ** 2))
    if protocol.kind == "topk_identification":
        identified = _topk_indices(vector, len(protocol.state_indices))
        return _jaccard(identified, protocol.state_indices)
    if protocol.kind == "full_vector_reconstruction":
        return float(np.linalg.norm(vector))
    coefficients = protocol.coefficients
    if coefficients is None:
        return float("nan")
    return float(np.dot(np.asarray(coefficients, dtype=np.float64), vector))


def _topk_indices(values: np.ndarray, k: int) -> tuple[int, ...]:
    vector = np.asarray(values, dtype=np.float64)
    indices = np.arange(vector.size)
    order = np.lexsort((indices, -np.abs(vector)))
    return tuple(sorted(int(index) for index in order[: int(k)]))


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
