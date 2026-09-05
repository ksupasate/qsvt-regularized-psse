from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.partial_observable_readout import estimate_overlap_from_hadamard_proxy
from robust_qsvt_se.qsvt.power_observable_mapping import (
    PowerObservable,
    build_power_observables,
)
from robust_qsvt_se.utils.io import ensure_directory

OBSERVABLE_ERROR_CLAIM = (
    "This diagnostic decomposes selected observable errors into update-state, "
    "norm/scaling, signed readout, shot-noise, and metadata components. It does "
    "not solve the full readout problem."
)

OBSERVABLE_ERROR_COLUMNS = [
    "observable_name",
    "observable_type",
    "state_indices",
    "physical_interpretation",
    "metadata_status",
    "ridge_value",
    "qsvt_raw_value",
    "qsvt_ridge_norm_value",
    "qsvt_best_scalar_value",
    "raw_absolute_error",
    "ridge_norm_absolute_error",
    "best_scalar_absolute_error",
    "shot_count",
    "shot_estimate",
    "shot_standard_error",
    "shot_absolute_error",
    "dominant_error_source",
    "readout_limitation",
]


def run_observable_error_decomposition(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "shots": [100, 1000, 10000, 100000],
        "seed": 123,
        "output_dir": "outputs/qsvt_observable_error_decomposition",
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
    computation = solve_gate_level_state_estimation_problem(
        H_tilde=subproblem.H_tilde,
        r_tilde=subproblem.r_tilde,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        shots=max(int(value) for value in resolved["shots"]),
        seed=int(resolved["seed"]),
        metadata=subproblem.metadata,
        transpile_qubit_limit=0,
        export_qasm=False,
    )
    rows = decompose_observable_errors(
        H_tilde=computation.H_tilde,
        r_tilde=computation.r_tilde,
        qsvt_update=computation.qsvt_update,
        ridge_update=computation.ridge_update,
        metadata=subproblem.metadata,
        shot_counts=[int(value) for value in resolved["shots"]],
        seed=int(resolved["seed"]),
    )
    artifacts = _write_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def decompose_observable_errors(
    *,
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    qsvt_update: np.ndarray,
    ridge_update: np.ndarray,
    metadata: dict[str, Any],
    shot_counts: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    observables = build_power_observables(H_tilde, metadata)
    qsvt = np.asarray(qsvt_update, dtype=np.float64)
    ridge = np.asarray(ridge_update, dtype=np.float64)
    ridge_norm = float(np.linalg.norm(ridge))
    qsvt_norm = float(np.linalg.norm(qsvt))
    qsvt_direction = qsvt / max(qsvt_norm, 1.0e-15)
    ridge_norm_update = ridge_norm * qsvt_direction
    best_scalar = best_scalar_for_residual(H_tilde, qsvt, r_tilde)
    best_scalar_update = best_scalar * qsvt
    rows: list[dict[str, Any]] = []
    for observable in observables:
        base = _base_observable_row(
            observable=observable,
            ridge_update=ridge,
            qsvt_update=qsvt,
            ridge_norm_update=ridge_norm_update,
            best_scalar_update=best_scalar_update,
        )
        for shots in shot_counts:
            rng = np.random.default_rng(int(seed) + int(shots) + len(rows))
            shot_estimate, shot_se = _shot_estimate(
                observable=observable,
                update=qsvt,
                shots=int(shots),
                rng=rng,
            )
            row = {
                **base,
                "shot_count": int(shots),
                "shot_estimate": shot_estimate,
                "shot_standard_error": shot_se,
                "shot_absolute_error": abs(float(shot_estimate) - float(base["ridge_value"])),
            }
            row["dominant_error_source"] = dominant_observable_error_source(row)
            row["readout_limitation"] = _readout_limitation(row)
            rows.append(row)
    return rows


def dominant_observable_error_source(row: dict[str, Any]) -> str:
    metadata_status = str(row["metadata_status"])
    raw_error = float(row["raw_absolute_error"])
    scaled_error = float(row["best_scalar_absolute_error"])
    shot_error = float(row["shot_absolute_error"])
    ridge_scale = max(abs(float(row["ridge_value"])), 1.0)
    if "metadata_unavailable" in metadata_status or "index_level" in metadata_status:
        return "metadata_or_index_level_interpretation"
    if scaled_error <= 0.25 * max(raw_error, 1.0e-15) and shot_error <= raw_error:
        return "norm_scaling_recovery"
    if scaled_error <= 1.0e-3 * ridge_scale and shot_error > 5.0 * scaled_error:
        return "shot_readout_cost"
    if scaled_error > 1.0e-3 * ridge_scale:
        return "update_state_approximation_or_signed_norm_recovery"
    return "low_exact_error_with_finite_shot_noise"


def _base_observable_row(
    *,
    observable: PowerObservable,
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
    ridge_norm_update: np.ndarray,
    best_scalar_update: np.ndarray,
) -> dict[str, Any]:
    ridge_value = observable_value(observable, ridge_update)
    qsvt_value = observable_value(observable, qsvt_update)
    ridge_norm_value = observable_value(observable, ridge_norm_update)
    best_scalar_value = observable_value(observable, best_scalar_update)
    return {
        "observable_name": observable.observable_name,
        "observable_type": observable.observable_type,
        "state_indices": " ".join(str(index) for index in observable.state_indices),
        "physical_interpretation": observable.physical_interpretation,
        "metadata_status": observable.metadata_status,
        "ridge_value": ridge_value,
        "qsvt_raw_value": qsvt_value,
        "qsvt_ridge_norm_value": ridge_norm_value,
        "qsvt_best_scalar_value": best_scalar_value,
        "raw_absolute_error": abs(qsvt_value - ridge_value),
        "ridge_norm_absolute_error": abs(ridge_norm_value - ridge_value),
        "best_scalar_absolute_error": abs(best_scalar_value - ridge_value),
    }


def observable_value(observable: PowerObservable, update: np.ndarray) -> float:
    vector = np.asarray(update, dtype=np.float64)
    if observable.observable_type == "selected_subset_update_energy":
        return float(np.sum(vector[list(observable.state_indices)] ** 2))
    if observable.coefficients is None:
        raise ValueError("linear observable requires coefficients")
    return float(np.dot(np.asarray(observable.coefficients, dtype=np.float64), vector))


def _shot_estimate(
    *,
    observable: PowerObservable,
    update: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    update = np.asarray(update, dtype=np.float64)
    norm = float(np.linalg.norm(update))
    state = update / max(norm, 1.0e-15)
    if observable.observable_type == "selected_subset_update_energy":
        probability = float(np.sum(state[list(observable.state_indices)] ** 2))
        probability = float(np.clip(probability, 0.0, 1.0))
        successes = int(rng.binomial(int(shots), probability))
        estimate = successes / float(shots)
        se = float(np.sqrt(probability * (1.0 - probability) / max(int(shots), 1)))
        return estimate * norm**2, se * norm**2
    coeffs = np.asarray(observable.coefficients, dtype=np.float64)
    coeff_norm = float(np.linalg.norm(coeffs))
    if coeff_norm <= 1.0e-15:
        return 0.0, 0.0
    overlap = complex(np.vdot(coeffs / coeff_norm, state))
    estimate, se = estimate_overlap_from_hadamard_proxy(overlap, int(shots), rng)
    return estimate * norm * coeff_norm, se * norm * coeff_norm


def _readout_limitation(row: dict[str, Any]) -> str:
    source = str(row["dominant_error_source"])
    if source == "shot_readout_cost":
        return "The main limitation is shot/readout cost."
    if source == "metadata_or_index_level_interpretation":
        return "The observable is index-level rather than physically mapped."
    if source == "norm_scaling_recovery":
        return "The main limitation is norm/scaling recovery for this observable."
    return "The main limitation is update-state approximation or signed/norm recovery."


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    decomposition_path = output_dir / "observable_error_decomposition.csv"
    scaling_path = output_dir / "observable_scaling_comparison.csv"
    shot_path = output_dir / "observable_shot_error_scaling.csv"
    interpretation_path = output_dir / "observable_error_interpretation.md"
    frame = pd.DataFrame(rows, columns=OBSERVABLE_ERROR_COLUMNS)
    frame.to_csv(decomposition_path, index=False)
    frame.drop_duplicates("observable_name")[
        [
            "observable_name",
            "observable_type",
            "metadata_status",
            "ridge_value",
            "qsvt_raw_value",
            "qsvt_ridge_norm_value",
            "qsvt_best_scalar_value",
            "raw_absolute_error",
            "ridge_norm_absolute_error",
            "best_scalar_absolute_error",
            "dominant_error_source",
        ]
    ].to_csv(scaling_path, index=False)
    frame[
        [
            "observable_name",
            "observable_type",
            "shot_count",
            "shot_estimate",
            "shot_standard_error",
            "shot_absolute_error",
            "dominant_error_source",
        ]
    ].to_csv(shot_path, index=False)
    interpretation_path.write_text(_interpretation_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "observable_error_decomposition": str(decomposition_path),
            "observable_scaling_comparison": str(scaling_path),
            "observable_shot_error_scaling": str(shot_path),
            "observable_error_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=OBSERVABLE_ERROR_CLAIM,
    )
    return {
        "manifest": manifest,
        "observable_error_decomposition": decomposition_path,
        "observable_scaling_comparison": scaling_path,
        "observable_shot_error_scaling": shot_path,
        "observable_error_interpretation": interpretation_path,
    }


def _interpretation_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        max_raw = max_scaled = max_shot = float("nan")
        sources: dict[str, int] = {}
    else:
        max_raw = float(frame["raw_absolute_error"].max())
        max_scaled = float(frame["best_scalar_absolute_error"].max())
        max_shot = float(frame["shot_absolute_error"].max())
        sources = frame["dominant_error_source"].value_counts().to_dict()
    return "\n".join(
        [
            "# Observable Error Decomposition",
            "",
            OBSERVABLE_ERROR_CLAIM,
            "",
            f"- Max raw observable error: {max_raw:.17g}",
            f"- Max best-scalar observable error: {max_scaled:.17g}",
            f"- Max shot observable error: {max_shot:.17g}",
            f"- Dominant error-source counts: {sources}",
            "",
            "These observables avoid full-vector tomography and provide a partial "
            "readout path for selected state-estimation quantities. They do not "
            "solve the full readout problem.",
            "",
        ]
    )
