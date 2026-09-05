from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.partial_observable_readout import (
    estimate_overlap_from_hadamard_proxy,
)
from robust_qsvt_se.utils.io import ensure_directory

POWER_OBSERVABLE_CLAIM = (
    "These observables avoid full-vector tomography and provide a partial "
    "readout path for selected state-estimation quantities. They do not solve "
    "the full readout problem."
)


@dataclass(frozen=True, slots=True)
class PowerObservable:
    observable_name: str
    observable_type: str
    state_indices: tuple[int, ...]
    coefficients: np.ndarray | None
    physical_interpretation: str
    metadata_status: str


def build_power_observables(H_tilde: np.ndarray, metadata: dict[str, Any]) -> list[PowerObservable]:
    """Build grid-interpretable observables when selected-state metadata supports it."""

    dimension = int(np.asarray(H_tilde).shape[1])
    records = list(metadata.get("state_index_mapping", []))
    observables: list[PowerObservable] = []
    angle_records = [record for record in records if record.get("state_type") == "angle"]
    voltage_records = [record for record in records if record.get("state_type") == "voltage"]
    if angle_records:
        record = angle_records[0]
        local = int(record["local_index"])
        observables.append(
            _linear_component_observable(
                name=f"bus_{record.get('bus_id', record.get('label'))}_angle_update",
                index=local,
                dimension=dimension,
                observable_type="bus_angle_update",
                interpretation=f"Delta theta update for {record.get('label')}",
                metadata_status="mapped_to_state_metadata",
            )
        )
    if voltage_records:
        record = voltage_records[0]
        local = int(record["local_index"])
        observables.append(
            _linear_component_observable(
                name=f"bus_{record.get('bus_id', record.get('label'))}_voltage_update",
                index=local,
                dimension=dimension,
                observable_type="voltage_magnitude_update",
                interpretation=f"Delta V update for {record.get('label')}",
                metadata_status="mapped_to_state_metadata",
            )
        )
    if len(angle_records) >= 2:
        first = int(angle_records[0]["local_index"])
        second = int(angle_records[1]["local_index"])
        coeffs = np.zeros(dimension, dtype=np.float64)
        coeffs[first] = 1.0
        coeffs[second] = -1.0
        observables.append(
            PowerObservable(
                observable_name="branch_angle_difference_proxy",
                observable_type="branch_angle_difference_proxy",
                state_indices=(first, second),
                coefficients=coeffs,
                physical_interpretation=(
                    f"{angle_records[0].get('label')} - {angle_records[1].get('label')} "
                    "angle-update proxy"
                ),
                metadata_status="mapped_to_state_metadata",
            )
        )
    if not observables:
        observables.append(
            _linear_component_observable(
                name="component_0_update",
                index=0,
                dimension=dimension,
                observable_type="generic_component_update",
                interpretation="Generic selected update entry; physical state metadata unavailable",
                metadata_status="metadata_unavailable_or_generic",
            )
        )
    subset = tuple(range(min(2, dimension)))
    observables.append(
        PowerObservable(
            observable_name="selected_subset_update_energy",
            observable_type="selected_subset_update_energy",
            state_indices=subset,
            coefficients=None,
            physical_interpretation="Squared update energy on the first selected local entries",
            metadata_status=(
                "mapped_to_state_metadata" if records else "metadata_unavailable_or_generic"
            ),
        )
    )
    row_index = 0
    row = np.asarray(H_tilde, dtype=np.float64)[row_index]
    observables.append(
        PowerObservable(
            observable_name="measurement_row_0_correction_proxy",
            observable_type="measurement_row_correction_proxy",
            state_indices=tuple(np.flatnonzero(np.abs(row) > 1.0e-12).astype(int).tolist()),
            coefficients=row.copy(),
            physical_interpretation="Linearized correction h_0^T Delta x for selected row 0",
            metadata_status="measurement_row_index_level",
        )
    )
    return observables


def evaluate_power_observables(
    *,
    H_tilde: np.ndarray,
    qsvt_update: np.ndarray,
    ridge_update: np.ndarray,
    metadata: dict[str, Any],
    shot_counts: list[int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observables = build_power_observables(H_tilde, metadata)
    metadata_rows = [_metadata_row(observable) for observable in observables]
    exact_rows = [
        _exact_row(observable, qsvt_update=qsvt_update, ridge_update=ridge_update, shot_count=0)
        for observable in observables
    ]
    shot_rows: list[dict[str, Any]] = []
    for shot_count in shot_counts:
        rng = np.random.default_rng(int(seed) + int(shot_count))
        for observable in observables:
            shot_rows.append(
                _shot_row(
                    observable,
                    qsvt_update=qsvt_update,
                    ridge_update=ridge_update,
                    shots=int(shot_count),
                    rng=rng,
                )
            )
    return metadata_rows, exact_rows, shot_rows


def run_power_observable_readout(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "shots": [100, 1000, 10000],
        "seed": 123,
        "output_dir": "outputs/qsvt_power_observable_readout",
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
    metadata_rows, exact_rows, shot_rows = evaluate_power_observables(
        H_tilde=computation.H_tilde,
        qsvt_update=computation.qsvt_update,
        ridge_update=computation.ridge_update,
        metadata=subproblem.metadata,
        shot_counts=[int(value) for value in resolved["shots"]],
        seed=int(resolved["seed"]),
    )
    artifacts = _write_outputs(output_dir, resolved, metadata_rows, exact_rows, shot_rows)
    return {
        "output_dir": output_dir,
        "metadata_rows": metadata_rows,
        "exact_rows": exact_rows,
        "shot_rows": shot_rows,
        "artifacts": artifacts,
    }


def _linear_component_observable(
    *,
    name: str,
    index: int,
    dimension: int,
    observable_type: str,
    interpretation: str,
    metadata_status: str,
) -> PowerObservable:
    coeffs = np.zeros(int(dimension), dtype=np.float64)
    coeffs[int(index)] = 1.0
    return PowerObservable(
        observable_name=name,
        observable_type=observable_type,
        state_indices=(int(index),),
        coefficients=coeffs,
        physical_interpretation=interpretation,
        metadata_status=metadata_status,
    )


def _metadata_row(observable: PowerObservable) -> dict[str, Any]:
    return {
        "observable_name": observable.observable_name,
        "observable_type": observable.observable_type,
        "state_indices": " ".join(str(index) for index in observable.state_indices),
        "physical_interpretation": observable.physical_interpretation,
        "metadata_status": observable.metadata_status,
    }


def _exact_row(
    observable: PowerObservable,
    *,
    qsvt_update: np.ndarray,
    ridge_update: np.ndarray,
    shot_count: int,
) -> dict[str, Any]:
    exact_qsvt = _observable_value(observable, qsvt_update)
    exact_ridge = _observable_value(observable, ridge_update)
    absolute = abs(exact_qsvt - exact_ridge)
    return {
        **_metadata_row(observable),
        "exact_ridge_value": exact_ridge,
        "exact_qsvt_value": exact_qsvt,
        "absolute_error": absolute,
        "relative_error_if_defined": absolute / abs(exact_ridge)
        if abs(exact_ridge) > 1.0e-15
        else np.nan,
        "shot_count": int(shot_count),
        "shot_estimate": np.nan,
        "shot_standard_error": np.nan,
    }


def _shot_row(
    observable: PowerObservable,
    *,
    qsvt_update: np.ndarray,
    ridge_update: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    exact = _exact_row(
        observable,
        qsvt_update=qsvt_update,
        ridge_update=ridge_update,
        shot_count=int(shots),
    )
    qsvt_norm = float(np.linalg.norm(qsvt_update))
    qsvt_state = qsvt_update / max(qsvt_norm, 1.0e-15)
    if observable.observable_type == "selected_subset_update_energy":
        probability = float(np.sum(np.abs(qsvt_state[list(observable.state_indices)]) ** 2))
        estimate, se_prob = _estimate_probability(probability, int(shots), rng)
        exact["shot_estimate"] = estimate * qsvt_norm**2
        exact["shot_standard_error"] = se_prob * qsvt_norm**2
    else:
        coeffs = _normalized_coefficients(observable)
        overlap = complex(np.vdot(coeffs, qsvt_state))
        estimate, se = estimate_overlap_from_hadamard_proxy(overlap, int(shots), rng)
        coeff_norm = _coefficient_norm(observable)
        exact["shot_estimate"] = estimate * qsvt_norm * coeff_norm
        exact["shot_standard_error"] = se * qsvt_norm * coeff_norm
    exact["absolute_shot_error"] = abs(float(exact["shot_estimate"]) - exact["exact_qsvt_value"])
    return exact


def _observable_value(observable: PowerObservable, update: np.ndarray) -> float:
    vector = np.asarray(update, dtype=np.float64)
    if observable.observable_type == "selected_subset_update_energy":
        return float(np.sum(vector[list(observable.state_indices)] ** 2))
    coeffs = np.asarray(observable.coefficients, dtype=np.float64)
    return float(np.dot(coeffs, vector))


def _normalized_coefficients(observable: PowerObservable) -> np.ndarray:
    coeffs = np.asarray(observable.coefficients, dtype=np.float64)
    norm = float(np.linalg.norm(coeffs))
    if norm <= 1.0e-15:
        raise ValueError("linear observable coefficients must have nonzero norm")
    return coeffs / norm


def _coefficient_norm(observable: PowerObservable) -> float:
    if observable.coefficients is None:
        return 1.0
    return float(np.linalg.norm(observable.coefficients))


def _estimate_probability(
    probability: float,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    p = float(np.clip(probability, 0.0, 1.0))
    successes = int(rng.binomial(int(shots), p))
    estimate = successes / float(shots)
    se = float(np.sqrt(p * (1.0 - p) / max(int(shots), 1)))
    return estimate, se


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    metadata_rows: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
    shot_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    metadata_path = output_dir / "observable_metadata.csv"
    exact_path = output_dir / "observable_exact_vs_qsvt.csv"
    shot_path = output_dir / "observable_shot_scaling.csv"
    interpretation_path = output_dir / "power_observable_interpretation.md"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(exact_rows).to_csv(exact_path, index=False)
    pd.DataFrame(shot_rows).to_csv(shot_path, index=False)
    interpretation_path.write_text(
        _interpretation_markdown(exact_rows, shot_rows),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "observable_metadata": str(metadata_path),
            "observable_exact_vs_qsvt": str(exact_path),
            "observable_shot_scaling": str(shot_path),
            "power_observable_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=POWER_OBSERVABLE_CLAIM,
    )
    return {
        "manifest": manifest,
        "observable_metadata": metadata_path,
        "observable_exact_vs_qsvt": exact_path,
        "observable_shot_scaling": shot_path,
        "power_observable_interpretation": interpretation_path,
    }


def _interpretation_markdown(
    exact_rows: list[dict[str, Any]],
    shot_rows: list[dict[str, Any]],
) -> str:
    max_exact = max((float(row["absolute_error"]) for row in exact_rows), default=float("nan"))
    max_shot = max(
        (float(row.get("absolute_shot_error", 0.0)) for row in shot_rows),
        default=float("nan"),
    )
    return "\n".join(
        [
            "# Power-System Observable Readout",
            "",
            POWER_OBSERVABLE_CLAIM,
            "",
            f"- Observables evaluated: {len(exact_rows)}",
            f"- Max exact observable error: {max_exact:.17g}",
            f"- Max shot proxy error: {max_shot:.17g}",
            "- Physical labels are only used when selected state metadata maps the "
            "local component to angle or voltage entries.",
            "",
        ]
    )
