from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    name: str
    observable_type: str
    indices: tuple[int, ...]


def default_update_observables(dimension: int) -> list[ObservableSpec]:
    if int(dimension) <= 0:
        raise ValueError("dimension must be positive")
    return [
        ObservableSpec("component_0_probability", "basis_probability", (0,)),
        ObservableSpec(
            "component_1_probability",
            "basis_probability",
            (min(1, int(dimension) - 1),),
        ),
        ObservableSpec(
            "first_two_state_energy",
            "subset_probability",
            tuple(range(min(2, int(dimension)))),
        ),
    ]


def add_computational_basis_measurements(circuit: Any) -> Any:
    measured = circuit.copy()
    measured.measure_all()
    return measured


def sample_statevector_counts(
    statevector: np.ndarray,
    *,
    shots: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    values = np.asarray(statevector, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("statevector must be one-dimensional")
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    probabilities = np.abs(values) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("statevector has zero probability mass")
    probabilities = probabilities / total
    samples = rng.multinomial(shot_count, probabilities)
    return {int(index): int(count) for index, count in enumerate(samples) if count}


def exact_postselected_observable_values(
    statevector: np.ndarray,
    *,
    encoded_dimension: int,
    observables: Sequence[ObservableSpec],
) -> dict[str, float]:
    post_state, success_probability = postselected_encoded_state(
        statevector,
        encoded_dimension=encoded_dimension,
    )
    values = {"success_probability": success_probability}
    for observable in observables:
        values[observable.name] = _observable_value(post_state, observable)
    return values


def postselected_encoded_state(
    statevector: np.ndarray,
    *,
    encoded_dimension: int,
    eps: float = 1.0e-15,
) -> tuple[np.ndarray, float]:
    values = np.asarray(statevector, dtype=np.complex128)
    dimension = int(encoded_dimension)
    if values.ndim != 1 or dimension <= 0 or dimension > values.size:
        raise ValueError("encoded_dimension must select a valid prefix of statevector")
    prefix = values[:dimension]
    success_probability = float(np.sum(np.abs(prefix) ** 2))
    if success_probability <= float(eps):
        raise ValueError("postselected success probability is too small")
    return prefix / np.sqrt(success_probability), success_probability


def shot_observable_summary(
    counts: dict[int, int],
    *,
    encoded_dimension: int,
    observables: Sequence[ObservableSpec],
    exact_qsvt_values: dict[str, float],
    ridge_values: dict[str, float],
    shots: int,
    seed: int,
) -> list[dict[str, Any]]:
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    success_count = sum(count for index, count in counts.items() if index < encoded_dimension)
    success_estimate = success_count / float(shot_count)
    rows: list[dict[str, Any]] = [
        {
            "observable_name": "success_probability",
            "observable_type": "postselection_probability",
            "indices": "encoded_prefix",
            "shots": shot_count,
            "seed": int(seed),
            "true_qsvt_value": exact_qsvt_values["success_probability"],
            "shot_estimate": success_estimate,
            "standard_error": _bernoulli_se(exact_qsvt_values["success_probability"], shot_count),
            "absolute_sampling_error": abs(
                success_estimate - exact_qsvt_values["success_probability"]
            ),
            "ridge_reference_value": np.nan,
            "absolute_error_vs_ridge": np.nan,
            "notes": "success probability for the encoded output block",
        }
    ]
    for observable in observables:
        if success_count == 0:
            estimate = float("nan")
            standard_error = float("nan")
        else:
            selected = _count_selected_encoded_indices(counts, observable.indices)
            estimate = selected / float(success_count)
            standard_error = _bernoulli_se(exact_qsvt_values[observable.name], success_count)
        ridge_value = ridge_values[observable.name]
        rows.append(
            {
                "observable_name": observable.name,
                "observable_type": observable.observable_type,
                "indices": " ".join(str(index) for index in observable.indices),
                "shots": shot_count,
                "seed": int(seed),
                "true_qsvt_value": exact_qsvt_values[observable.name],
                "shot_estimate": estimate,
                "standard_error": standard_error,
                "absolute_sampling_error": abs(estimate - exact_qsvt_values[observable.name]),
                "ridge_reference_value": ridge_value,
                "absolute_error_vs_ridge": abs(estimate - ridge_value),
                "notes": "conditional on measuring the encoded output block",
            }
        )
    return rows


def observable_values_from_state(
    state: np.ndarray,
    observables: Sequence[ObservableSpec],
) -> dict[str, float]:
    values = np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-15:
        raise ValueError("state has zero norm")
    normalized = values / norm
    return {
        observable.name: _observable_value(normalized, observable) for observable in observables
    }


def _observable_value(state: np.ndarray, observable: ObservableSpec) -> float:
    if observable.observable_type == "basis_probability":
        return float(np.abs(state[observable.indices[0]]) ** 2)
    if observable.observable_type == "subset_probability":
        return float(np.sum(np.abs(state[list(observable.indices)]) ** 2))
    raise ValueError(f"unsupported observable type: {observable.observable_type}")


def _count_selected_encoded_indices(counts: dict[int, int], indices: tuple[int, ...]) -> int:
    selected = set(int(index) for index in indices)
    return sum(count for index, count in counts.items() if index in selected)


def _bernoulli_se(probability: float, shots: int) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    return float(np.sqrt(p * (1.0 - p) / max(int(shots), 1)))
