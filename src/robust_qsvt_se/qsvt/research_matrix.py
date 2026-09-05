from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.utils.seed import make_rng

DEFAULT_MEASUREMENT_CONFIG: dict[str, Any] = {
    "include_voltage_magnitudes": True,
    "include_p_injections": True,
    "include_q_injections": True,
    "include_p_branch_flows": True,
    "include_q_branch_flows": True,
    "voltage_std": 0.01,
    "injection_p_std": 0.03,
    "injection_q_std": 0.03,
    "flow_p_std": 0.02,
    "flow_q_std": 0.02,
    "weak_area_buses": [],
    "weak_area_std_multiplier": 1.0,
}

DEFAULT_LINEARIZATION_CONFIG: dict[str, Any] = {
    "angle_perturbation_std": 0.005,
    "voltage_perturbation_std": 0.005,
    "min_voltage_magnitude": 0.5,
}


@dataclass(frozen=True, slots=True)
class ResearchMatrix:
    matrix: np.ndarray
    normalized_matrix: np.ndarray
    metadata: dict[str, Any]

    @property
    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.normalized_matrix, compute_uv=False)


def load_research_matrix_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("research matrix config must contain a mapping")
    return validate_research_matrix_config(loaded)


def validate_research_matrix_config(config: dict[str, Any]) -> dict[str, Any]:
    matrix_config = dict(config.get("matrix", config))
    defaults: dict[str, Any] = {
        "case_name": "ieee14",
        "case_source": "pypower",
        "mode": "ac_weighted_jacobian",
        "matrix_scope": "submatrix",
        "use_full_matrix": False,
        "submatrix_size": 4,
        "target_shape": None,
        "seed": 123,
        "selection": "largest_column_then_row_norm",
        "selection_strategy": "high_leverage",
        "measurement": DEFAULT_MEASUREMENT_CONFIG,
        "linearization": DEFAULT_LINEARIZATION_CONFIG,
    }
    resolved = {**defaults, **matrix_config}
    resolved["measurement"] = {**DEFAULT_MEASUREMENT_CONFIG, **dict(resolved["measurement"])}
    resolved["linearization"] = {
        **DEFAULT_LINEARIZATION_CONFIG,
        **dict(resolved["linearization"]),
    }
    if str(resolved["mode"]) != "ac_weighted_jacobian":
        raise ValueError("matrix.mode must be ac_weighted_jacobian")
    if str(resolved["matrix_scope"]) not in {"full_matrix", "submatrix"}:
        raise ValueError("matrix.matrix_scope must be full_matrix or submatrix")
    if bool(resolved["use_full_matrix"]):
        resolved["matrix_scope"] = "full_matrix"
    if int(resolved["submatrix_size"]) <= 0:
        raise ValueError("matrix.submatrix_size must be positive")
    if resolved.get("target_shape") is not None:
        target_shape = list(resolved["target_shape"])
        if len(target_shape) != 2 or any(int(value) <= 0 for value in target_shape):
            raise ValueError("matrix.target_shape must contain two positive integers")
        resolved["target_shape"] = [int(target_shape[0]), int(target_shape[1])]
        if int(target_shape[0]) != int(target_shape[1]):
            raise ValueError("matrix.target_shape must be square for current QSVT circuit demos")
        resolved["submatrix_size"] = int(target_shape[0])
    if str(resolved["selection"]) not in {
        "largest_column_then_row_norm",
        "high_leverage",
        "largest_singular_vector_support",
    }:
        raise ValueError("matrix.selection must be a supported deterministic strategy")
    if str(resolved["selection_strategy"]) not in {
        "largest_column_then_row_norm",
        "high_leverage",
        "largest_singular_vector_support",
    }:
        raise ValueError("matrix.selection_strategy must be a supported deterministic strategy")
    return {"matrix": resolved}


def extract_research_matrix(config: dict[str, Any]) -> ResearchMatrix:
    resolved = validate_research_matrix_config(config)["matrix"]
    full = extract_weighted_jacobian_matrix(
        case_name=str(resolved["case_name"]),
        mode=str(resolved["mode"]),
        case_source=str(resolved["case_source"]),
        measurement_profile="default",
        normalize=False,
        seed=int(resolved["seed"]),
        measurement_config=dict(resolved["measurement"]),
        linearization_config=dict(resolved["linearization"]),
    )
    if str(resolved["matrix_scope"]) == "full_matrix":
        return _normalize_research_matrix(
            full.matrix,
            {
                **full.metadata,
                "used_shape": list(full.matrix.shape),
                "is_full_matrix": True,
                "matrix_scope": "full_matrix",
                "row_selection_strategy": "none_full_matrix",
                "column_selection_strategy": "none_full_matrix",
                "reason_for_submatrix": "",
                "selected_rows": list(range(full.matrix.shape[0])),
                "selected_columns": list(range(full.matrix.shape[1])),
                "selected_measurement_labels": full.metadata.get("measurement_labels", []),
                "selected_measurement_types": full.metadata.get("measurement_types", []),
                "selected_state_labels": full.metadata.get("state_labels", []),
            },
        )

    size = int(resolved["submatrix_size"])
    return extract_qsvt_submatrix(
        full,
        target_shape=(size, size),
        strategy=str(resolved.get("selection_strategy") or resolved["selection"]),
        seed=int(resolved["seed"]),
    )


def extract_weighted_jacobian_matrix(
    case_name: str,
    mode: str,
    case_source: str = "pypower",
    measurement_profile: str = "default",
    normalize: bool = True,
    seed: int = 123,
    measurement_config: dict[str, Any] | None = None,
    linearization_config: dict[str, Any] | None = None,
) -> ResearchMatrix:
    """Build the full weighted AC state-estimation Jacobian for an IEEE case."""
    if mode != "ac_weighted_jacobian":
        raise ValueError("Only ac_weighted_jacobian research matrices are supported")
    if measurement_profile != "default":
        raise ValueError("Only the default measurement profile is currently supported")
    rng = make_rng(int(seed))
    system = build_ac_weighted_system(
        case_name=case_name,
        case_source=case_source,
        linearization_config={**DEFAULT_LINEARIZATION_CONFIG, **dict(linearization_config or {})},
        measurement_config={**DEFAULT_MEASUREMENT_CONFIG, **dict(measurement_config or {})},
        rng=rng,
        metadata={"qsvt_research_matrix_source": True},
    )
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    measurement_labels = list(system.metadata.get("measurement_labels", []))
    measurement_types = list(system.metadata.get("measurement_types", []))
    state_labels = _state_labels(system.metadata)
    metadata = {
        "case_name": system.metadata.get("case_name"),
        "source_case_name": system.metadata.get("case_name"),
        "case_source": case_source,
        "dataset_source": system.metadata.get("dataset_source"),
        "dataset_source_detail": system.metadata.get("dataset_source_detail"),
        "matrix_type": "weighted_jacobian",
        "measurement_mode": mode,
        "measurement_profile": measurement_profile,
        "full_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "full_matrix_rows": int(matrix.shape[0]),
        "full_matrix_columns": int(matrix.shape[1]),
        "used_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "measurement_labels": measurement_labels,
        "measurement_types": measurement_types,
        "state_labels": state_labels,
        "angle_state_buses": list(system.metadata.get("angle_state_buses", [])),
        "voltage_state_buses": list(system.metadata.get("voltage_state_buses", [])),
        "row_selection_strategy": "none_full_matrix",
        "column_selection_strategy": "none_full_matrix",
        "is_full_matrix": True,
        "reason_for_submatrix": "",
    }
    if not normalize:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        metadata.update(_spectral_metadata(singular_values, normalization_factor=1.0))
        return ResearchMatrix(matrix=matrix, normalized_matrix=matrix, metadata=metadata)
    return _normalize_research_matrix(matrix, metadata)


def extract_qsvt_submatrix(
    matrix: ResearchMatrix | np.ndarray,
    target_shape: tuple[int, int],
    strategy: str = "high_leverage",
    seed: int = 123,
) -> ResearchMatrix:
    """Extract a deterministic QSVT-ready submatrix from a research Jacobian."""
    del seed  # deterministic strategies do not use randomness, but keep API stable.
    source_matrix = matrix.matrix if isinstance(matrix, ResearchMatrix) else np.asarray(matrix)
    source_metadata = dict(matrix.metadata) if isinstance(matrix, ResearchMatrix) else {}
    full_matrix = np.asarray(source_matrix, dtype=np.float64)
    if len(target_shape) != 2 or any(int(value) <= 0 for value in target_shape):
        raise ValueError("target_shape must contain two positive integers")
    row_count, column_count = int(target_shape[0]), int(target_shape[1])
    if row_count != column_count:
        raise ValueError("QSVT circuit submatrices must be square")
    if row_count > full_matrix.shape[0] or column_count > full_matrix.shape[1]:
        raise ValueError(
            f"target_shape={target_shape} exceeds weighted Jacobian shape {full_matrix.shape}"
        )
    strategy_name = _canonical_strategy(strategy)
    if strategy_name == "largest_singular_vector_support":
        selected_rows, selected_columns = _singular_vector_support_indices(
            full_matrix,
            row_count=row_count,
            column_count=column_count,
        )
        reason = (
            "Rows and columns selected by largest absolute support in leading left and "
            "right singular vectors of the weighted Jacobian."
        )
    else:
        selected_columns = _top_indices(np.linalg.norm(full_matrix, axis=0), column_count)
        row_scores = np.linalg.norm(full_matrix[:, selected_columns], axis=1)
        selected_rows = _top_indices(row_scores, row_count)
        reason = (
            "Columns selected by largest weighted Jacobian column norms; rows selected by "
            "largest row norms restricted to those columns."
        )
    submatrix = full_matrix[np.ix_(selected_rows, selected_columns)]
    measurement_labels = list(source_metadata.get("measurement_labels", []))
    measurement_types = list(source_metadata.get("measurement_types", []))
    state_labels = list(source_metadata.get("state_labels", []))
    metadata = {
        **source_metadata,
        "used_shape": [row_count, column_count],
        "submatrix_rows": row_count,
        "submatrix_columns": column_count,
        "is_full_matrix": False,
        "matrix_scope": "submatrix",
        "row_selection_strategy": strategy_name,
        "column_selection_strategy": strategy_name,
        "selected_rows": selected_rows.astype(int).tolist(),
        "selected_columns": selected_columns.astype(int).tolist(),
        "selected_measurement_labels": [
            measurement_labels[index] for index in selected_rows if index < len(measurement_labels)
        ],
        "selected_measurement_types": [
            measurement_types[index] for index in selected_rows if index < len(measurement_types)
        ],
        "selected_state_labels": [
            state_labels[index] for index in selected_columns if index < len(state_labels)
        ],
        "selection_reason": reason,
        "reason_for_submatrix": reason,
    }
    return _normalize_research_matrix(submatrix, metadata)


def singular_values_frame(research_matrix: ResearchMatrix) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "singular_index": np.arange(research_matrix.singular_values.size),
            "singular_value": research_matrix.singular_values,
        }
    )


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:count])


def _canonical_strategy(strategy: str) -> str:
    if strategy in {"largest_column_then_row_norm", "high_leverage"}:
        return "high_leverage"
    if strategy == "largest_singular_vector_support":
        return strategy
    raise ValueError(f"unsupported submatrix selection strategy: {strategy}")


def _singular_vector_support_indices(
    matrix: np.ndarray,
    *,
    row_count: int,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    selected_rows = _top_indices(np.abs(u[:, 0]), row_count)
    selected_columns = _top_indices(np.abs(vh[0, :]), column_count)
    return selected_rows, selected_columns


def _normalize_research_matrix(matrix: np.ndarray, metadata: dict[str, Any]) -> ResearchMatrix:
    raw = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(raw, compute_uv=False)
    normalization = max(float(singular_values[0]), np.finfo(float).eps)
    normalized = raw / normalization
    normalized_singular_values = np.linalg.svd(normalized, compute_uv=False)
    normalized_metadata = {
        **metadata,
        "normalization_factor": normalization,
        "spectral_norm_before": float(singular_values[0]) if singular_values.size else 0.0,
        "spectral_norm_after": (
            float(normalized_singular_values[0]) if normalized_singular_values.size else 0.0
        ),
        "spectral_norm_after_normalization": (
            float(normalized_singular_values[0]) if normalized_singular_values.size else 0.0
        ),
        "singular_values": normalized_singular_values.tolist(),
        "condition_number": _condition_number(normalized_singular_values),
    }
    normalized_metadata.setdefault("matrix_scope", "full_matrix")
    return ResearchMatrix(matrix=raw, normalized_matrix=normalized, metadata=normalized_metadata)


def _spectral_metadata(
    singular_values: np.ndarray,
    *,
    normalization_factor: float,
) -> dict[str, Any]:
    singular_values = np.asarray(singular_values, dtype=np.float64)
    return {
        "normalization_factor": normalization_factor,
        "spectral_norm_before": float(singular_values[0]) if singular_values.size else 0.0,
        "spectral_norm_after": float(singular_values[0]) if singular_values.size else 0.0,
        "spectral_norm_after_normalization": (
            float(singular_values[0]) if singular_values.size else 0.0
        ),
        "singular_values": singular_values.tolist(),
        "condition_number": _condition_number(singular_values),
    }


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values)[np.asarray(singular_values) > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())


def _state_labels(metadata: dict[str, Any]) -> list[str]:
    angle_buses = list(metadata.get("angle_state_buses", []))
    voltage_buses = list(metadata.get("voltage_state_buses", []))
    return [f"theta_{bus}" for bus in angle_buses] + [f"V_{bus}" for bus in voltage_buses]
