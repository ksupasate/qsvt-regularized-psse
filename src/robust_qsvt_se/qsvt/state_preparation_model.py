from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

EXACT_DENSE_LOADING_LIMITATION = (
    "Exact dense amplitude loading is used only for simulator validation and is "
    "not an efficient scalable state-preparation proof."
)
QRAM_LOADING_LIMITATION = (
    "QRAM-style state preparation is an assumed access model, not implemented hardware."
)


class StatePreparationModel(Enum):
    EXACT_DENSE_AMPLITUDE_LOADING = "exact_dense_amplitude_loading"
    QRAM_AMPLITUDE_ORACLE = "qram_amplitude_oracle"
    SPARSE_RESIDUAL_LOADING = "sparse_residual_loading"
    RESOURCE_ESTIMATE_ONLY = "resource_estimate_only"


@dataclass(frozen=True, slots=True)
class StatePreparationEstimate:
    dimension: int
    padded_dimension: int
    input_norm: float
    nonzero_count: int
    preparation_model: str
    index_qubits: int
    ancilla_qubits: int
    estimated_depth_proxy: int | None
    estimated_query_cost: int | None
    approximation_error: float | None
    assumptions: list[str]
    limitations: list[str]

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "dimension": self.dimension,
            "padded_dimension": self.padded_dimension,
            "input_norm": self.input_norm,
            "nonzero_count": self.nonzero_count,
            "preparation_model": self.preparation_model,
            "index_qubits": self.index_qubits,
            "ancilla_qubits": self.ancilla_qubits,
            "estimated_depth_proxy": self.estimated_depth_proxy,
            "estimated_query_cost": self.estimated_query_cost,
            "approximation_error": self.approximation_error,
            "assumptions": "; ".join(self.assumptions),
            "limitations": "; ".join(self.limitations),
        }
        if extra:
            row.update(extra)
        return row


def estimate_state_preparation(
    vector: np.ndarray,
    model: StatePreparationModel | str = StatePreparationModel.QRAM_AMPLITUDE_ORACLE,
    *,
    tolerance: float = 1.0e-12,
) -> StatePreparationEstimate:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    prep_model = _model(model)
    dimension = int(values.size)
    padded_dimension = _next_power_of_two(dimension)
    index_qubits = _qubits(padded_dimension)
    nonzero_count = int(np.count_nonzero(np.abs(values) > float(tolerance)))
    input_norm = float(np.linalg.norm(values))
    if prep_model is StatePreparationModel.EXACT_DENSE_AMPLITUDE_LOADING:
        assumptions = ["state amplitudes are loaded exactly in the simulator"]
        limitations = [EXACT_DENSE_LOADING_LIMITATION]
        depth = int(padded_dimension)
        queries = None
        error = 0.0
        ancilla = 1
    elif prep_model is StatePreparationModel.QRAM_AMPLITUDE_ORACLE:
        assumptions = [
            "qRAM amplitude oracle returns indexed residual amplitudes and normalization data"
        ]
        limitations = [QRAM_LOADING_LIMITATION, "data-loading cost is not synthesized"]
        depth = int(index_qubits**2 + 1)
        queries = 1
        error = None
        ancilla = 2
    elif prep_model is StatePreparationModel.SPARSE_RESIDUAL_LOADING:
        assumptions = ["only nonzero residual entries are addressed by a sparse loader"]
        limitations = [
            "sparse loader circuit is not synthesized",
            "normalization and alias-table construction are classical metadata",
        ]
        depth = int(max(1, nonzero_count) * max(1, index_qubits))
        queries = max(1, nonzero_count)
        error = None
        ancilla = 2
    else:
        assumptions = ["state preparation is counted as an abstract oracle call"]
        limitations = [
            "no amplitude-loading implementation is provided",
            "normalization is simulator metadata",
        ]
        depth = None
        queries = 1
        error = None
        ancilla = 1
    return StatePreparationEstimate(
        dimension=dimension,
        padded_dimension=int(padded_dimension),
        input_norm=input_norm,
        nonzero_count=nonzero_count,
        preparation_model=prep_model.value,
        index_qubits=int(index_qubits),
        ancilla_qubits=int(ancilla),
        estimated_depth_proxy=depth,
        estimated_query_cost=queries,
        approximation_error=error,
        assumptions=assumptions,
        limitations=limitations,
    )


def build_state_preparation_outputs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_state_preparation_model",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "seed": 123,
        "models": [
            StatePreparationModel.EXACT_DENSE_AMPLITUDE_LOADING.value,
            StatePreparationModel.QRAM_AMPLITUDE_ORACLE.value,
            StatePreparationModel.SPARSE_RESIDUAL_LOADING.value,
            StatePreparationModel.RESOURCE_ESTIMATE_ONLY.value,
        ],
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case in list(resolved["cases"]):
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": resolved["matrix_source"],
                "seed": int(resolved["seed"]),
            }
        )
        for model in list(resolved["models"]):
            estimate = estimate_state_preparation(system.r_tilde, model)
            rows.append(
                estimate.to_row(
                    {
                        "case": case,
                        "matrix_source": matrix_source,
                    }
                )
            )
    summary_csv = output_dir / "state_preparation_summary.csv"
    assumptions_md = output_dir / "state_preparation_assumptions.md"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    assumptions_md.write_text(_assumptions_markdown(), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "state_preparation_summary": str(summary_csv),
            "state_preparation_assumptions": str(assumptions_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame(rows),
        "artifacts": {
            "manifest": manifest,
            "state_preparation_summary": summary_csv,
            "state_preparation_assumptions": assumptions_md,
        },
    }


def _model(value: StatePreparationModel | str) -> StatePreparationModel:
    if isinstance(value, StatePreparationModel):
        return value
    return StatePreparationModel(str(value))


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _assumptions_markdown() -> str:
    return "\n".join(
        [
            "# State-Preparation Model Assumptions",
            "",
            EXACT_DENSE_LOADING_LIMITATION,
            "",
            QRAM_LOADING_LIMITATION,
            "",
            "- Sparse residual loading is a resource model over nonzero residual entries.",
            "- Resource-estimate-only rows assume one abstract state-preparation oracle call.",
            "- None of these rows implements a hardware state-preparation circuit.",
            "",
        ]
    )
