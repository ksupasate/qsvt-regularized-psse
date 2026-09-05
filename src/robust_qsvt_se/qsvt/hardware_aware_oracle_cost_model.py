from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

HARDWARE_AWARE_ORACLE_CLAIM = (
    "Full IEEE-scale QSVT remains an oracle-model pathway. This cost model "
    "estimates the resources required under sparse-access, value-loading, and "
    "state-preparation assumptions; it does not constitute hardware execution."
)

ORACLE_COST_COLUMNS = [
    "case",
    "matrix_shape",
    "nnz",
    "density",
    "max_row_sparsity",
    "max_col_sparsity",
    "row_qubits",
    "col_qubits",
    "value_precision_bits",
    "work_qubits",
    "ancilla_qubits",
    "index_lookup_model",
    "value_loading_model",
    "rotation_synthesis_model",
    "state_preparation_model",
    "estimated_row_lookup_cost",
    "estimated_value_loading_cost",
    "estimated_rotation_cost",
    "estimated_uncompute_cost",
    "estimated_block_encoding_query_cost",
    "qsvt_degree",
    "signal_unitary_calls",
    "projector_phase_operations",
    "alternating_sequence_length",
    "qsvt_query_count",
    "success_probability_proxy",
    "amplitude_amplification_proxy",
    "estimated_total_query_cost",
    "estimated_total_gate_cost_proxy",
    "observable_readout_shots",
    "resource_status",
    "dominant_cost_source",
    "assumption_risk_level",
]


def build_hardware_aware_oracle_cost_model(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "seed": 123,
        "value_precision_bits": [8, 12, 16],
        "degrees": [35, 51, 101],
        "observable_readout_shots": [1000, 10000],
        "index_lookup_model": "table_lookup_or_qrom_proxy",
        "value_loading_model": "fixed_point_value_register",
        "rotation_synthesis_model": "clifford_t_proxy",
        "state_preparation_model": "sparse_residual_loading",
        "amplitude_amplification_enabled": True,
        "output_dir": "outputs/hardware_aware_oracle_cost_model",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    rows = oracle_cost_rows(
        cases=[str(value) for value in resolved["cases"]],
        case_source=str(resolved["case_source"]),
        matrix_source=str(resolved["matrix_source"]),
        seed=int(resolved["seed"]),
        value_precision_bits=[int(value) for value in resolved["value_precision_bits"]],
        degrees=[int(value) for value in resolved["degrees"]],
        observable_readout_shots=[int(value) for value in resolved["observable_readout_shots"]],
        index_lookup_model=str(resolved["index_lookup_model"]),
        value_loading_model=str(resolved["value_loading_model"]),
        rotation_synthesis_model=str(resolved["rotation_synthesis_model"]),
        state_preparation_model=str(resolved["state_preparation_model"]),
        amplitude_amplification_enabled=bool(resolved["amplitude_amplification_enabled"]),
    )
    artifacts = write_oracle_cost_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def oracle_cost_rows(
    *,
    cases: list[str],
    case_source: str,
    matrix_source: str,
    seed: int,
    value_precision_bits: list[int],
    degrees: list[int],
    observable_readout_shots: list[int],
    index_lookup_model: str,
    value_loading_model: str,
    rotation_synthesis_model: str,
    state_preparation_model: str,
    amplitude_amplification_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            system, _matrix_source = build_engineering_system(
                {
                    "case_name": case,
                    "case_source": case_source,
                    "matrix_source": matrix_source,
                    "seed": int(seed),
                }
            )
            matrix = np.asarray(system.H_tilde, dtype=np.float64)
            stats = matrix_sparse_stats(matrix)
            for precision in value_precision_bits:
                for degree in degrees:
                    for shots in observable_readout_shots:
                        rows.append(
                            estimate_oracle_cost_row(
                                case=case,
                                stats=stats,
                                value_precision_bits=int(precision),
                                qsvt_degree=int(degree),
                                observable_readout_shots=int(shots),
                                index_lookup_model=index_lookup_model,
                                value_loading_model=value_loading_model,
                                rotation_synthesis_model=rotation_synthesis_model,
                                state_preparation_model=state_preparation_model,
                                amplitude_amplification_enabled=amplitude_amplification_enabled,
                                resource_status="oracle_model_estimate",
                            )
                        )
        except Exception as exc:  # pragma: no cover - case availability depends on optional data
            for precision in value_precision_bits:
                for degree in degrees:
                    for shots in observable_readout_shots:
                        rows.append(
                            failure_oracle_cost_row(
                                case=case,
                                value_precision_bits=int(precision),
                                qsvt_degree=int(degree),
                                observable_readout_shots=int(shots),
                                failure=f"{type(exc).__name__}: {exc}",
                                index_lookup_model=index_lookup_model,
                                value_loading_model=value_loading_model,
                                rotation_synthesis_model=rotation_synthesis_model,
                                state_preparation_model=state_preparation_model,
                            )
                        )
    return rows


def matrix_sparse_stats(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    nonzero = np.abs(values) > float(tolerance)
    nnz = int(np.count_nonzero(nonzero))
    rows, cols = values.shape
    row_counts = np.count_nonzero(nonzero, axis=1)
    col_counts = np.count_nonzero(nonzero, axis=0)
    return {
        "matrix_shape": (int(rows), int(cols)),
        "nnz": nnz,
        "density": float(nnz / max(values.size, 1)),
        "max_row_sparsity": int(np.max(row_counts)) if row_counts.size else 0,
        "max_col_sparsity": int(np.max(col_counts)) if col_counts.size else 0,
    }


def estimate_oracle_cost_row(
    *,
    case: str,
    stats: dict[str, Any],
    value_precision_bits: int,
    qsvt_degree: int,
    observable_readout_shots: int,
    index_lookup_model: str,
    value_loading_model: str,
    rotation_synthesis_model: str,
    state_preparation_model: str,
    amplitude_amplification_enabled: bool,
    resource_status: str,
) -> dict[str, Any]:
    rows, cols = stats["matrix_shape"]
    row_qubits = _qubits(rows)
    col_qubits = _qubits(cols)
    sparsity_qubits = _qubits(max(1, stats["max_row_sparsity"], stats["max_col_sparsity"]))
    precision = int(value_precision_bits)
    work_qubits = int(row_qubits + col_qubits + sparsity_qubits + precision)
    ancilla_qubits = int(4 + np.ceil(np.log2(max(precision, 2))))
    row_lookup_cost = _index_lookup_cost(
        rows=rows,
        cols=cols,
        nnz=int(stats["nnz"]),
        max_sparsity=int(max(stats["max_row_sparsity"], stats["max_col_sparsity"], 1)),
        row_qubits=row_qubits,
        col_qubits=col_qubits,
        model=index_lookup_model,
    )
    value_loading_cost = _value_loading_cost(
        precision=precision,
        max_sparsity=int(max(stats["max_row_sparsity"], stats["max_col_sparsity"], 1)),
        model=value_loading_model,
    )
    rotation_cost = _rotation_cost(precision=precision, model=rotation_synthesis_model)
    uncompute_cost = row_lookup_cost + value_loading_cost
    block_query_cost = row_lookup_cost + value_loading_cost + rotation_cost + uncompute_cost
    signal_unitary_calls = int(qsvt_degree)
    projector_phase_operations = int(qsvt_degree) + 1
    alternating_sequence_length = 2 * int(qsvt_degree) + 1
    success_proxy = _success_probability_proxy(stats["density"], int(qsvt_degree), precision)
    amplification = (
        float(1.0 / np.sqrt(max(success_proxy, 1.0e-300)))
        if amplitude_amplification_enabled
        else 1.0
    )
    total_query_cost = float(block_query_cost * signal_unitary_calls * amplification)
    readout_cost = int(observable_readout_shots) * max(1, row_qubits + col_qubits)
    total_gate_cost = float(total_query_cost + readout_cost)
    costs = {
        "row_lookup": row_lookup_cost,
        "value_loading": value_loading_cost,
        "rotation": rotation_cost,
        "uncompute": uncompute_cost,
        "readout": readout_cost,
    }
    dominant = max(costs, key=costs.get)
    return {
        "case": case,
        "matrix_shape": f"{rows}x{cols}",
        "nnz": int(stats["nnz"]),
        "density": float(stats["density"]),
        "max_row_sparsity": int(stats["max_row_sparsity"]),
        "max_col_sparsity": int(stats["max_col_sparsity"]),
        "row_qubits": row_qubits,
        "col_qubits": col_qubits,
        "value_precision_bits": precision,
        "work_qubits": work_qubits,
        "ancilla_qubits": ancilla_qubits,
        "index_lookup_model": index_lookup_model,
        "value_loading_model": value_loading_model,
        "rotation_synthesis_model": rotation_synthesis_model,
        "state_preparation_model": state_preparation_model,
        "estimated_row_lookup_cost": float(row_lookup_cost),
        "estimated_value_loading_cost": float(value_loading_cost),
        "estimated_rotation_cost": float(rotation_cost),
        "estimated_uncompute_cost": float(uncompute_cost),
        "estimated_block_encoding_query_cost": float(block_query_cost),
        "qsvt_degree": int(qsvt_degree),
        "signal_unitary_calls": signal_unitary_calls,
        "projector_phase_operations": projector_phase_operations,
        "alternating_sequence_length": alternating_sequence_length,
        # Backward-compatible alias: a query here means one signal-unitary call.
        "qsvt_query_count": signal_unitary_calls,
        "success_probability_proxy": success_proxy,
        "amplitude_amplification_proxy": amplification,
        "estimated_total_query_cost": total_query_cost,
        "estimated_total_gate_cost_proxy": total_gate_cost,
        "observable_readout_shots": int(observable_readout_shots),
        "resource_status": resource_status,
        "dominant_cost_source": dominant,
        "assumption_risk_level": _risk_level(case, precision, state_preparation_model),
    }


def failure_oracle_cost_row(
    *,
    case: str,
    value_precision_bits: int,
    qsvt_degree: int,
    observable_readout_shots: int,
    failure: str,
    index_lookup_model: str,
    value_loading_model: str,
    rotation_synthesis_model: str,
    state_preparation_model: str,
) -> dict[str, Any]:
    row = {column: np.nan for column in ORACLE_COST_COLUMNS}
    row.update(
        {
            "case": case,
            "matrix_shape": "",
            "value_precision_bits": int(value_precision_bits),
            "index_lookup_model": index_lookup_model,
            "value_loading_model": value_loading_model,
            "rotation_synthesis_model": rotation_synthesis_model,
            "state_preparation_model": state_preparation_model,
            "qsvt_degree": int(qsvt_degree),
            "signal_unitary_calls": int(qsvt_degree),
            "projector_phase_operations": int(qsvt_degree) + 1,
            "alternating_sequence_length": 2 * int(qsvt_degree) + 1,
            "qsvt_query_count": int(qsvt_degree),
            "observable_readout_shots": int(observable_readout_shots),
            "resource_status": f"failed: {failure}",
            "dominant_cost_source": "unavailable",
            "assumption_risk_level": "high",
        }
    )
    return row


def write_oracle_cost_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = pd.DataFrame(rows)
    for column in ORACLE_COST_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    by_case_path = output_dir / "oracle_cost_by_case.csv"
    by_precision_path = output_dir / "oracle_cost_by_precision.csv"
    total_path = output_dir / "qsvt_total_cost_estimate.csv"
    amplification_path = output_dir / "amplification_adjusted_cost.csv"
    assumptions_path = output_dir / "oracle_cost_assumption_summary.md"
    frame.sort_values(["case", "value_precision_bits", "qsvt_degree"], kind="mergesort").to_csv(
        by_case_path,
        index=False,
    )
    frame.sort_values(["value_precision_bits", "case", "qsvt_degree"], kind="mergesort").to_csv(
        by_precision_path,
        index=False,
    )
    frame.to_csv(total_path, index=False)
    frame[
        [
            "case",
            "value_precision_bits",
            "qsvt_degree",
            "qsvt_query_count",
            "success_probability_proxy",
            "amplitude_amplification_proxy",
            "estimated_total_query_cost",
            "estimated_total_gate_cost_proxy",
            "assumption_risk_level",
        ]
    ].to_csv(amplification_path, index=False)
    assumptions_path.write_text(oracle_cost_assumption_summary(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "oracle_cost_by_case": str(by_case_path),
            "oracle_cost_by_precision": str(by_precision_path),
            "qsvt_total_cost_estimate": str(total_path),
            "amplification_adjusted_cost": str(amplification_path),
            "oracle_cost_assumption_summary": str(assumptions_path),
        },
        input_config=resolved,
        claim_boundary=HARDWARE_AWARE_ORACLE_CLAIM,
        seed_provenance={
            "status": "deterministic_no_rng",
            "seed": None,
            "source": "code inspection and recorded command",
            "applies_to": (
                "closed-form oracle/QROM cost counting over fixed case matrices; "
                "no stochastic RNG anywhere in the generator"
            ),
            "regeneration_command": "python scripts/build_hardware_aware_oracle_cost_model.py",
        },
    )
    return {
        "manifest": manifest,
        "oracle_cost_by_case": by_case_path,
        "oracle_cost_by_precision": by_precision_path,
        "qsvt_total_cost_estimate": total_path,
        "amplification_adjusted_cost": amplification_path,
        "oracle_cost_assumption_summary": assumptions_path,
    }


def oracle_cost_assumption_summary(frame: pd.DataFrame) -> str:
    cases = sorted(str(value) for value in frame["case"].dropna().unique())
    precisions = sorted(int(value) for value in frame["value_precision_bits"].dropna().unique())
    worst_block = float(frame["estimated_block_encoding_query_cost"].max(skipna=True))
    worst_total = float(frame["estimated_total_gate_cost_proxy"].max(skipna=True))
    dominant = (
        frame["dominant_cost_source"].mode().iloc[0]
        if "dominant_cost_source" in frame and not frame.empty
        else "unknown"
    )
    return "\n".join(
        [
            "# Hardware-Aware Sparse-Oracle Cost Model",
            "",
            HARDWARE_AWARE_ORACLE_CLAIM,
            "",
            "## Evidence Categories",
            "- Implemented toy sparse-oracle circuits: only small 4x4 reversible lookup/"
            "value-rotation demos.",
            "- Software sparse-access abstractions: classical CSR/CSC row, column, and "
            "value access.",
            "- Full IEEE-scale oracle-model estimates: cost rows under sparse-access, "
            "value-loading, "
            "state-preparation, amplification, and readout assumptions.",
            "- Missing hardware-native sparse-oracle synthesis: no full IEEE sparse oracle "
            "has been "
            "compiled into a calibrated hardware-native circuit.",
            "",
            f"- IEEE cases included: {', '.join(cases)}",
            f"- Value precision bits: {precisions}",
            f"- Dominant cost source: {dominant}",
            f"- Worst estimated block-encoding query cost: {worst_block:.17g}",
            f"- Worst amplification-adjusted total gate-cost proxy: {worst_total:.17g}",
            "",
            "Highest-risk assumptions are sparse row/column lookup, precision-controlled value "
            "loading, scalable residual state preparation, and hardware-native oracle synthesis.",
            "",
        ]
    )


def _index_lookup_cost(
    *,
    rows: int,
    cols: int,
    nnz: int,
    max_sparsity: int,
    row_qubits: int,
    col_qubits: int,
    model: str,
) -> float:
    if model == "table_lookup_or_qrom_proxy":
        return float(max_sparsity * (row_qubits + col_qubits) + np.ceil(np.log2(max(nnz, 2))))
    return float(max(rows, cols) * max(1, row_qubits + col_qubits))


def _value_loading_cost(*, precision: int, max_sparsity: int, model: str) -> float:
    if model == "multiplexed_rotation":
        return float(max_sparsity * precision)
    return float(max_sparsity * precision + precision**2)


def _rotation_cost(*, precision: int, model: str) -> float:
    if model == "cx_depth_proxy":
        return float(8 * precision)
    return float(3 * precision**2)


def _success_probability_proxy(density: float, degree: int, precision: int) -> float:
    density_factor = float(np.clip(density, 1.0e-6, 1.0))
    degree_factor = 1.0 / np.sqrt(max(int(degree), 1))
    precision_factor = 1.0 - 2.0 ** (-max(int(precision), 1))
    return float(np.clip(density_factor * degree_factor * precision_factor, 1.0e-12, 1.0))


def _risk_level(case: str, precision: int, state_preparation_model: str) -> str:
    if case in {"ieee118", "ieee300"} or int(precision) >= 16:
        return "high"
    if "dense" in state_preparation_model:
        return "high"
    return "moderate"


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))
