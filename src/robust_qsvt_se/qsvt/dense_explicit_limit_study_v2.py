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
from robust_qsvt_se.utils.io import ensure_directory

DENSE_LIMIT_V2_CLAIM = (
    "Dense explicit block encoding is used as a correctness and small-circuit "
    "evidence path. It is not treated as the scalable route for full IEEE matrices."
)


def run_dense_explicit_limit_study_v2(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_sizes": [4, 8, 16],
        "alpha": 1.0e-4,
        "degree": 35,
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/dense_explicit_qsvt_limit_study_v2",
        "transpile_qubit_limit": 3,
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    executed_rows: list[dict[str, Any]] = []
    construction_rows: list[dict[str, Any]] = []
    transpilation_rows: list[dict[str, Any]] = []
    for size in [int(value) for value in resolved["submatrix_sizes"]]:
        try:
            subproblem = extract_state_estimation_subproblem(
                case=str(resolved["case"]),
                model=str(resolved["model"]),
                submatrix_size=size,
                seed=int(resolved["seed"]),
                case_source=str(resolved["case_source"]),
            )
            computation = solve_gate_level_state_estimation_problem(
                H_tilde=subproblem.H_tilde,
                r_tilde=subproblem.r_tilde,
                alpha=float(resolved["alpha"]),
                degree=int(resolved["degree"]),
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                metadata=subproblem.metadata,
                transpile_qubit_limit=int(resolved["transpile_qubit_limit"]),
                export_qasm=False,
            )
            construction = _construction_row(size, computation)
            construction_rows.append(construction)
            transpilation_rows.append(_transpilation_row(size, computation))
            if construction["solver_validated"]:
                executed_rows.append(_executed_row(size, computation))
        except Exception as exc:  # pragma: no cover - resource/version dependent
            construction_rows.append(_failure_construction_row(size, exc))
            transpilation_rows.append(_failure_transpilation_row(size, exc))
    artifacts = _write_outputs(
        output_dir,
        resolved,
        executed_rows=executed_rows,
        construction_rows=construction_rows,
        transpilation_rows=transpilation_rows,
    )
    return {
        "output_dir": output_dir,
        "executed_rows": executed_rows,
        "construction_rows": construction_rows,
        "transpilation_rows": transpilation_rows,
        "artifacts": artifacts,
    }


def _executed_row(size: int, computation: Any) -> dict[str, Any]:
    summary = computation.summary
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "ancilla_qubits": 1,
        "transpiled_depth": computation.resource_row.get("depth_after_transpile"),
        "two_qubit_gates": summary["two_qubit_gate_count"],
        "state_error_vs_ridge": summary["state_error_vs_ridge"],
        "residual_no_update": summary["residual_before_update"],
        "residual_after_qsvt": summary["residual_after_qsvt_update"],
        "residual_after_ridge": summary["residual_after_ridge_update"],
        "residual_reduction_ratio": summary["residual_reduction_ratio_vs_no_update"],
        "success_probability": summary["success_probability"],
        "run_status": "validated_solver_evidence",
    }


def _construction_row(size: int, computation: Any) -> dict[str, Any]:
    resource = computation.resource_row
    summary = computation.summary
    transpile_success = bool(resource.get("transpile_success", False))
    residual_reduction = bool(
        summary["residual_after_qsvt_update"] < summary["residual_before_update"]
    )
    solver_validated = bool(transpile_success and residual_reduction)
    skipped_reason = (
        "" if transpile_success else str(resource.get("transpile_message", "not attempted"))
    )
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "ancilla_qubits": 1,
        "raw_depth": resource.get("depth_before_transpile"),
        "raw_gate_count": resource.get("gate_count_total_before_transpile"),
        "statevector_constructed": True,
        "transpilation_attempted": bool(transpile_success or "skipped" not in skipped_reason),
        "transpilation_skipped_reason": skipped_reason,
        "solver_validated": solver_validated,
        "residual_reduction_observed": residual_reduction,
        "run_status": "solver_validated" if solver_validated else "construction_resource_only",
    }


def _transpilation_row(size: int, computation: Any) -> dict[str, Any]:
    resource = computation.resource_row
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "transpile_success": bool(resource.get("transpile_success", False)),
        "transpile_message": resource.get("transpile_message", ""),
        "depth_before_transpile": resource.get("depth_before_transpile"),
        "depth_after_transpile": resource.get("depth_after_transpile"),
        "two_qubit_gates": computation.summary.get("two_qubit_gate_count"),
    }


def _failure_construction_row(size: int, exc: Exception) -> dict[str, Any]:
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "ancilla_qubits": 1,
        "raw_depth": np.nan,
        "raw_gate_count": np.nan,
        "statevector_constructed": False,
        "transpilation_attempted": False,
        "transpilation_skipped_reason": f"{type(exc).__name__}: {exc}",
        "solver_validated": False,
        "residual_reduction_observed": False,
        "run_status": "failed",
    }


def _failure_transpilation_row(size: int, exc: Exception) -> dict[str, Any]:
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "transpile_success": False,
        "transpile_message": f"{type(exc).__name__}: {exc}",
        "depth_before_transpile": np.nan,
        "depth_after_transpile": np.nan,
        "two_qubit_gates": np.nan,
    }


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    executed_rows: list[dict[str, Any]],
    construction_rows: list[dict[str, Any]],
    transpilation_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    executed_path = output_dir / "executed_solver_results.csv"
    construction_path = output_dir / "construction_resource_results.csv"
    transpilation_path = output_dir / "transpilation_feasibility.csv"
    interpretation_path = output_dir / "dense_limit_interpretation.md"
    pd.DataFrame(executed_rows).to_csv(executed_path, index=False)
    pd.DataFrame(construction_rows).to_csv(construction_path, index=False)
    pd.DataFrame(transpilation_rows).to_csv(transpilation_path, index=False)
    interpretation_path.write_text(
        _interpretation_markdown(executed_rows, construction_rows),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "executed_solver_results": str(executed_path),
            "construction_resource_results": str(construction_path),
            "transpilation_feasibility": str(transpilation_path),
            "dense_limit_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=DENSE_LIMIT_V2_CLAIM,
    )
    return {
        "manifest": manifest,
        "executed_solver_results": executed_path,
        "construction_resource_results": construction_path,
        "transpilation_feasibility": transpilation_path,
        "dense_limit_interpretation": interpretation_path,
    }


def _interpretation_markdown(
    executed_rows: list[dict[str, Any]],
    construction_rows: list[dict[str, Any]],
) -> str:
    executed_sizes = [str(row["submatrix_size"]) for row in executed_rows]
    construction_only = [
        str(row["submatrix_size"])
        for row in construction_rows
        if not bool(row.get("solver_validated", False))
    ]
    return "\n".join(
        [
            "# Dense Explicit QSVT Limit Study V2",
            "",
            DENSE_LIMIT_V2_CLAIM,
            "",
            f"- Executed validated solver sizes: {', '.join(executed_sizes) or 'none'}",
            f"- Construction/resource-only sizes: {', '.join(construction_only) or 'none'}",
            "- Raw circuit depth is reported separately from transpiled depth.",
            "- Non-reducing larger dense runs are retained as construction diagnostics, "
            "not solver evidence.",
            "",
        ]
    )
