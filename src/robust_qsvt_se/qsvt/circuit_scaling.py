from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from robust_qsvt_se.qsvt.hardware_qsvt import run_explicit_hardware_qsvt
from robust_qsvt_se.qsvt.research_matrix import extract_research_matrix
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml
from robust_qsvt_se.utils.logging import configure_run_logger


def load_circuit_scaling_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("circuit scaling config must contain a mapping")
    return validate_circuit_scaling_config(loaded)


def validate_circuit_scaling_config(config: dict[str, Any]) -> dict[str, Any]:
    scaling = dict(config.get("scaling", config.get("demo", {})))
    defaults: dict[str, Any] = {
        "run_id": "qsvt_circuit_scaling",
        "output_dir": "outputs/qsvt_circuit_scaling",
        "cases": ["ieee14", "ieee30"],
        "sizes": [2, 4, 8, 16],
        "case_source": "pypower",
        "selection_strategy": "high_leverage",
        "seed": 123,
        "alpha": 0.05,
        "polynomial_degree": 5,
        "grid_size": 512,
        "angle_solver": "iterative",
        "phase_cache_dir": "outputs/qsvt_phase_cache",
        "transpile_basis_gates": ["rz", "sx", "x", "cx"],
        "transpile_optimization_level": 1,
        "max_simulated_size": 8,
        "evaluate_infeasible_matrices": True,
    }
    resolved = {**defaults, **scaling}
    if not isinstance(resolved["cases"], list) or not resolved["cases"]:
        raise ValueError("scaling.cases must be a non-empty list")
    if not isinstance(resolved["sizes"], list) or not resolved["sizes"]:
        raise ValueError("scaling.sizes must be a non-empty list")
    resolved["sizes"] = [int(size) for size in resolved["sizes"]]
    if any(size <= 0 for size in resolved["sizes"]):
        raise ValueError("scaling.sizes must be positive")
    degree = int(resolved["polynomial_degree"])
    if degree < 1 or degree % 2 == 0:
        raise ValueError("scaling.polynomial_degree must be a positive odd integer")
    if int(resolved["grid_size"]) <= degree + 1:
        raise ValueError("scaling.grid_size must be greater than polynomial_degree + 1")
    if int(resolved["max_simulated_size"]) <= 0:
        raise ValueError("scaling.max_simulated_size must be positive")
    return {"scaling": resolved}


def run_circuit_scaling(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_circuit_scaling_config(config)
    scaling = resolved["scaling"]
    output_dir = ensure_directory(Path(scaling["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting QSVT circuit scaling experiment %s", scaling["run_id"])
    rows: list[dict[str, Any]] = []

    for case_name in list(scaling["cases"]):
        for size in list(scaling["sizes"]):
            logger.info("Evaluating %s %sx%s", case_name, size, size)
            matrix_config = {
                "case_name": str(case_name),
                "case_source": str(scaling["case_source"]),
                "matrix_scope": "submatrix",
                "submatrix_size": int(size),
                "selection_strategy": str(scaling["selection_strategy"]),
                "seed": int(scaling["seed"]),
            }
            row = _base_row(scaling, case_name=str(case_name), size=int(size))
            try:
                research_matrix = extract_research_matrix({"matrix": matrix_config})
                row.update(
                    {
                        "full_matrix_shape": str(research_matrix.metadata.get("full_shape")),
                        "used_matrix_shape": str(research_matrix.metadata.get("used_shape")),
                        "condition_number": research_matrix.metadata.get("condition_number"),
                        "selection_strategy": research_matrix.metadata.get(
                            "row_selection_strategy"
                        ),
                        "spectral_norm_after": research_matrix.metadata.get("spectral_norm_after"),
                    }
                )
                if int(size) > int(scaling["max_simulated_size"]):
                    row.update(
                        {
                            "status": "infeasible",
                            "feasible": False,
                            "failure_reason": (
                                f"size {size} exceeds configured max_simulated_size="
                                f"{scaling['max_simulated_size']}"
                            ),
                        }
                    )
                    rows.append(row)
                    continue

                start = time.perf_counter()
                result = run_explicit_hardware_qsvt(
                    research_matrix.normalized_matrix,
                    alpha=float(scaling["alpha"]),
                    polynomial_degree=int(scaling["polynomial_degree"]),
                    grid_size=int(scaling["grid_size"]),
                    angle_solver=str(scaling["angle_solver"]),
                    phase_cache_dir=str(scaling["phase_cache_dir"]),
                    basis_gates=list(scaling["transpile_basis_gates"]),
                    transpile_optimization_level=int(scaling["transpile_optimization_level"]),
                )
                elapsed = time.perf_counter() - start
                row.update(
                    {
                        "status": "completed",
                        "feasible": True,
                        "failure_reason": "",
                        "qubits": result.summary.get("qubits"),
                        "polynomial_degree": result.summary.get("polynomial_degree"),
                        "phase_count": result.summary.get("n_phase_angles"),
                        "depth_before_transpile": result.summary.get("depth_before_transpile"),
                        "depth_after_transpile": result.summary.get("depth_after_transpile"),
                        "cx_count": result.summary.get("cx_count_after_transpile"),
                        "total_gate_count_after_transpile": result.summary.get(
                            "gate_count_total_after_transpile"
                        ),
                        "transpile_time_seconds": result.summary.get("transpile_seconds"),
                        "simulation_time_seconds": result.summary.get("simulation_seconds"),
                        "elapsed_seconds": elapsed,
                        "max_error_vs_classical": result.summary.get("max_error_vs_classical"),
                        "mean_error_vs_classical": result.summary.get("mean_error_vs_classical"),
                        "transpile_success": result.summary.get("transpile_success"),
                    }
                )
            except Exception as exc:  # preserve failed rows for the scaling evidence
                row.update(
                    {
                        "status": "failed",
                        "feasible": False,
                        "failure_reason": str(exc),
                    }
                )
                logger.exception("Scaling run failed for %s %sx%s", case_name, size, size)
            rows.append(row)

    results = pd.DataFrame(rows)
    summary = _summary_payload(scaling, results)
    artifacts = _write_artifacts(output_dir, resolved, results, summary)
    logger.info("Completed QSVT circuit scaling experiment %s", scaling["run_id"])
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "summary": summary,
        "results": results,
    }


def _base_row(config: dict[str, Any], *, case_name: str, size: int) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "case_name": case_name,
        "matrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "full_or_submatrix": "submatrix",
        "polynomial_degree": int(config["polynomial_degree"]),
        "phase_count": int(config["polynomial_degree"]) + 1,
        "selection_strategy": config["selection_strategy"],
        "qubits": None,
        "depth_before_transpile": None,
        "depth_after_transpile": None,
        "cx_count": None,
        "total_gate_count_after_transpile": None,
        "transpile_time_seconds": None,
        "simulation_time_seconds": None,
        "max_error_vs_classical": None,
        "mean_error_vs_classical": None,
        "status": "pending",
        "feasible": False,
        "failure_reason": "",
    }


def _summary_payload(config: dict[str, Any], results: pd.DataFrame) -> dict[str, Any]:
    completed = results[results["status"] == "completed"]
    return {
        "run_id": config["run_id"],
        "qsvt_construction_type": "explicit_block_encoding_qsvt_scaling",
        "implementation_scope": "small research-derived submatrix circuit scaling",
        "cases": list(config["cases"]),
        "sizes": list(config["sizes"]),
        "max_simulated_size": int(config["max_simulated_size"]),
        "polynomial_degree": int(config["polynomial_degree"]),
        "n_phase_angles": int(config["polynomial_degree"]) + 1,
        "basis_gates": list(config["transpile_basis_gates"]),
        "n_rows": len(results),
        "n_completed": int((results["status"] == "completed").sum()),
        "n_infeasible": int((results["status"] == "infeasible").sum()),
        "n_failed": int((results["status"] == "failed").sum()),
        "max_depth_after_transpile": _safe_max(completed, "depth_after_transpile"),
        "max_cx_count": _safe_max(completed, "cx_count"),
        "max_error_vs_classical": _safe_max(completed, "max_error_vs_classical"),
        "mean_error_vs_classical": _safe_mean(completed, "mean_error_vs_classical"),
        "is_dense_unitary_only": False,
        "scope_note": (
            "Scaling rows use deterministic IEEE weighted-Jacobian submatrices. "
            "Rows above max_simulated_size are recorded as infeasible instead of "
            "being silently skipped."
        ),
    }


def _safe_max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def _safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _write_artifacts(
    output_dir: Path,
    resolved_config: dict[str, Any],
    results: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    write_yaml(output_dir / "config_resolved.yaml", resolved_config)
    results.to_csv(output_dir / "circuit_scaling_results.csv", index=False)
    write_json(output_dir / "circuit_scaling_summary.json", summary)
    write_json(output_dir / "circuit_summary.json", summary)
    _plot_scaling(
        results,
        output_dir / "circuit_scaling_plot_depth.png",
        y_column="depth_after_transpile",
        y_label="Transpiled depth",
    )
    _plot_scaling(
        results,
        output_dir / "circuit_scaling_plot_cx.png",
        y_column="cx_count",
        y_label="CX count",
    )
    _plot_scaling(
        results,
        output_dir / "circuit_scaling_plot_error.png",
        y_column="max_error_vs_classical",
        y_label="Max error vs classical filter",
    )
    return {
        "config_resolved": str(output_dir / "config_resolved.yaml"),
        "circuit_scaling_results": str(output_dir / "circuit_scaling_results.csv"),
        "circuit_scaling_summary": str(output_dir / "circuit_scaling_summary.json"),
        "circuit_scaling_plot_depth": str(output_dir / "circuit_scaling_plot_depth.png"),
        "circuit_scaling_plot_cx": str(output_dir / "circuit_scaling_plot_cx.png"),
        "circuit_scaling_plot_error": str(output_dir / "circuit_scaling_plot_error.png"),
        "run_log": str(output_dir / "run.log"),
    }


def _plot_scaling(results: pd.DataFrame, path: Path, *, y_column: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    completed = results[results["status"] == "completed"].copy()
    if completed.empty or y_column not in completed:
        ax.text(0.5, 0.5, "No completed circuit rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        completed[y_column] = pd.to_numeric(completed[y_column], errors="coerce")
        for case_name, frame in completed.groupby("case_name", sort=True):
            sorted_frame = frame.sort_values("matrix_size")
            ax.plot(
                sorted_frame["matrix_size"],
                sorted_frame[y_column],
                marker="o",
                label=str(case_name),
            )
        ax.set_xlabel("Submatrix dimension")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run QSVT circuit scaling experiment")
    parser.add_argument("--config", required=True, help="Path to QSVT scaling config")
    args = parser.parse_args(argv)
    run_circuit_scaling(load_circuit_scaling_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
