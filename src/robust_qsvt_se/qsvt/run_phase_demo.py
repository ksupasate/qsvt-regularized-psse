from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from robust_qsvt_se.qsvt.circuit import qsp_circuit_summary
from robust_qsvt_se.qsvt.phase_synthesis import (
    qsp_response,
    synthesize_pennylane_phases_cached,
    synthesize_qsp_phases,
)
from robust_qsvt_se.qsvt.polynomial import (
    OddPolynomialApproximation,
    fit_odd_regularized_polynomial,
    regularized_filter_on_normalized_domain,
)
from robust_qsvt_se.qsvt.qsp_validation import validate_pennylane_qsvt_phases
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.logging import configure_run_logger


def load_phase_demo_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("QSVT phase demo config must contain a mapping")
    return validate_phase_demo_config(config)


def validate_phase_demo_config(config: dict[str, Any]) -> dict[str, Any]:
    demo = dict(config.get("demo", config))
    defaults: dict[str, Any] = {
        "run_id": "qsvt_phase_demo",
        "output_dir": "outputs/qsvt_phase_demo",
        "alpha": 0.01,
        "degree": 11,
        "domain_min": 0.05,
        "domain_max": 1.0,
        "block_encoding_normalization": 1.0,
        "grid_size": 2048,
        "seed": 123,
        "max_nfev": 4000,
        "phase_synthesis_method": "auto",
        "angle_solver": "root-finding",
        "phase_cache_dir": "outputs/qsvt_phase_cache",
        "phase_validation_grid_size": 129,
        "require_phase_validation": False,
        "polynomial_max_error_target": 1.0e-2,
        "polynomial_mean_error_target": 1.0e-3,
        "phase_max_error_target": 5.0e-2,
        "phase_mean_error_target": 1.0e-2,
        "singular_values": [0.1, 0.25, 0.5, 0.85],
    }
    resolved = {**defaults, **demo}
    if not isinstance(resolved["run_id"], str) or not resolved["run_id"]:
        raise ValueError("demo.run_id must be a non-empty string")
    if not isinstance(resolved["output_dir"], str) or not resolved["output_dir"]:
        raise ValueError("demo.output_dir must be a non-empty string")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("demo.alpha must be positive")
    if int(resolved["degree"]) < 1 or int(resolved["degree"]) % 2 == 0:
        raise ValueError("demo.degree must be a positive odd integer")
    if int(resolved["grid_size"]) <= int(resolved["degree"]) + 1:
        raise ValueError("demo.grid_size must be greater than degree + 1")
    if int(resolved["max_nfev"]) <= 0:
        raise ValueError("demo.max_nfev must be positive")
    if str(resolved["phase_synthesis_method"]) not in {
        "auto",
        "pennylane_poly_to_angles",
        "scipy_least_squares_scalar_qsp",
        "none",
    }:
        raise ValueError("demo.phase_synthesis_method is invalid")
    if str(resolved["angle_solver"]) not in {"root-finding", "iterative", "iterative-optax"}:
        raise ValueError("demo.angle_solver is invalid")
    if int(resolved["phase_validation_grid_size"]) <= 1:
        raise ValueError("demo.phase_validation_grid_size must be greater than 1")
    if (
        bool(resolved["require_phase_validation"])
        and str(resolved["phase_synthesis_method"]) == "none"
    ):
        raise ValueError("demo.require_phase_validation does not allow phase_synthesis_method=none")
    if float(resolved["block_encoding_normalization"]) <= 0.0:
        raise ValueError("demo.block_encoding_normalization must be positive")
    if not 0.0 <= float(resolved["domain_min"]) < float(resolved["domain_max"]) <= 1.0:
        raise ValueError("demo domain must satisfy 0 <= domain_min < domain_max <= 1")
    singular_values = np.asarray(resolved["singular_values"], dtype=np.float64)
    if singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("demo.singular_values must be a non-empty numeric list")
    if np.any(singular_values < 0.0) or np.any(
        singular_values > float(resolved["block_encoding_normalization"])
    ):
        raise ValueError("demo.singular_values must lie in the block-encoding interval")
    resolved["singular_values"] = singular_values.tolist()
    return {"demo": resolved}


def run_phase_demo(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_phase_demo_config(config)["demo"]
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting QSP/QSVT phase demo %s", resolved["run_id"])

    alpha = float(resolved["alpha"])
    degree = int(resolved["degree"])
    domain_min = float(resolved["domain_min"])
    domain_max = float(resolved["domain_max"])
    block_encoding_normalization = float(resolved["block_encoding_normalization"])
    grid_size = int(resolved["grid_size"])
    seed = int(resolved["seed"])

    approximation = fit_odd_regularized_polynomial(
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
        degree=degree,
        domain_min=domain_min,
        domain_max=domain_max,
        grid_size=grid_size,
    )
    phase_angles, phase_metadata = _phase_angles_for_polynomial(
        approximation,
        method=str(resolved["phase_synthesis_method"]),
        seed=seed,
        grid_size=grid_size,
        max_nfev=int(resolved["max_nfev"]),
        angle_solver=str(resolved["angle_solver"]),
        cache_dir=str(resolved["phase_cache_dir"]),
    )

    polynomial = approximation.polynomial
    scaled_polynomial = polynomial / approximation.scale_factor
    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    scaled_target = target / approximation.scale_factor
    polynomial_values = polynomial(grid)
    scaled_polynomial_values = scaled_polynomial(grid)
    qsp_values = _maybe_scalar_qsp_values(
        grid,
        phase_angles,
        phase_metadata,
        approximation.scale_factor,
    )
    approximation_error = pd.DataFrame(
        {
            "normalized_singular_value": grid,
            "target": target,
            "scaled_target": scaled_target,
            "polynomial_approximation": polynomial_values,
            "scaled_polynomial_approximation": scaled_polynomial_values,
            "qsp_response_approximation": qsp_values,
            "abs_error": np.abs(polynomial_values - target),
            "scaled_abs_error": np.abs(scaled_polynomial_values - scaled_target),
            "qsp_abs_error": np.abs(qsp_values - target),
        }
    )
    singular_values = np.asarray(resolved["singular_values"], dtype=np.float64)
    normalized_singular_values = singular_values / block_encoding_normalization
    exact_filter = singular_values / (singular_values**2 + alpha)
    polynomial_filter = polynomial(normalized_singular_values)
    scaled_filter = scaled_polynomial(normalized_singular_values)
    demo_results = pd.DataFrame(
        {
            "singular_value": singular_values,
            "normalized_singular_value": normalized_singular_values,
            "exact_filter": exact_filter,
            "polynomial_filter": polynomial_filter,
            "scaled_polynomial_filter": scaled_filter,
            "polynomial_abs_error": np.abs(polynomial_filter - exact_filter),
        }
    )
    phase_validation = _phase_validation_frame(
        approximation,
        phase_angles,
        phase_metadata,
        grid_size=int(resolved["phase_validation_grid_size"]),
    )
    phase_circuit_summary = (
        qsp_circuit_summary(phase_angles)
        if phase_metadata.get("phase_synthesis_method") != "none"
        else {
            "qiskit_available": None,
            "qiskit_error": "phase synthesis not requested",
            "circuit_depth": None,
            "gate_counts": {},
            "gate_count_total": 0,
        }
    )
    circuit_summary = {
        **phase_circuit_summary,
        **phase_metadata,
        "run_id": resolved["run_id"],
        "degree": degree,
        "n_phase_angles": int(phase_angles.size),
        "target_scale": approximation.scale_factor,
        "scale_factor": approximation.scale_factor,
        "max_abs_error": approximation.max_error,
        "mean_abs_error": approximation.mean_error,
        "scaled_max_abs_error": approximation.scaled_max_error,
        "scaled_mean_abs_error": approximation.scaled_mean_error,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "alpha": alpha,
        "parity": "odd",
        "scope_note": (
            "Small diagonal/block-encoding proof of concept only; large IEEE benchmarks "
            "remain classical spectral simulations plus resource proxies."
        ),
    }
    if phase_validation is not None:
        validation_summary, validation_frame = phase_validation
        circuit_summary.update(validation_summary)
    else:
        validation_frame = pd.DataFrame()
    validation_report = _phase_validation_report(
        resolved=resolved,
        approximation=approximation,
        approximation_error=approximation_error,
        phase_angles=phase_angles,
        phase_metadata=phase_metadata,
        phase_validation=validation_frame,
    )
    circuit_summary.update(
        {
            "validation_passed": validation_report["validation_passed"],
            "dummy_phase_check_passed": validation_report["dummy_phase_check_passed"],
            "boundedness_check_passed": validation_report["boundedness_check_passed"],
            "parity_check_passed": validation_report["parity_check_passed"],
        }
    )

    artifacts = _write_phase_demo_artifacts(
        output_dir=output_dir,
        resolved_config={"demo": resolved},
        phase_angles=phase_angles,
        approximation=approximation,
        approximation_error=approximation_error,
        phase_validation=validation_frame,
        demo_results=demo_results,
        circuit_summary=circuit_summary,
        phase_metadata=phase_metadata,
        validation_report=validation_report,
    )
    logger.info("Completed QSP/QSVT phase demo %s", resolved["run_id"])
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "phase_angles": phase_angles,
        "circuit_summary": circuit_summary,
    }


def _phase_angles_for_polynomial(
    approximation: OddPolynomialApproximation,
    *,
    method: str,
    seed: int,
    grid_size: int,
    max_nfev: int,
    angle_solver: str,
    cache_dir: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled_coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
    scaled_coefficients = scaled_coefficients / approximation.scale_factor
    if method == "auto" and approximation.degree > 15:
        method = "pennylane_poly_to_angles"
    if method in {"auto", "pennylane_poly_to_angles"}:
        try:
            cached = synthesize_pennylane_phases_cached(
                scaled_coefficients,
                angle_solver=angle_solver,
                cache_dir=cache_dir,
                cache_metadata={
                    "alpha": approximation.alpha,
                    "degree": approximation.degree,
                    "domain_min": approximation.domain_min,
                    "domain_max": approximation.domain_max,
                    "scale_factor": approximation.scale_factor,
                    "parity": "odd",
                },
            )
            metadata = dict(cached.metadata)
            metadata["phase_cache_hit"] = cached.cache_hit
            return cached.phases, metadata
        except Exception as exc:
            if method == "pennylane_poly_to_angles":
                raise
            fallback_reason = f"PennyLane poly_to_angles unavailable: {exc}"
    else:
        fallback_reason = "custom scalar QSP optimizer requested"

    if method in {"auto", "scipy_least_squares_scalar_qsp"}:
        phase_result = synthesize_qsp_phases(
            alpha=approximation.alpha,
            block_encoding_normalization=approximation.block_encoding_normalization,
            degree=approximation.degree,
            domain_min=approximation.domain_min,
            domain_max=approximation.domain_max,
            grid_size=grid_size,
            seed=seed,
            max_nfev=max_nfev,
        )
        return phase_result.phases, {
            "phase_synthesis_method": "scipy_least_squares_scalar_qsp",
            "phase_synthesis_success": phase_result.optimization_success,
            "phase_synthesis_message": phase_result.optimization_message,
            "phase_synthesis_fallback_reason": fallback_reason,
            "scalar_qsp_max_abs_error": phase_result.max_abs_error,
            "scalar_qsp_mean_abs_error": phase_result.mean_abs_error,
        }

    phases = np.zeros(approximation.degree + 1, dtype=np.float64)
    return phases, {
        "phase_synthesis_method": "none",
        "phase_synthesis_success": False,
        "phase_synthesis_message": "No phase synthesis backend was requested",
    }


def _phase_validation_frame(
    approximation: OddPolynomialApproximation,
    phase_angles: np.ndarray,
    phase_metadata: dict[str, Any],
    *,
    grid_size: int,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    if phase_metadata.get("phase_synthesis_method") != "pennylane_poly_to_angles":
        return None
    validation = validate_pennylane_qsvt_phases(
        power_coefficients=np.asarray(approximation.power_coefficients, dtype=np.float64),
        scale_factor=approximation.scale_factor,
        phases=phase_angles,
        alpha=approximation.alpha,
        block_encoding_normalization=approximation.block_encoding_normalization,
        domain_min=approximation.domain_min,
        domain_max=approximation.domain_max,
        grid_size=grid_size,
    )
    return (
        {
            "phase_validation_backend": validation.backend,
            "phase_implemented_max_abs_error": validation.max_abs_error,
            "phase_implemented_mean_abs_error": validation.mean_abs_error,
            "phase_implemented_scaled_max_abs_error": validation.max_scaled_abs_error,
            "phase_implemented_scaled_mean_abs_error": validation.mean_scaled_abs_error,
        },
        validation.frame,
    )


def _maybe_scalar_qsp_values(
    grid: np.ndarray,
    phase_angles: np.ndarray,
    phase_metadata: dict[str, Any],
    scale_factor: float,
) -> np.ndarray:
    if phase_metadata.get("phase_synthesis_method") != "scipy_least_squares_scalar_qsp":
        return np.full_like(grid, np.nan, dtype=np.float64)
    return qsp_response(grid, phase_angles) * scale_factor


def _phase_validation_report(
    *,
    resolved: dict[str, Any],
    approximation: OddPolynomialApproximation,
    approximation_error: pd.DataFrame,
    phase_angles: np.ndarray,
    phase_metadata: dict[str, Any],
    phase_validation: pd.DataFrame,
) -> dict[str, Any]:
    coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
    scaled_coefficients = coefficients / approximation.scale_factor
    unit_grid = np.linspace(-1.0, 1.0, max(int(resolved["grid_size"]), 2048), dtype=np.float64)
    validation_grid = np.linspace(
        approximation.domain_min,
        approximation.domain_max,
        max(int(resolved["phase_validation_grid_size"]), 2),
        dtype=np.float64,
    )
    scaled_target = (
        regularized_filter_on_normalized_domain(
            validation_grid,
            alpha=approximation.alpha,
            block_encoding_normalization=approximation.block_encoding_normalization,
        )
        / approximation.scale_factor
    )
    scaled_polynomial = (approximation.polynomial / approximation.scale_factor)(unit_grid)
    parity_check_passed = bool(np.all(np.abs(scaled_coefficients[::2]) <= 1.0e-10))
    boundedness_check_passed = bool(
        np.max(np.abs(scaled_target)) <= 1.0 + 1.0e-10
        and np.max(np.abs(scaled_polynomial)) <= 1.0 + 1.0e-6
    )
    dummy_phase_check_passed = bool(
        phase_angles.size == approximation.degree + 1
        and np.all(np.isfinite(phase_angles))
        and not np.allclose(phase_angles, 0.0)
        and phase_metadata.get("phase_synthesis_method") != "none"
    )
    max_phase_error = (
        float(phase_validation["phase_abs_error"].max())
        if not phase_validation.empty and "phase_abs_error" in phase_validation
        else float("inf")
    )
    mean_phase_error = (
        float(phase_validation["phase_abs_error"].mean())
        if not phase_validation.empty and "phase_abs_error" in phase_validation
        else float("inf")
    )
    max_polynomial_error = float(approximation_error["abs_error"].max())
    mean_polynomial_error = float(approximation_error["abs_error"].mean())
    validation_passed = bool(
        parity_check_passed
        and boundedness_check_passed
        and dummy_phase_check_passed
        and max_polynomial_error <= float(resolved["polynomial_max_error_target"])
        and mean_polynomial_error <= float(resolved["polynomial_mean_error_target"])
        and max_phase_error <= float(resolved["phase_max_error_target"])
        and mean_phase_error <= float(resolved["phase_mean_error_target"])
    )
    return {
        "validation_passed": validation_passed,
        "target_function": "P_alpha(sigma) = sigma / (sigma^2 + alpha)",
        "alpha": approximation.alpha,
        "domain": [approximation.domain_min, approximation.domain_max],
        "scale_factor": approximation.scale_factor,
        "polynomial_degree": approximation.degree,
        "phase_synthesis_method": phase_metadata.get("phase_synthesis_method"),
        "phase_solver": phase_metadata.get("phase_synthesis_angle_solver"),
        "phase_count": int(phase_angles.size),
        "max_polynomial_error": max_polynomial_error,
        "mean_polynomial_error": mean_polynomial_error,
        "max_phase_implemented_error": max_phase_error,
        "mean_phase_implemented_error": mean_phase_error,
        "parity_check_passed": parity_check_passed,
        "boundedness_check_passed": boundedness_check_passed,
        "dummy_phase_check_passed": dummy_phase_check_passed,
        "target_magnitude_bounded": float(np.max(np.abs(scaled_target))),
        "polynomial_magnitude_bounded": float(np.max(np.abs(scaled_polynomial))),
        "polynomial_max_error_target": float(resolved["polynomial_max_error_target"]),
        "polynomial_mean_error_target": float(resolved["polynomial_mean_error_target"]),
        "phase_max_error_target": float(resolved["phase_max_error_target"]),
        "phase_mean_error_target": float(resolved["phase_mean_error_target"]),
    }


def _write_phase_demo_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    phase_angles: np.ndarray,
    approximation: OddPolynomialApproximation,
    approximation_error: pd.DataFrame,
    phase_validation: pd.DataFrame,
    demo_results: pd.DataFrame,
    circuit_summary: dict[str, Any],
    phase_metadata: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, str]:
    config_path = output_dir / "config_resolved.yaml"
    qsvt_config_path = output_dir / "qsvt_demo_config_resolved.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=True)
    with qsvt_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=True)

    phase_path = output_dir / "phase_angles.csv"
    coefficient_path = output_dir / "polynomial_coefficients.csv"
    error_path = output_dir / "approximation_error.csv"
    phase_metadata_path = output_dir / "phase_synthesis_metadata.json"
    validation_path = output_dir / "qsp_validation_grid.csv"
    phase_error_path = output_dir / "phase_implemented_error.csv"
    summary_path = output_dir / "circuit_summary.json"
    results_path = output_dir / "qsvt_demo_results.csv"
    plot_path = output_dir / "qsvt_demo_plot.png"
    phase_synthesis_plot_path = output_dir / "qsvt_phase_synthesis_plot.png"
    validation_report_path = output_dir / "phase_validation_report.json"
    validation_plot_path = output_dir / "phase_validation_plot.png"

    pd.DataFrame(
        {
            "phase_index": np.arange(len(phase_angles)),
            "phase_angle": phase_angles,
        }
    ).to_csv(phase_path, index=False)
    coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
    pd.DataFrame(
        {
            "coefficient_index": np.arange(len(coefficients)),
            "power_coefficient": coefficients,
            "scaled_power_coefficient": coefficients / approximation.scale_factor,
        }
    ).to_csv(coefficient_path, index=False)
    approximation_error.to_csv(error_path, index=False)
    phase_validation.to_csv(validation_path, index=False)
    phase_validation.to_csv(phase_error_path, index=False)
    demo_results.to_csv(results_path, index=False)
    write_json(phase_metadata_path, phase_metadata)
    write_json(summary_path, circuit_summary)
    write_json(validation_report_path, validation_report)
    _plot_phase_demo(approximation_error, demo_results, plot_path)
    _plot_phase_synthesis(approximation_error, phase_validation, phase_synthesis_plot_path)
    _plot_phase_synthesis(approximation_error, phase_validation, validation_plot_path)
    return {
        "config_resolved": str(config_path),
        "qsvt_demo_config_resolved": str(qsvt_config_path),
        "phase_angles": str(phase_path),
        "polynomial_coefficients": str(coefficient_path),
        "approximation_error": str(error_path),
        "phase_synthesis_metadata": str(phase_metadata_path),
        "qsp_validation_grid": str(validation_path),
        "phase_implemented_error": str(phase_error_path),
        "circuit_summary": str(summary_path),
        "qsvt_demo_results": str(results_path),
        "qsvt_demo_plot": str(plot_path),
        "qsvt_phase_synthesis_plot": str(phase_synthesis_plot_path),
        "phase_validation_report": str(validation_report_path),
        "phase_validation_plot": str(validation_plot_path),
        "run_log": str(output_dir / "run.log"),
    }


def _plot_phase_demo(
    approximation_error: pd.DataFrame,
    demo_results: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["target"],
        label="Exact regularized filter",
    )
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_target"],
        label="Scaled target",
    )
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["polynomial_approximation"],
        linestyle="--",
        label="Odd polynomial approximation",
    )
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_polynomial_approximation"],
        linestyle="--",
        label="Scaled odd polynomial",
    )
    axes[0].scatter(
        demo_results["normalized_singular_value"],
        demo_results["exact_filter"],
        color="black",
        s=28,
        label="Demo singular values",
    )
    axes[0].set_xlabel("Normalized singular value")
    axes[0].set_ylabel("Filter value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize="x-small")

    axes[1].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["abs_error"],
        label="Polynomial absolute error",
    )
    axes[1].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_abs_error"],
        linestyle="--",
        label="Scaled absolute error",
    )
    axes[1].set_xlabel("Normalized singular value")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_phase_synthesis(
    approximation_error: pd.DataFrame,
    phase_validation: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_target"],
        label="Scaled target",
    )
    axes[0].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_polynomial_approximation"],
        linestyle="--",
        label="Scaled polynomial",
    )
    if not phase_validation.empty:
        axes[0].plot(
            phase_validation["normalized_singular_value"],
            phase_validation["scaled_phase_response"],
            linestyle=":",
            label="Phase-implemented response",
        )
    axes[0].set_xlabel("Normalized singular value")
    axes[0].set_ylabel("Scaled filter value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize="x-small")

    axes[1].plot(
        approximation_error["normalized_singular_value"],
        approximation_error["scaled_abs_error"],
        label="Scaled polynomial error",
    )
    if not phase_validation.empty:
        axes[1].plot(
            phase_validation["normalized_singular_value"],
            phase_validation["phase_scaled_abs_error"],
            linestyle=":",
            label="Phase response error",
        )
    axes[1].set_xlabel("Normalized singular value")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small QSP/QSVT phase demo")
    parser.add_argument("--config", required=True, help="Path to a YAML QSVT phase demo config")
    args = parser.parse_args(argv)
    config = load_phase_demo_config(args.config)
    run_phase_demo(config)


if __name__ == "__main__":
    main()
