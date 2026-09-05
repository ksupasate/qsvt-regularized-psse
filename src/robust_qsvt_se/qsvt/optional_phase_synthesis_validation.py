from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.polynomial_approximation import (
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

PHASE_SYNTHESIS_CAVEAT = (
    "Actual phase synthesis is attempted only when the optional dependency is "
    "available. A synthesized phase sequence is not hardware validation and does "
    "not demonstrate quantum speedup, quantum advantage, or QSVT superiority over "
    "Ridge/Tikhonov under the same alpha."
)


def run_optional_phase_synthesis_validation(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    dependency_available = importlib.util.find_spec("pennylane") is not None
    for alpha in resolved["alpha"]:
        if bool(resolved["force_dependency_missing"]) or not dependency_available:
            summary_rows.append(
                _skip_row(
                    context,
                    alpha=float(alpha),
                    dependency_available=dependency_available,
                    reason="PennyLane dependency unavailable or forced missing by config",
                    resolved=resolved,
                )
            )
            continue
        try:
            summary, phases, errors = _run_phase_synthesis_row(
                context=context,
                alpha=float(alpha),
                resolved=resolved,
                output_dir=output_dir,
            )
            summary_rows.append(summary)
            phase_rows.extend(phases)
            error_rows.extend(errors)
        except Exception as exc:
            summary_rows.append(
                _skip_row(
                    context,
                    alpha=float(alpha),
                    dependency_available=dependency_available,
                    reason=str(exc),
                    resolved=resolved,
                    status="failed_validation",
                )
            )

    summary_frame = pd.DataFrame(summary_rows)
    phase_frame = pd.DataFrame(
        phase_rows,
        columns=["alpha", "phase_index", "phase_angle"],
    )
    error_frame = pd.DataFrame(error_rows)
    summary_csv = output_dir / "phase_synthesis_summary.csv"
    summary_json = output_dir / "phase_synthesis_summary.json"
    phase_csv = output_dir / "phase_angles.csv"
    error_csv = output_dir / "phase_pointwise_errors.csv"
    summary_frame.to_csv(summary_csv, index=False)
    phase_frame.to_csv(phase_csv, index=False)
    error_frame.to_csv(error_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "phase_synthesis_summary_csv": str(summary_csv),
            "phase_synthesis_summary_json": str(summary_json),
            "phase_angles_csv": str(phase_csv),
            "phase_pointwise_errors_csv": str(error_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "phase_synthesis_summary_csv": summary_csv,
            "phase_synthesis_summary_json": summary_json,
            "phase_angles_csv": phase_csv,
            "phase_pointwise_errors_csv": error_csv,
            "manifest": manifest_path,
        },
    }


def _run_phase_synthesis_row(
    *,
    context: Any,
    alpha: float,
    resolved: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    polynomial = evaluate_polynomial_approximation(
        context=context,
        alpha=alpha,
        degree=int(resolved["degree"]),
        method=str(resolved["polynomial_method"]),
        grid_size=int(resolved["grid_size"]),
    )
    validation = validate_qsvt_polynomial(
        polynomial.power_coefficients,
        parity="odd",
        grid_size=1001,
        bound_tolerance=float(resolved["bound_tolerance"]),
    )
    phase_result = synthesize_pennylane_phases_cached(
        polynomial.power_coefficients,
        angle_solver=str(resolved["angle_solver"]),
        cache_dir=output_dir / "phase_cache",
        cache_metadata={
            "alpha": float(alpha),
            "degree": int(polynomial.degree),
            "method": polynomial.method,
        },
    )
    grid = np.linspace(context.domain_min, context.domain_max, int(resolved["grid_size"]))
    target = evaluate_polynomial_approximation(
        context=context,
        alpha=alpha,
        degree=int(resolved["degree"]),
        method=str(resolved["polynomial_method"]),
        grid_size=int(resolved["grid_size"]),
    )
    mask = np.asarray(target.evaluation_kind) == "grid"
    response = pennylane_qsvt_response(
        grid,
        phase_result.phases,
        phase_order=str(resolved["phase_order"]),
        phase_sign=str(resolved["phase_sign"]),
        phase_offset_rule=str(resolved["phase_offset_rule"]),
        signal_operator_convention=str(resolved["signal_operator_convention"]),
        response_component=str(resolved["response_component"]),
    )
    phase_errors = np.abs(response - target.bounded_target_values[mask])
    max_error = float(np.max(phase_errors))
    status = "passed" if max_error <= float(resolved["target_tolerance"]) else "failed_validation"
    summary = {
        "case_name": context.case_name,
        "matrix_source": context.matrix_source,
        "alpha": float(alpha),
        "target_tolerance": float(resolved["target_tolerance"]),
        "phase_method": "pennylane_poly_to_angles",
        "dependency_used": phase_result.metadata.get("phase_synthesis_backend", "pennylane"),
        "dependency_available": True,
        "degree": int(polynomial.degree),
        "phase_count": len(phase_result.phases),
        "phase_order": str(resolved["phase_order"]),
        "phase_sign": str(resolved["phase_sign"]),
        "phase_offset_rule": str(resolved["phase_offset_rule"]),
        "signal_operator_convention": str(resolved["signal_operator_convention"]),
        "response_component": str(resolved["response_component"]),
        "convention": (
            f"{resolved['phase_order']}/{resolved['phase_sign']}/"
            f"{resolved['phase_offset_rule']}/{resolved['signal_operator_convention']}/"
            f"{resolved['response_component']}"
        ),
        "query_count_estimate": int(2 * polynomial.degree + 1),
        "bounded_scaling_C": polynomial.bounded_scaling_C,
        "max_pointwise_error": max_error,
        "mean_pointwise_error": float(np.mean(phase_errors)),
        "rms_pointwise_error": float(np.sqrt(np.mean(phase_errors**2))),
        "status": status,
        "skip_reason": "",
        "polynomial_validation_max_abs": validation["max_abs_on_unit_interval"],
        "cache_hit": phase_result.cache_hit,
        "caveat": PHASE_SYNTHESIS_CAVEAT,
    }
    phases = [
        {"alpha": float(alpha), "phase_index": index, "phase_angle": float(phase)}
        for index, phase in enumerate(phase_result.phases)
    ]
    errors = [
        {
            "case_name": context.case_name,
            "matrix_source": context.matrix_source,
            "alpha": float(alpha),
            "degree": int(polynomial.degree),
            "sigma_normalized": float(sigma),
            "target_bounded_value": float(target_value),
            "phase_response_value": float(response_value),
            "pointwise_error": float(error),
            "phase_order": str(resolved["phase_order"]),
            "phase_sign": str(resolved["phase_sign"]),
            "phase_offset_rule": str(resolved["phase_offset_rule"]),
            "signal_operator_convention": str(resolved["signal_operator_convention"]),
            "response_component": str(resolved["response_component"]),
        }
        for sigma, target_value, response_value, error in zip(
            grid,
            target.bounded_target_values[mask],
            response,
            phase_errors,
            strict=True,
        )
    ]
    return summary, phases, errors


def _skip_row(
    context: Any,
    *,
    alpha: float,
    dependency_available: bool,
    reason: str,
    resolved: dict[str, Any],
    status: str = "skipped_dependency_missing",
) -> dict[str, Any]:
    return {
        "case_name": context.case_name,
        "matrix_source": context.matrix_source,
        "alpha": float(alpha),
        "target_tolerance": float(resolved["target_tolerance"]),
        "phase_method": "pennylane_poly_to_angles",
        "dependency_used": "none",
        "dependency_available": bool(dependency_available),
        "degree": int(resolved["degree"]),
        "phase_count": 0,
        "phase_order": str(resolved.get("phase_order", "")),
        "phase_sign": str(resolved.get("phase_sign", "")),
        "phase_offset_rule": str(resolved.get("phase_offset_rule", "")),
        "signal_operator_convention": str(resolved.get("signal_operator_convention", "")),
        "response_component": str(resolved.get("response_component", "")),
        "convention": "",
        "query_count_estimate": int(2 * int(resolved["degree"]) + 1),
        "bounded_scaling_C": 1.0,
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "status": status,
        "skip_reason": reason,
        "caveat": PHASE_SYNTHESIS_CAVEAT,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_optional_phase_synthesis_validation",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": [1.0e-2],
        "degree": 35,
        "target_tolerance": 1.0e-3,
        "grid_size": 256,
        "polynomial_method": "odd_chebyshev_minimax_lp",
        "angle_solver": "root-finding",
        "bound_tolerance": 1.0e-5,
        "phase_order": "original",
        "phase_sign": "phi",
        "phase_offset_rule": "none",
        "signal_operator_convention": "pennylane_rx_pcphase",
        "response_component": "real_u00",
        "force_dependency_missing": False,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run optional QSP/QSVT phase synthesis validation")
    parser.parse_args(argv)
    run = run_optional_phase_synthesis_validation()
    print(f"QSVT optional phase synthesis validation complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
