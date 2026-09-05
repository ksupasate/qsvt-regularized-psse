from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.qsvt.polynomial_approximation import (
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

PHASE_RESPONSE_CAVEAT = (
    "Phase-response convention diagnostic only. Passing a scalar phase-response "
    "check is not hardware execution, quantum speedup, quantum advantage, or "
    "evidence that QSVT outperforms Ridge/Tikhonov under the same alpha."
)
NO_PASS_MESSAGE = (
    "No tested phase-response convention passed validation. The polynomial "
    "approximation remains valid as a bounded diagnostic, but full phase-level "
    "QSP/QSVT validation remains unresolved."
)
PASS_MESSAGE = "Phase-response validation passed under the documented convention."


def diagnose_phase_response_conventions(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    dependency_available = importlib.util.find_spec("pennylane") is not None
    if not dependency_available or bool(resolved["force_dependency_missing"]):
        summary_rows.append(_dependency_skip_row(resolved, dependency_available))
    else:
        sanity_polynomials = _sanity_polynomials()
        ridge = _ridge_polynomial_spec(resolved)
        for spec in [*sanity_polynomials, ridge]:
            rows, values = _search_conventions_for_polynomial(spec, resolved)
            summary_rows.extend(rows)
            value_rows.extend(values)
            if spec["target_type"] == "sanity_polynomial":
                best = _best_row(rows)
                sanity_rows.append(
                    {
                        "polynomial_name": spec["polynomial_name"],
                        "degree": spec["degree"],
                        "best_max_pointwise_error": best["max_pointwise_error"],
                        "best_status": best["status"],
                        "best_convention": _convention_label(best),
                        "sanity_tolerance": float(resolved["sanity_tolerance"]),
                    }
                )

    summary_frame = pd.DataFrame(summary_rows)
    value_frame = pd.DataFrame(value_rows)
    sanity_frame = pd.DataFrame(sanity_rows)
    summary_csv = output_dir / "convention_search_summary.csv"
    summary_json = output_dir / "convention_search_summary.json"
    sanity_csv = output_dir / "sanity_polynomial_results.csv"
    values_csv = output_dir / "phase_response_values.csv"
    report_md = output_dir / "best_convention_report.md"
    summary_frame.to_csv(summary_csv, index=False)
    sanity_frame.to_csv(sanity_csv, index=False)
    value_frame.to_csv(values_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_best_convention_report(summary_frame, sanity_frame), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "convention_search_summary_csv": str(summary_csv),
            "convention_search_summary_json": str(summary_json),
            "sanity_polynomial_results_csv": str(sanity_csv),
            "phase_response_values_csv": str(values_csv),
            "best_convention_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "convention_search_summary_csv": summary_csv,
            "convention_search_summary_json": summary_json,
            "sanity_polynomial_results_csv": sanity_csv,
            "phase_response_values_csv": values_csv,
            "best_convention_report_md": report_md,
            "manifest": manifest_path,
        },
    }


def pennylane_qsvt_response(
    normalized_sigma: np.ndarray,
    phases: np.ndarray,
    *,
    phase_order: str = "original",
    phase_sign: str = "phi",
    phase_offset_rule: str = "none",
    signal_operator_convention: str = "pennylane_rx_pcphase",
    response_component: str = "real_u00",
) -> np.ndarray:
    values = np.asarray(normalized_sigma, dtype=np.float64)
    phase_values = _transform_phases(
        np.asarray(phases, dtype=np.float64),
        phase_order=phase_order,
        phase_sign=phase_sign,
        phase_offset_rule=phase_offset_rule,
    )
    responses = np.empty(values.shape, dtype=np.float64)
    for index, value in np.ndenumerate(values):
        unitary = _qsvt_unitary(float(value), phase_values, signal_operator_convention)
        responses[index] = _response_component(unitary[0, 0], response_component)
    return responses


def _search_conventions_for_polynomial(
    spec: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        return [_synthesis_failure_row(spec, "pennylane import failed", str(exc), config)], []

    coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
    try:
        validation = validate_qsvt_polynomial(
            coefficients,
            parity="odd",
            grid_size=int(config["bound_validation_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phases = np.asarray(
            qml.poly_to_angles(
                coefficients,
                "QSVT",
                angle_solver=str(config["angle_solver"]),
            ),
            dtype=np.float64,
        )
    except Exception as exc:
        return [
            _synthesis_failure_row(
                spec,
                "phase synthesis or polynomial validation failed",
                str(exc),
                config,
            )
        ], []

    grid = np.asarray(spec["grid"], dtype=np.float64)
    target = np.asarray(spec["target"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for phase_order in config["phase_order"]:
        for phase_sign in config["phase_sign"]:
            for phase_offset_rule in config["phase_offset_rule"]:
                for signal_operator in config["signal_operator_convention"]:
                    for response_component in config["response_component"]:
                        try:
                            response = pennylane_qsvt_response(
                                grid,
                                phases,
                                phase_order=str(phase_order),
                                phase_sign=str(phase_sign),
                                phase_offset_rule=str(phase_offset_rule),
                                signal_operator_convention=str(signal_operator),
                                response_component=str(response_component),
                            )
                            errors = np.abs(response - target)
                            max_error = float(np.max(errors))
                            status = (
                                "passed"
                                if max_error <= float(spec["tolerance"])
                                else "failed_validation"
                            )
                            failure_reason = ""
                        except Exception as exc:
                            response = np.full_like(grid, np.nan, dtype=np.float64)
                            errors = np.full_like(grid, np.nan, dtype=np.float64)
                            max_error = np.nan
                            status = "failed_convention_evaluation"
                            failure_reason = str(exc)
                        row = {
                            "polynomial_name": spec["polynomial_name"],
                            "target_type": spec["target_type"],
                            "alpha_if_applicable": spec.get("alpha"),
                            "degree": int(spec["degree"]),
                            "phase_method": "pennylane_poly_to_angles",
                            "dependency_used": f"pennylane-{qml.__version__}",
                            "phase_count": int(phases.size),
                            "phase_order": phase_order,
                            "phase_sign": phase_sign,
                            "phase_offset_rule": phase_offset_rule,
                            "signal_operator_convention": signal_operator,
                            "response_component": response_component,
                            "coefficient_basis_input": "monomial_power_low_to_high",
                            "coefficient_basis_expected": "monomial_power_low_to_high",
                            "max_pointwise_error": max_error,
                            "mean_pointwise_error": float(np.nanmean(errors)),
                            "rms_pointwise_error": float(np.sqrt(np.nanmean(errors**2))),
                            "passed_1e_minus_2": bool(max_error <= 1.0e-2),
                            "passed_5e_minus_3": bool(max_error <= 5.0e-3),
                            "passed_1e_minus_3": bool(max_error <= 1.0e-3),
                            "status": status,
                            "failure_reason": failure_reason,
                            "polynomial_bound_max_abs": validation["max_abs_on_unit_interval"],
                            "caveat": PHASE_RESPONSE_CAVEAT,
                        }
                        rows.append(row)
                        if str(config["write_all_response_values"]).lower() == "true":
                            value_rows.extend(
                                _value_rows(
                                    spec=spec,
                                    row=row,
                                    grid=grid,
                                    target=target,
                                    response=response,
                                    errors=errors,
                                )
                            )
    return rows, value_rows


def _qsvt_unitary(
    x: float,
    phases: np.ndarray,
    signal_operator_convention: str,
) -> np.ndarray:
    if not -1.0 <= x <= 1.0:
        raise ValueError("normalized signal must lie in [-1, 1]")
    signal = _signal_operator(x, signal_operator_convention)
    if signal_operator_convention.endswith("_existing_order"):
        unitary = _phase_rotation(phases[0])
        for phase in phases[1:]:
            unitary = unitary @ signal @ _phase_rotation(float(phase))
        return unitary
    # PennyLane QSVT decomposes the scalar block encoding as
    # P0, (U^\dagger P1 U), P2, ..., U, Plast in operation order.
    # Matrix multiplication therefore applies each later operation on the left.
    unitary = _phase_rotation(float(phases[0]))
    signal_adjoint = signal.conj().T
    for phase_index in range(1, len(phases) - 1, 2):
        unitary = signal_adjoint @ _phase_rotation(float(phases[phase_index])) @ signal @ unitary
        unitary = _phase_rotation(float(phases[phase_index + 1])) @ unitary
    if len(phases) % 2 == 0:
        unitary = signal @ unitary
        unitary = _phase_rotation(float(phases[-1])) @ unitary
    return unitary


def _signal_operator(x: float, convention: str) -> np.ndarray:
    off = np.sqrt(max(0.0, 1.0 - x**2))
    if convention in {"pennylane_rx_pcphase", "pennylane_rx_existing_order"}:
        return np.array([[x, -1j * off], [-1j * off, x]], dtype=np.complex128)
    if convention in {"repository_plus_i", "repository_plus_i_existing_order"}:
        return np.array([[x, 1j * off], [1j * off, x]], dtype=np.complex128)
    if convention == "real_rotation":
        return np.array([[x, -off], [off, x]], dtype=np.complex128)
    raise ValueError(f"unknown signal operator convention: {convention}")


def _phase_rotation(phase: float) -> np.ndarray:
    return np.array(
        [[np.exp(1j * phase), 0.0], [0.0, np.exp(-1j * phase)]],
        dtype=np.complex128,
    )


def _transform_phases(
    phases: np.ndarray,
    *,
    phase_order: str,
    phase_sign: str,
    phase_offset_rule: str,
) -> np.ndarray:
    values = np.asarray(phases, dtype=np.float64).copy()
    if phase_order == "reversed":
        values = values[::-1]
    elif phase_order != "original":
        raise ValueError(f"unknown phase_order: {phase_order}")
    if phase_sign == "minus_phi":
        values = -values
    elif phase_sign != "phi":
        raise ValueError(f"unknown phase_sign: {phase_sign}")
    if phase_offset_rule == "none":
        return values
    if phase_offset_rule == "first_plus_pi_over_2":
        values[0] += np.pi / 2.0
    elif phase_offset_rule == "first_minus_pi_over_2":
        values[0] -= np.pi / 2.0
    elif phase_offset_rule == "last_plus_pi_over_2":
        values[-1] += np.pi / 2.0
    elif phase_offset_rule == "last_minus_pi_over_2":
        values[-1] -= np.pi / 2.0
    else:
        raise ValueError(f"unknown phase_offset_rule: {phase_offset_rule}")
    return values


def _response_component(value: complex, component: str) -> float:
    if component == "real_u00":
        return float(np.real(value))
    if component == "imag_u00":
        return float(np.imag(value))
    if component == "negative_real_u00":
        return float(-np.real(value))
    if component == "negative_imag_u00":
        return float(-np.imag(value))
    if component == "abs_u00":
        return float(np.abs(value))
    raise ValueError(f"unknown response_component: {component}")


def _sanity_polynomials() -> list[dict[str, Any]]:
    grid = np.linspace(-0.95, 0.95, 101, dtype=np.float64)
    specs = [
        ("x", np.array([0.0, 1.0], dtype=np.float64)),
        ("0.5x", np.array([0.0, 0.5], dtype=np.float64)),
        ("x^3", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("0.5x_plus_0.25x^3", np.array([0.0, 0.5, 0.0, 0.25], dtype=np.float64)),
    ]
    return [
        {
            "polynomial_name": name,
            "target_type": "sanity_polynomial",
            "coefficients": coefficients,
            "degree": int(coefficients.size - 1),
            "grid": grid,
            "target": np.polynomial.Polynomial(coefficients)(grid),
            "tolerance": 1.0e-6,
            "alpha": None,
        }
        for name, coefficients in specs
    ]


def _ridge_polynomial_spec(config: dict[str, Any]) -> dict[str, Any]:
    context = build_approximation_context(config)
    result = evaluate_polynomial_approximation(
        context=context,
        alpha=float(config["alpha"]),
        degree=int(config["ridge_degree"]),
        method=str(config["ridge_polynomial_method"]),
        grid_size=int(config["ridge_grid_size"]),
    )
    mask = np.asarray(result.evaluation_kind) == "grid"
    return {
        "polynomial_name": "bounded_ridge_tikhonov_target",
        "target_type": "ridge_tikhonov_bounded_target",
        "alpha": float(config["alpha"]),
        "coefficients": result.power_coefficients,
        "degree": int(result.degree),
        "grid": result.evaluation_points[mask],
        "target": result.bounded_target_values[mask],
        "tolerance": float(config["target_tolerance"]),
        "polynomial_approximation_error": float(np.max(result.pointwise_errors)),
    }


def _value_rows(
    *,
    spec: dict[str, Any],
    row: dict[str, Any],
    grid: np.ndarray,
    target: np.ndarray,
    response: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "polynomial_name": spec["polynomial_name"],
            "target_type": spec["target_type"],
            "alpha_if_applicable": spec.get("alpha"),
            "degree": int(spec["degree"]),
            "phase_order": row["phase_order"],
            "phase_sign": row["phase_sign"],
            "phase_offset_rule": row["phase_offset_rule"],
            "signal_operator_convention": row["signal_operator_convention"],
            "response_component": row["response_component"],
            "sigma_normalized": float(sigma),
            "target_value": float(target_value),
            "response_value": float(response_value),
            "pointwise_error": float(error),
        }
        for sigma, target_value, response_value, error in zip(
            grid,
            target,
            response,
            errors,
            strict=True,
        )
    ]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if np.isfinite(row["max_pointwise_error"])]
    if not finite:
        return rows[0]
    return min(finite, key=lambda row: float(row["max_pointwise_error"]))


def _convention_label(row: dict[str, Any]) -> str:
    return (
        f"{row['phase_order']}/{row['phase_sign']}/{row['phase_offset_rule']}/"
        f"{row['signal_operator_convention']}/{row['response_component']}"
    )


def _best_convention_report(summary: pd.DataFrame, sanity: pd.DataFrame) -> str:
    if summary.empty:
        return f"# Phase-Response Convention Diagnostics\n\n{NO_PASS_MESSAGE}\n"
    ridge = summary[summary["target_type"] == "ridge_tikhonov_bounded_target"]
    canonical = _canonical_rows(summary)
    best_ridge = None if ridge.empty else ridge.sort_values("max_pointwise_error").iloc[0]
    best_sanity_error = (
        float(sanity["best_max_pointwise_error"].max()) if not sanity.empty else np.nan
    )
    canonical_sanity = canonical[canonical["target_type"] == "sanity_polynomial"]
    canonical_sanity_error = (
        float(canonical_sanity["max_pointwise_error"].max())
        if not canonical_sanity.empty
        else np.nan
    )
    ridge_status = NO_PASS_MESSAGE
    if best_ridge is not None and bool(best_ridge["passed_1e_minus_3"]):
        ridge_status = PASS_MESSAGE
    sanity_lines = "\n".join(
        (
            f"- `{row.polynomial_name}`: `{row.best_status}`, "
            f"best max error `{row.best_max_pointwise_error:.6g}`"
        )
        for row in sanity.itertuples()
    )
    best_ridge_text = "not available"
    if best_ridge is not None:
        best_ridge_text = (
            f"{_convention_label(best_ridge)} with max error "
            f"{float(best_ridge['max_pointwise_error']):.6g}"
        )
    return f"""# Phase-Response Convention Diagnostics

## Executive Summary

Sanity-polynomial maximum best error: `{best_sanity_error:.6g}`.

Canonical PennyLane `original/phi/none/pennylane_rx_pcphase/real_u00`
sanity maximum error: `{canonical_sanity_error:.6g}`.

Best Ridge/Tikhonov bounded-target convention: {best_ridge_text}.

{ridge_status}

## Sanity Polynomial Status

{sanity_lines}

## Claim Boundary

{PHASE_RESPONSE_CAVEAT}
"""


def _canonical_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "phase_order",
        "phase_sign",
        "phase_offset_rule",
        "signal_operator_convention",
        "response_component",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()
    return summary[
        (summary["phase_order"] == "original")
        & (summary["phase_sign"] == "phi")
        & (summary["phase_offset_rule"] == "none")
        & (summary["signal_operator_convention"] == "pennylane_rx_pcphase")
        & (summary["response_component"] == "real_u00")
    ]


def _dependency_skip_row(config: dict[str, Any], dependency_available: bool) -> dict[str, Any]:
    return {
        "polynomial_name": "dependency_check",
        "target_type": "dependency_check",
        "alpha_if_applicable": float(config["alpha"]),
        "degree": int(config["ridge_degree"]),
        "phase_method": "pennylane_poly_to_angles",
        "dependency_used": "none",
        "phase_count": 0,
        "phase_order": "",
        "phase_sign": "",
        "phase_offset_rule": "",
        "signal_operator_convention": "",
        "response_component": "",
        "coefficient_basis_input": "monomial_power_low_to_high",
        "coefficient_basis_expected": "monomial_power_low_to_high",
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "passed_1e_minus_2": False,
        "passed_5e_minus_3": False,
        "passed_1e_minus_3": False,
        "status": "skipped_dependency_missing",
        "failure_reason": (
            "PennyLane unavailable"
            if not dependency_available
            else "dependency forced missing by config"
        ),
        "caveat": PHASE_RESPONSE_CAVEAT,
    }


def _synthesis_failure_row(
    spec: dict[str, Any],
    status: str,
    reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "polynomial_name": spec["polynomial_name"],
        "target_type": spec["target_type"],
        "alpha_if_applicable": spec.get("alpha"),
        "degree": int(spec["degree"]),
        "phase_method": "pennylane_poly_to_angles",
        "dependency_used": "pennylane",
        "phase_count": 0,
        "phase_order": "",
        "phase_sign": "",
        "phase_offset_rule": "",
        "signal_operator_convention": "",
        "response_component": "",
        "coefficient_basis_input": "monomial_power_low_to_high",
        "coefficient_basis_expected": "monomial_power_low_to_high",
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "passed_1e_minus_2": False,
        "passed_5e_minus_3": False,
        "passed_1e_minus_3": False,
        "status": status,
        "failure_reason": reason,
        "target_tolerance": float(config["target_tolerance"]),
        "caveat": PHASE_RESPONSE_CAVEAT,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase_response_convention_diagnostics",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "ridge_degree": 35,
        "ridge_polynomial_method": "odd_chebyshev_minimax_lp",
        "ridge_grid_size": 256,
        "target_tolerance": 1.0e-3,
        "sanity_tolerance": 1.0e-6,
        "angle_solver": "root-finding",
        "bound_validation_grid_size": 1001,
        "bound_tolerance": 1.0e-5,
        "phase_order": ["original", "reversed"],
        "phase_sign": ["phi", "minus_phi"],
        "phase_offset_rule": [
            "none",
            "first_plus_pi_over_2",
            "first_minus_pi_over_2",
            "last_plus_pi_over_2",
            "last_minus_pi_over_2",
        ],
        "signal_operator_convention": [
            "pennylane_rx_pcphase",
            "repository_plus_i",
            "pennylane_rx_existing_order",
            "repository_plus_i_existing_order",
        ],
        "response_component": [
            "real_u00",
            "imag_u00",
            "negative_real_u00",
            "negative_imag_u00",
            "abs_u00",
        ],
        "write_all_response_values": "true",
        "force_dependency_missing": False,
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose QSVT phase-response conventions")
    parser.parse_args(argv)
    run = diagnose_phase_response_conventions()
    print(f"QSVT phase-response convention diagnostics complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
