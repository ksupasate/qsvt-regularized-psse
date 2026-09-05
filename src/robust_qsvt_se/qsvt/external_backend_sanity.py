from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.external_phase_candidates import ExternalPhaseCandidate
from robust_qsvt_se.qsvt.phase_backend_adapters import available_backend_adapters
from robust_qsvt_se.utils.io import ensure_directory, write_json

SANITY_CAVEAT = (
    "External-backend scalar sanity regression only. Passing sanity rows does not "
    "imply bounded Ridge/Tikhonov target phase validation."
)


def run_external_backend_sanity_regression(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    adapters = available_backend_adapters(enable_local_optimization=True)
    for adapter in adapters:
        for candidate in _sanity_candidates(int(resolved["grid_size"])):
            result = adapter.synthesize(candidate)
            if result.status != "passed_synthesis" or result.phases is None:
                rows.append(_skip_or_failure_row(adapter.backend_name, candidate, result))
                continue
            phases = np.asarray(result.phases, dtype=np.float64)
            try:
                response = adapter.evaluate_response(candidate.full_domain_grid, phases, candidate)
                errors = np.abs(response - candidate.full_domain_target)
                max_error = float(np.max(errors))
                passed = bool(max_error <= float(resolved["tolerance"]))
                status = "passed" if passed else "failed_response_validation"
                failure_reason = "" if passed else "sanity response exceeds tolerance"
                rows.append(
                    {
                        "backend_name": adapter.backend_name,
                        "target_name": candidate.candidate_name,
                        "degree": int(candidate.degree),
                        "phase_count": int(phases.size),
                        "max_error": max_error,
                        "mean_error": float(np.mean(errors)),
                        "rms_error": float(np.sqrt(np.mean(errors**2))),
                        "passed": passed,
                        "status": status,
                        "failure_reason": failure_reason,
                    }
                )
                value_rows.extend(
                    _value_rows(
                        adapter.backend_name,
                        candidate,
                        response,
                        errors,
                    )
                )
            except Exception as exc:
                rows.append(_response_failure_row(adapter.backend_name, candidate, str(exc)))

    summary = pd.DataFrame(rows)
    values = pd.DataFrame(value_rows)
    summary_csv = output_dir / "external_backend_sanity_summary.csv"
    summary_json = output_dir / "external_backend_sanity_summary.json"
    values_csv = output_dir / "external_backend_sanity_response_values.csv"
    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": rows, "caveat": SANITY_CAVEAT})
    values.to_csv(values_csv, index=False)
    manifest = write_manifest(
        output_dir,
        artifacts={
            "external_backend_sanity_summary_csv": str(summary_csv),
            "external_backend_sanity_summary_json": str(summary_json),
            "external_backend_sanity_response_values_csv": str(values_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "external_backend_sanity_summary_csv": summary_csv,
            "external_backend_sanity_summary_json": summary_json,
            "external_backend_sanity_response_values_csv": values_csv,
            "manifest": manifest,
        },
    }


def sanity_passed_backends(summary: pd.DataFrame) -> set[str]:
    passed: set[str] = set()
    if summary.empty:
        return passed
    for backend, group in summary.groupby("backend_name"):
        if set(group["target_name"]) == {
            "x",
            "0.5x",
            "x^3",
            "0.5x_plus_0.25x^3",
        } and bool(group["passed"].all()):
            passed.add(str(backend))
    return passed


def _sanity_candidates(grid_size: int) -> list[ExternalPhaseCandidate]:
    specs = [
        ("x", [0.0, 1.0]),
        ("0.5x", [0.0, 0.5]),
        ("x^3", [0.0, 0.0, 0.0, 1.0]),
        ("0.5x_plus_0.25x^3", [0.0, 0.5, 0.0, 0.25]),
    ]
    grid = np.linspace(-0.95, 0.95, grid_size)
    candidates: list[ExternalPhaseCandidate] = []
    for name, coefficients in specs:
        monomial = np.asarray(coefficients, dtype=np.float64)
        cheb = Polynomial(monomial).convert(kind=Chebyshev).coef
        target = Polynomial(monomial)(grid)
        candidates.append(
            ExternalPhaseCandidate(
                candidate_name=name,
                alpha=np.nan,
                degree=int(monomial.size - 1),
                native_basis="sanity_polynomial",
                method="sanity_polynomial",
                lambda_if_any=None,
                chebyshev_coefficients=cheb,
                monomial_coefficients=monomial,
                full_domain_grid=grid,
                full_domain_target=target,
                full_domain_polynomial=target,
                actual_singular_values=grid,
                actual_singular_targets=target,
                actual_singular_polynomial=target,
                domain_min=0.0,
                domain_max=0.95,
                bounded_scaling_C=1.0,
                native_max_error_full_domain=0.0,
                native_max_error_actual_singular_values=0.0,
                native_max_abs_value=float(np.max(np.abs(target))),
                bounded_in_native_basis=True,
                parity_error=0.0,
                monomial_dynamic_range=1.0,
                monomial_bounded_after_conversion=True,
                supported_input_bases=("chebyshev", "monomial", "function_values"),
            )
        )
    return candidates


def _skip_or_failure_row(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    result: Any,
) -> dict[str, Any]:
    return {
        "backend_name": backend_name,
        "target_name": candidate.candidate_name,
        "degree": int(candidate.degree),
        "phase_count": int(result.phase_count),
        "max_error": np.nan,
        "mean_error": np.nan,
        "rms_error": np.nan,
        "passed": False,
        "status": result.status,
        "failure_reason": result.error_message or "",
    }


def _response_failure_row(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    reason: str,
) -> dict[str, Any]:
    return {
        "backend_name": backend_name,
        "target_name": candidate.candidate_name,
        "degree": int(candidate.degree),
        "phase_count": 0,
        "max_error": np.nan,
        "mean_error": np.nan,
        "rms_error": np.nan,
        "passed": False,
        "status": "failed_response_validation",
        "failure_reason": reason,
    }


def _value_rows(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    response: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "backend_name": backend_name,
            "target_name": candidate.candidate_name,
            "degree": int(candidate.degree),
            "x": float(x),
            "target_value": float(target),
            "phase_response_value": float(value),
            "pointwise_error": float(error),
        }
        for x, target, value, error in zip(
            candidate.full_domain_grid,
            candidate.full_domain_target,
            response,
            errors,
            strict=True,
        )
    ]


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_external_backend_sanity_regression",
        "grid_size": 101,
        "tolerance": 1.0e-6,
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run external backend sanity regression")
    parser.parse_args(argv)
    run = run_external_backend_sanity_regression()
    print(f"External backend sanity regression complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
