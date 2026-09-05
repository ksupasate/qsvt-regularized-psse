from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json

AUDIT_CAVEAT = (
    "Backend audit only. Availability of a phase or polynomial utility is not "
    "bounded Ridge/Tikhonov target phase validation, hardware execution, quantum "
    "speedup, quantum advantage, or evidence that QSVT outperforms Ridge/Tikhonov."
)

AUDIT_COLUMNS = [
    "backend_name",
    "available",
    "version_if_available",
    "accepts_monomial_coefficients",
    "accepts_chebyshev_coefficients",
    "accepts_function_values",
    "requires_low_to_high_order",
    "supports_parity_constraints",
    "supports_boundedness_checks",
    "can_return_phase_angles",
    "can_evaluate_response",
    "status",
    "notes",
]


def audit_phase_backend_options(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = backend_audit_rows()
    frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS)

    summary_csv = output_dir / "phase_backend_audit_summary.csv"
    summary_json = output_dir / "phase_backend_audit_summary.json"
    capabilities_md = output_dir / "phase_backend_capabilities.md"
    frame.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": rows, "caveat": AUDIT_CAVEAT})
    capabilities_md.write_text(_capabilities_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "phase_backend_audit_summary_csv": str(summary_csv),
            "phase_backend_audit_summary_json": str(summary_json),
            "phase_backend_capabilities_md": str(capabilities_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "phase_backend_audit_summary_csv": summary_csv,
            "phase_backend_audit_summary_json": summary_json,
            "phase_backend_capabilities_md": capabilities_md,
            "manifest": manifest,
        },
    }


def backend_audit_rows() -> list[dict[str, Any]]:
    rows = [
        _pennylane_row(),
        _qiskit_row(),
        _pyqsp_row(),
        _qsppack_row(),
        _repository_phase_synthesis_row(),
        _repository_phase_response_row(),
        _numpy_polynomial_row(),
        _scipy_row(),
        _sympy_row(),
        _mpmath_row(),
        _decimal_row(),
        _numpy_longdouble_row(),
    ]
    return rows


def _pennylane_row() -> dict[str, Any]:
    available = _module_available("pennylane")
    version = _version("pennylane") if available else ""
    has_poly_to_angles = False
    if available:
        try:
            module = importlib.import_module("pennylane")
            has_poly_to_angles = callable(getattr(module, "poly_to_angles", None))
        except Exception:
            has_poly_to_angles = False
    return _row(
        backend_name="pennylane.poly_to_angles",
        available=available and has_poly_to_angles,
        version=version,
        accepts_monomial=True,
        accepts_chebyshev=False,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=available and has_poly_to_angles,
        can_evaluate_response=False,
        status=(
            "available_monomial_phase_backend"
            if available and has_poly_to_angles
            else "unavailable"
        ),
        notes=(
            "Accepts low-to-high monomial coefficients; repository wrappers must "
            "perform parity, boundedness, and coefficient-stability checks first."
        ),
    )


def _qiskit_row() -> dict[str, Any]:
    available = _module_available("qiskit")
    version = _version("qiskit") if available else ""
    qsp_available = False
    if available:
        qsp_available = any(
            _module_available(name)
            for name in [
                "qiskit.synthesis.qsvt",
                "qiskit.synthesis.qsp",
                "qiskit.quantum_info.qsp",
            ]
        )
    return _row(
        backend_name="qiskit_qsp_utilities",
        available=qsp_available,
        version=version,
        accepts_monomial=False,
        accepts_chebyshev=False,
        accepts_function_values=False,
        requires_low_to_high=False,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=qsp_available,
        can_evaluate_response=False,
        status="qiskit_available_no_validated_qsp_phase_backend"
        if available and not qsp_available
        else "available"
        if qsp_available
        else "unavailable",
        notes=(
            "Qiskit package availability is distinct from an exposed, validated "
            "polynomial-to-QSP-phase backend."
        ),
    )


def _pyqsp_row() -> dict[str, Any]:
    available = _module_available("pyqsp")
    return _row(
        backend_name="pyqsp",
        available=available,
        version=_version("pyqsp") if available else "",
        accepts_monomial=available,
        accepts_chebyshev=False,
        accepts_function_values=available,
        requires_low_to_high=False,
        supports_parity=available,
        supports_boundedness=available,
        can_return_phases=available,
        can_evaluate_response=available,
        status="available_not_integrated" if available else "unavailable_optional_dependency",
        notes=(
            "Optional package was not installed by this audit."
            if not available
            else "Not integrated."
        ),
    )


def _qsppack_row() -> dict[str, Any]:
    candidates = ["qsppack", "QSPPACK"]
    available = any(_module_available(name) for name in candidates)
    return _row(
        backend_name="QSPPACK",
        available=available,
        version="",
        accepts_monomial=available,
        accepts_chebyshev=available,
        accepts_function_values=available,
        requires_low_to_high=False,
        supports_parity=available,
        supports_boundedness=available,
        can_return_phases=available,
        can_evaluate_response=available,
        status="available_not_integrated" if available else "unavailable_optional_dependency",
        notes="Optional QSPPACK-style package was not installed by this audit.",
    )


def _repository_phase_synthesis_row() -> dict[str, Any]:
    return _row(
        backend_name="repository_custom_least_squares_qsp",
        available=True,
        version="local",
        accepts_monomial=False,
        accepts_chebyshev=False,
        accepts_function_values=False,
        requires_low_to_high=False,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=True,
        can_evaluate_response=True,
        status="available_target_specific_optimizer",
        notes=(
            "The local optimizer synthesizes phases for the built-in regularized "
            "target formula, not arbitrary Chebyshev-basis candidate coefficients."
        ),
    )


def _repository_phase_response_row() -> dict[str, Any]:
    return _row(
        backend_name="repository_scalar_phase_response_evaluator",
        available=True,
        version="local",
        accepts_monomial=False,
        accepts_chebyshev=False,
        accepts_function_values=False,
        requires_low_to_high=False,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_response_evaluator_only",
        notes=(
            "Evaluates scalar responses for existing phase angles; it does not synthesize phases."
        ),
    )


def _numpy_polynomial_row() -> dict[str, Any]:
    return _row(
        backend_name="numpy.polynomial",
        available=True,
        version=_version("numpy"),
        accepts_monomial=True,
        accepts_chebyshev=True,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_polynomial_utility_not_phase_backend",
        notes="Useful for fitting, evaluation, and basis conversion; does not return QSP phases.",
    )


def _scipy_row() -> dict[str, Any]:
    available = _module_available("scipy")
    return _row(
        backend_name="scipy_optimization",
        available=available,
        version=_version("scipy") if available else "",
        accepts_monomial=False,
        accepts_chebyshev=False,
        accepts_function_values=True,
        requires_low_to_high=False,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=False,
        status="available_optimization_utility_not_phase_backend" if available else "unavailable",
        notes="Used for minimax LP and least-squares diagnostics, not direct QSP phase synthesis.",
    )


def _sympy_row() -> dict[str, Any]:
    available = _module_available("sympy")
    return _row(
        backend_name="sympy_high_precision_conversion",
        available=available,
        version=_version("sympy") if available else "",
        accepts_monomial=True,
        accepts_chebyshev=True,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_conversion_utility" if available else "unavailable_optional_dependency",
        notes="Potential exact/symbolic basis conversion utility; not a phase backend.",
    )


def _mpmath_row() -> dict[str, Any]:
    available = _module_available("mpmath")
    return _row(
        backend_name="mpmath_high_precision_conversion",
        available=available,
        version=_version("mpmath") if available else "",
        accepts_monomial=True,
        accepts_chebyshev=True,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_conversion_utility" if available else "unavailable_optional_dependency",
        notes="Potential high-precision conversion/evaluation utility; not a phase backend.",
    )


def _decimal_row() -> dict[str, Any]:
    return _row(
        backend_name="python_decimal",
        available=True,
        version="stdlib",
        accepts_monomial=True,
        accepts_chebyshev=True,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_conversion_utility",
        notes="Standard-library high-precision arithmetic for conversion diagnostics only.",
    )


def _numpy_longdouble_row() -> dict[str, Any]:
    eps = np.finfo(np.longdouble).eps
    wider_than_float64 = bool(eps < np.finfo(np.float64).eps)
    return _row(
        backend_name="numpy.longdouble",
        available=True,
        version=_version("numpy"),
        accepts_monomial=True,
        accepts_chebyshev=True,
        accepts_function_values=False,
        requires_low_to_high=True,
        supports_parity=False,
        supports_boundedness=False,
        can_return_phases=False,
        can_evaluate_response=True,
        status="available_conversion_utility",
        notes=f"longdouble eps={eps:.3g}; wider_than_float64={wider_than_float64}.",
    )


def _row(
    *,
    backend_name: str,
    available: bool,
    version: str,
    accepts_monomial: bool,
    accepts_chebyshev: bool,
    accepts_function_values: bool,
    requires_low_to_high: bool,
    supports_parity: bool,
    supports_boundedness: bool,
    can_return_phases: bool,
    can_evaluate_response: bool,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "backend_name": backend_name,
        "available": bool(available),
        "version_if_available": version,
        "accepts_monomial_coefficients": bool(accepts_monomial),
        "accepts_chebyshev_coefficients": bool(accepts_chebyshev),
        "accepts_function_values": bool(accepts_function_values),
        "requires_low_to_high_order": bool(requires_low_to_high),
        "supports_parity_constraints": bool(supports_parity),
        "supports_boundedness_checks": bool(supports_boundedness),
        "can_return_phase_angles": bool(can_return_phases),
        "can_evaluate_response": bool(can_evaluate_response),
        "status": status,
        "notes": notes,
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return ""


def _capabilities_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# QSVT Phase Backend Capability Audit",
        "",
        "## Executive Summary",
        "",
        (
            "The audit records which installed or local tools can synthesize phases, "
            "evaluate scalar responses, or only assist polynomial conversion."
        ),
        "",
        "No optional package is installed or used by force during this audit.",
        "",
        "## Backend Table",
        "",
        "| backend | available | Chebyshev input | monomial input | returns phases | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            f"{row.backend_name} | {row.available} | "
            f"{row.accepts_chebyshev_coefficients} | "
            f"{row.accepts_monomial_coefficients} | "
            f"{row.can_return_phase_angles} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A direct Chebyshev-basis phase backend is available only if a row "
                "has both `accepts_chebyshev_coefficients` and "
                "`can_return_phase_angles` set to true. In the current environment, "
                "candidate phase synthesis should therefore use PennyLane only for "
                "monomial coefficient rows that pass the safety gates."
            ),
            "",
            "## Claim Boundary",
            "",
            AUDIT_CAVEAT,
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {"output_dir": "outputs/qsvt_phase_backend_audit"}
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit QSVT phase backend options")
    parser.parse_args(argv)
    run = audit_phase_backend_options()
    print(f"QSVT phase backend audit complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
