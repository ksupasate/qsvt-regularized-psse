from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json

EXTERNAL_AUDIT_CAVEAT = (
    "External backend audit only. Installing or detecting a backend is not target-level "
    "phase validation, hardware execution, quantum speedup, quantum advantage, or evidence "
    "that QSVT outperforms Ridge/Tikhonov."
)

EXTERNAL_BACKEND_COLUMNS = [
    "backend_name",
    "package_name",
    "available_before_install",
    "install_attempted",
    "install_success",
    "version",
    "api_entry_points",
    "accepts_monomial",
    "accepts_chebyshev",
    "accepts_function_values",
    "requires_low_to_high_order",
    "supports_qsp",
    "supports_qsvt",
    "supports_gqsp",
    "returns_phase_angles",
    "has_response_evaluator",
    "status",
    "notes",
]


def install_or_audit_external_phase_backends(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    install = bool(resolved["install"])
    before = _availability_snapshot()
    install_log_lines: list[str] = []

    if install:
        for package in resolved["install_packages"]:
            result = _pip_install(str(package))
            install_log_lines.append(result)
    else:
        install_log_lines.append("Install not requested. Audit only.")

    rows = _audit_rows(before, install_attempted=install, install_log="\n".join(install_log_lines))
    frame = pd.DataFrame(rows, columns=EXTERNAL_BACKEND_COLUMNS)

    summary_csv = output_dir / "external_backend_audit_summary.csv"
    summary_json = output_dir / "external_backend_audit_summary.json"
    install_log = output_dir / "external_backend_install_log.txt"
    capabilities_md = output_dir / "external_backend_capabilities.md"
    frame.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": rows, "caveat": EXTERNAL_AUDIT_CAVEAT})
    install_log.write_text("\n\n".join(install_log_lines), encoding="utf-8")
    capabilities_md.write_text(_capabilities_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "external_backend_audit_summary_csv": str(summary_csv),
            "external_backend_audit_summary_json": str(summary_json),
            "external_backend_install_log_txt": str(install_log),
            "external_backend_capabilities_md": str(capabilities_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "external_backend_audit_summary_csv": summary_csv,
            "external_backend_audit_summary_json": summary_json,
            "external_backend_install_log_txt": install_log,
            "external_backend_capabilities_md": capabilities_md,
            "manifest": manifest,
        },
    }


def _audit_rows(
    before: dict[str, bool],
    *,
    install_attempted: bool,
    install_log: str,
) -> list[dict[str, Any]]:
    pyqsp_available = _module_available("pyqsp")
    pennylane_available = _module_available("pennylane")
    scipy_available = _module_available("scipy")
    return [
        _row(
            backend_name="pennylane_poly_to_angles",
            package_name="pennylane",
            available_before_install=before["pennylane"],
            install_attempted=False,
            install_success=pennylane_available,
            version=_version("pennylane") if pennylane_available else "",
            api_entry_points=_entry_points("pennylane", ["poly_to_angles"]),
            accepts_monomial=True,
            accepts_chebyshev=False,
            accepts_function_values=False,
            requires_low_to_high_order=True,
            supports_qsp=True,
            supports_qsvt=True,
            supports_gqsp=False,
            returns_phase_angles=pennylane_available,
            has_response_evaluator=False,
            status="available" if pennylane_available else "unavailable",
            notes="Monomial low-to-high coefficients only; unsafe high-degree conversion is gated.",
        ),
        _row(
            backend_name="pyqsp_sym_qsp",
            package_name="pyqsp",
            available_before_install=before["pyqsp"],
            install_attempted=install_attempted,
            install_success=pyqsp_available,
            version=_version("pyqsp") if pyqsp_available else "",
            api_entry_points=_entry_points(
                "pyqsp.angle_sequence",
                ["QuantumSignalProcessingPhases", "ComputeQSPResponse"],
            ),
            accepts_monomial=False,
            accepts_chebyshev=True,
            accepts_function_values=False,
            requires_low_to_high_order=True,
            supports_qsp=True,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=pyqsp_available,
            has_response_evaluator=pyqsp_available,
            status="available" if pyqsp_available else _install_status("pyqsp", install_log),
            notes=(
                "Uses pyqsp symmetric QSP with Chebyshev coefficients and local "
                "response validation."
            ),
        ),
        _row(
            backend_name="qsppack_if_available",
            package_name="QSPPACK",
            available_before_install=before["qsppack"],
            install_attempted=False,
            install_success=False,
            version="",
            api_entry_points="",
            accepts_monomial=False,
            accepts_chebyshev=False,
            accepts_function_values=False,
            requires_low_to_high_order=False,
            supports_qsp=True,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=False,
            has_response_evaluator=False,
            status="not_directly_usable_from_python",
            notes="QSPPACK is treated as unavailable unless a callable Python package is present.",
        ),
        _row(
            backend_name="local_optimization_qsp",
            package_name="scipy",
            available_before_install=before["scipy"],
            install_attempted=False,
            install_success=scipy_available,
            version=_version("scipy") if scipy_available else "",
            api_entry_points=_entry_points("scipy.optimize", ["least_squares", "minimize"]),
            accepts_monomial=False,
            accepts_chebyshev=False,
            accepts_function_values=True,
            requires_low_to_high_order=False,
            supports_qsp=True,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=scipy_available,
            has_response_evaluator=True,
            status="available_experimental" if scipy_available else "unavailable",
            notes=(
                "Optimization-based scalar phase fitting; not theorem-level certified "
                "phase synthesis."
            ),
        ),
        _row(
            backend_name="qiskit_qsp_utilities",
            package_name="qiskit",
            available_before_install=before["qiskit"],
            install_attempted=False,
            install_success=_module_available("qiskit"),
            version=_version("qiskit") if _module_available("qiskit") else "",
            api_entry_points=_qiskit_entry_points(),
            accepts_monomial=False,
            accepts_chebyshev=False,
            accepts_function_values=False,
            requires_low_to_high_order=False,
            supports_qsp=False,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=False,
            has_response_evaluator=False,
            status="package_available_no_validated_phase_api"
            if _module_available("qiskit")
            else "unavailable",
            notes="No callable QSP/QSVT polynomial phase-factor API was found.",
        ),
        _row(
            backend_name="sympy_high_precision_conversion",
            package_name="sympy",
            available_before_install=before["sympy"],
            install_attempted=install_attempted,
            install_success=_module_available("sympy"),
            version=_version("sympy") if _module_available("sympy") else "",
            api_entry_points=_entry_points("sympy", ["Poly", "chebyshevt"]),
            accepts_monomial=True,
            accepts_chebyshev=True,
            accepts_function_values=False,
            requires_low_to_high_order=False,
            supports_qsp=False,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=False,
            has_response_evaluator=True,
            status="available_conversion_utility"
            if _module_available("sympy")
            else _install_status("sympy", install_log),
            notes="High-precision conversion utility only; not a phase backend.",
        ),
        _row(
            backend_name="mpmath_high_precision_conversion",
            package_name="mpmath",
            available_before_install=before["mpmath"],
            install_attempted=install_attempted,
            install_success=_module_available("mpmath"),
            version=_version("mpmath") if _module_available("mpmath") else "",
            api_entry_points=_entry_points("mpmath", ["mp"]),
            accepts_monomial=True,
            accepts_chebyshev=True,
            accepts_function_values=False,
            requires_low_to_high_order=False,
            supports_qsp=False,
            supports_qsvt=False,
            supports_gqsp=False,
            returns_phase_angles=False,
            has_response_evaluator=True,
            status="available_conversion_utility"
            if _module_available("mpmath")
            else _install_status("mpmath", install_log),
            notes="High-precision arithmetic utility only; not a phase backend.",
        ),
    ]


def _pip_install(package: str) -> str:
    command = [sys.executable, "-m", "pip", "install", package]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        return f"$ {' '.join(command)}\nFAILED: {type(exc).__name__}: {exc}"
    status = "SUCCESS" if completed.returncode == 0 else f"FAILED({completed.returncode})"
    return (
        f"$ {' '.join(command)}\n{status}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


def _availability_snapshot() -> dict[str, bool]:
    return {
        "pennylane": _module_available("pennylane"),
        "pyqsp": _module_available("pyqsp"),
        "qsppack": _module_available("qsppack") or _module_available("QSPPACK"),
        "qiskit": _module_available("qiskit"),
        "scipy": _module_available("scipy"),
        "sympy": _module_available("sympy"),
        "mpmath": _module_available("mpmath"),
    }


def _row(**kwargs: Any) -> dict[str, Any]:
    return {column: kwargs[column] for column in EXTERNAL_BACKEND_COLUMNS}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except Exception:
        module = importlib.import_module(package)
        return str(getattr(module, "__version__", ""))


def _entry_points(module_name: str, names: list[str]) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return ""
    available = [name for name in names if hasattr(module, name)]
    return ";".join(available)


def _qiskit_entry_points() -> str:
    candidates = [
        "qiskit.synthesis.qsp",
        "qiskit.synthesis.qsvt",
        "qiskit.quantum_info.qsp",
    ]
    return ";".join(name for name in candidates if _module_available(name))


def _install_status(package: str, install_log: str) -> str:
    if not install_log or "Install not requested" in install_log:
        return "unavailable_install_not_attempted"
    if package in install_log and "SUCCESS" in install_log:
        return "available_after_install"
    return "unavailable_after_install_attempt"


def _capabilities_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# External QSP/QSVT Phase Backend Audit",
        "",
        "## Backend Table",
        "",
        "| backend | package | version | Chebyshev | monomial | phases | status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            f"{row.backend_name} | {row.package_name} | {row.version} | "
            f"{row.accepts_chebyshev} | {row.accepts_monomial} | "
            f"{row.returns_phase_angles} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Only rows that return phase angles and pass backend sanity regression "
                "are trusted for target-level validation. Chebyshev-capable backends "
                "avoid the unstable high-degree monomial conversion failure mode."
            ),
            "",
            "## Claim Boundary",
            "",
            EXTERNAL_AUDIT_CAVEAT,
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_phase_external_backend_audit",
        "install": False,
        "install_packages": ["pyqsp", "mpmath", "sympy"],
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Install or audit external QSVT phase backends")
    parser.add_argument("--install", action="store_true", help="Attempt optional backend install")
    args = parser.parse_args(argv)
    run = install_or_audit_external_phase_backends({"install": bool(args.install)})
    print(f"External QSVT phase backend audit complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
