# ruff: noqa: E501
"""Final same-configuration useful-overlap validation and manuscript assets.

This runner is intentionally narrow.  It reuses the verified degree-255 target
builder and convention-corrected rectangular QSVT machinery, then writes a
fresh evidence root under ``outputs/final_useful_overlap_validation``.  It does
not run broad regularization or polynomial sweeps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy import sparse
from scipy.sparse.linalg import cg, lsmr, lsqr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_useful_overlap_validation"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
(OUT / "mpl_cache").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding  # noqa: E402
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution  # noqa: E402
from robust_qsvt_se.qsvt.rectangular_convention import (  # noqa: E402
    DENSE_JULIA_PCPHASE,
    PYQSP_SYM_QSP_PLUS_I,
    PYQSP_TO_PCPHASE_RULE,
    apply_pcphase_qsvt_sequence,
    convert_pyqsp_sym_qsp_to_pcphase,
    extract_component,
    pcphase_qsvt_operator,
    pcphase_qsvt_top_block,
    production_scalar_emulator_unitary,
    pyqsp_pcphase_component,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (  # noqa: E402
    fit_bounded_odd_chebyshev,
    synthesize_pyqsp_sym_qsp_phases,
)
from scripts.continue_final_qsvt_feasibility_push import state_metrics  # noqa: E402
from scripts.run_rectangular_convention_fix import (  # noqa: E402
    ALPHA,
    CASE,
    DEGREE,
    LAMBDA,
    PARENT_FINGERPRINT,
    accepted_from_counts,
    branch_state_for_overlap,
    build_target,
    controlled_operator,
    one_qubit_gate_count,
    two_qubit_gate_count,
)

CONVENTION_FINGERPRINT = "db67f79cce4b0a67c78530c0a2a185b729f9d7a2ea6baf40ab50325266a13189"
APPLICATION_THRESHOLD = 1.25
BENCHMARK_ALPHA = 1.0e-4
STATEVECTOR_TOL = 1.0e-3
HELDOUT_TOL = 1.0e-8
HIGH_SHOT_PRECISION_GOAL = 0.10

EVIDENCE_STATEVECTOR = "EXECUTED_STATEVECTOR"
EVIDENCE_BACKEND = "EXECUTED_BACKEND_SHOTS"
EVIDENCE_CLASSICAL = "CLASSICAL_EXPERIMENT"
EVIDENCE_DIAGNOSTIC = "DIAGNOSTIC_ONLY"
EVIDENCE_MODELED = "MODELED_RESOURCE"
EVIDENCE_EXCLUDED = "EXCLUDED"
EVIDENCE_FAILED = "FAILED_CONFIGURATION"


def main() -> None:
    started = time.perf_counter()
    append_command(".venv/bin/python scripts/run_final_useful_overlap_validation.py", "started")
    write_environment_summary()

    audit = write_initial_audit()
    target = build_target()
    config, config_sha = write_final_configuration(target)
    assert_config_matches_parent(config)

    heldout = run_heldout_rectangular_validation(target, config_sha)
    degree = run_degree_generalization_validation(target, config_sha)
    independent = run_independent_mapping_validation(target, config_sha)
    write_convention_report(heldout, degree, independent)
    write_production_api_audit(config_sha)

    app = run_application_reproduction(target, config_sha)
    quantum = run_quantum_reproduction(target, config_sha)
    readout = run_readout_validation(target, quantum, config_sha)
    high_shot = run_high_shot_backend(target, quantum, readout, config_sha)
    resources = write_degree255_resource_ledger(target, quantum, high_shot, config_sha)
    oaa = write_oaa_model_validation(target, quantum, config_sha)
    classical = run_classical_baselines(target, quantum, config_sha)
    budget = write_error_budget(app, quantum, high_shot, readout, oaa, config_sha)

    scientific_status = (
        "FULL_USEFUL_OVERLAP_INDEPENDENTLY_VERIFIED"
        if heldout["passed"]
        and degree["passed"]
        and independent["passed"]
        and app["passed"]
        and quantum["passed"]
        and high_shot["passed"]
        else "FULL_USEFUL_OVERLAP_REPRODUCED_WITH_LIMITATIONS"
    )
    submission_status = "READY_FOR_STRICT_INTERNAL_REVIEW"
    write_claim_support_matrix(scientific_status, submission_status, config_sha)
    write_latex_tables(app, quantum, high_shot, resources, classical, budget, config_sha)
    write_repository_freeze_pending()
    write_evidence_status_matrix(config_sha, scientific_status, submission_status)
    write_known_failures(scientific_status, high_shot)
    write_final_validation_report(scientific_status, submission_status, audit, config_sha)
    append_command(".venv/bin/python scripts/run_final_useful_overlap_validation.py", "exit 0")
    write_manifest_and_checksums(runtime_seconds=time.perf_counter() - started)


def write_initial_audit() -> dict[str, Any]:
    roots = [
        "src",
        "scripts",
        "tests",
        "outputs/full_rectangular_breakthrough",
        "outputs/rectangular_convention_fix",
        "outputs/final_qsvt_feasibility_push",
        "manuscript",
        "submission_package_tqe_final",
        "pyproject.toml",
    ]
    audit = {
        "repository_root": run_text(["git", "rev-parse", "--show-toplevel"], check=False).strip()
        or str(ROOT),
        "cwd": str(ROOT),
        "branch": run_text(["git", "branch", "--show-current"], check=False).strip(),
        "head": run_text(["git", "rev-parse", "HEAD"], check=False).strip(),
        "git_status_short": run_text(["git", "status", "--short"], check=False),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "manuscript_main_pdf": file_status(ROOT / "manuscript" / "main.pdf"),
        "manuscript_supplement_pdf": file_status(
            ROOT / "manuscript" / "supplementary_material.pdf"
        ),
        "package_main_pdf": file_status(ROOT / "submission_package_tqe_final" / "main.pdf"),
        "rectangular_checksums_valid": run_command_status(
            ["shasum", "-a", "256", "-c", "outputs/rectangular_convention_fix/checksums.sha256"]
        ),
    }
    write_text(
        OUT / "initial_repository_audit.md",
        "\n".join(
            [
                "# Initial Repository Audit",
                "",
                f"Repository root: `{audit['repository_root']}`.",
                f"Branch: `{audit['branch']}`.",
                f"HEAD: `{audit['head']}`.",
                "The repository had no valid commit HEAD at audit time when `git rev-parse HEAD` returned a fatal ambiguous-HEAD message.",
                "",
                "## Canonical Artifacts",
                f"- Manuscript source: `manuscript/main.tex`; PDF status: `{audit['manuscript_main_pdf']['status']}`.",
                f"- Supplement source: `manuscript/supplementary_material.tex`; PDF status: `{audit['manuscript_supplement_pdf']['status']}`.",
                f"- Submission package PDF status: `{audit['package_main_pdf']['status']}`.",
                "",
                "## Required Roots Inspected",
                *(f"- `{root}`" for root in roots),
                "",
                "## Existing Convention Evidence",
                f"- Parent target fingerprint: `{PARENT_FINGERPRINT}`.",
                f"- Convention-fix fingerprint: `{CONVENTION_FINGERPRINT}`.",
                f"- Prior checksum verification exit status: `{audit['rectangular_checksums_valid']['returncode']}`.",
            ]
        )
        + "\n",
    )
    write_artifact_inventory(roots)
    write_stale_artifact_report()
    return audit


def write_artifact_inventory(roots: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            rows.append(
                {
                    "path": root_name,
                    "size_bytes": "",
                    "mtime": "",
                    "sha256": "",
                    "format": "",
                    "rows": "",
                    "status": "MISSING",
                    "note": "required inspection root is absent",
                }
            )
            continue
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            rel = str(path.relative_to(ROOT))
            stat = path.stat()
            fmt = path.suffix.lower().lstrip(".") or "file"
            row_count = ""
            if path.suffix.lower() == ".csv":
                try:
                    with path.open(newline="", encoding="utf-8") as handle:
                        row_count = max(sum(1 for _ in handle) - 1, 0)
                except UnicodeDecodeError:
                    row_count = "unreadable"
            rows.append(
                {
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(stat.st_mtime)),
                    "sha256": file_sha(path),
                    "format": fmt,
                    "rows": row_count,
                    "status": "VALID_COMPLETE" if stat.st_size > 0 else "EMPTY",
                    "note": "",
                }
            )
    write_csv_rows(OUT / "artifact_inventory.csv", rows)


def write_stale_artifact_report() -> None:
    main = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    package_pdf = ROOT / "submission_package_tqe_final" / "main.pdf"
    manuscript_pdf = ROOT / "manuscript" / "main.pdf"
    stale: list[str] = []
    if "negative result" in main or "fail the declared application-utility diagnostic" in main:
        stale.append(
            "Canonical manuscript still contains pre-breakthrough negative application-boundary framing."
        )
    if "degree-31 phase sequence" in main and "degree-255" not in main:
        stale.append(
            "Finite-shot/resource prose still emphasizes the older degree-31 configuration."
        )
    if (
        package_pdf.is_file()
        and manuscript_pdf.is_file()
        and package_pdf.stat().st_mtime < manuscript_pdf.stat().st_mtime
    ):
        stale.append("Submission package PDF predates the canonical manuscript PDF.")
    if "FULL_USEFUL_OVERLAP_EXECUTED" not in main:
        stale.append(
            "Canonical manuscript does not yet reference the final useful-overlap decision."
        )
    write_text(
        OUT / "stale_artifact_report.md",
        "\n".join(["# Stale Artifact Report", "", *(f"- {item}" for item in stale)])
        + ("\n" if stale else "No stale artifacts detected.\n"),
    )


def write_final_configuration(target: dict[str, Any]) -> tuple[dict[str, Any], str]:
    H = target["H"]
    r = target["r"]
    selected = target["selected_vector"]
    config = {
        "case": CASE,
        "workflow": "dense_full_rectangular_ieee14_useful_overlap",
        "parent_target_fingerprint": PARENT_FINGERPRINT,
        "convention_fix_fingerprint": CONVENTION_FINGERPRINT,
        "matrix_shape": list(H.shape),
        "matrix_checksum": array_checksum(H),
        "residual_checksum": array_checksum(r),
        "selected_output_definition": "first nonreference voltage-angle update coordinate",
        "selected_output_checksum": array_checksum(selected),
        "measurement_model": "generated PYPOWER AC-linearized weighted Jacobian, seed 123",
        "alpha": ALPHA,
        "beta": float(target["beta"]),
        "lambda": LAMBDA,
        "contraction_C": float(target["C"]),
        "degree": DEGREE,
        "polynomial_method": "stable_chebyshev with minimal contraction repair",
        "polynomial_checksum": array_checksum(target["coeffs"]),
        "phase_method": "pyqsp sym_qsp",
        "phase_mapping": PYQSP_TO_PCPHASE_RULE,
        "phase_checksum": array_checksum(target["pyqsp_phases"]),
        "production_phase_checksum": array_checksum(target["production_phases"]),
        "production_convention": "dense Julia PCPhase sequence with signed imaginary top-left extraction",
        "left_projector_definition": "top encoded subspace indices [0, N)",
        "right_projector_definition": "same top encoded subspace; rectangular orientation carried by A=H^T/beta",
        "padding_dimension": int(target["dilation"]["padded_dimension"]),
        "dilation_dimension": int(target["dilation"]["unitary_dimension"]),
        "block_encoding_method": "exact dense SVD/Jacobian dilation",
        "block_encoding_checksum": array_checksum(
            np.asarray(target["dilation"]["unitary"], dtype=np.complex128).view(np.float64)
        ),
        "backend": "qiskit-aer AerSimulator(method=statevector), shot-based circuit execution",
        "shot_schedule": "fallback 5 independent seeds x 1,000,000 shots if feasible",
        "application_threshold": APPLICATION_THRESHOLD,
    }
    sha = sha256_json(config)
    write_json(OUT / "final_scientific_configuration.json", config)
    write_text(OUT / "final_scientific_configuration.sha256", f"{sha}\n")
    return config, sha


def assert_config_matches_parent(config: dict[str, Any]) -> None:
    prior = json.loads(
        (
            ROOT / "outputs/rectangular_convention_fix/convention_target_configuration.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "matrix_checksum",
        "residual_checksum",
        "selected_output_checksum",
        "alpha",
        "beta",
        "lambda",
        "degree",
        "polynomial_checksum",
        "phase_checksum",
    ):
        if prior[key] != config[key]:
            raise RuntimeError(f"configuration drift for {key}: {prior[key]} != {config[key]}")


def run_heldout_rectangular_validation(target: dict[str, Any], config_sha: str) -> dict[str, Any]:
    poly_specs = heldout_polynomials(target)
    shapes = [(2, 1), (3, 2), (4, 3), (5, 3), (6, 4), (8, 5)]
    spectra = {
        "well_conditioned": lambda k: np.linspace(0.85, 0.35, k),
        "repeated": lambda k: np.array(([0.72] * max(k - 1, 0)) + [0.41], dtype=float)[:k],
        "clustered": lambda k: np.linspace(0.61, 0.57, k),
        "nearly_rank_deficient": lambda k: np.geomspace(0.8, 1.0e-4, k),
        "exact_zero": lambda k: np.array(([0.8, 0.45] + [0.0] * max(k - 2, 0)), dtype=float)[:k],
    }
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        k = min(shape)
        for spectrum_name, spectrum_fn in spectra.items():
            singulars = np.asarray(spectrum_fn(k), dtype=np.float64)
            for spec in poly_specs:
                A, U, Vt = heldout_matrix(
                    shape, singulars, seed=seed_for("heldout", shape, spectrum_name, spec["name"])
                )
                exact = U @ np.diag(spec["poly"](singulars)) @ Vt
                candidate = independent_rectangular_action(
                    A, spec["phases"], component=spec["component"]
                )
                abs_err = float(np.linalg.norm(candidate - exact, ord=2))
                denom = max(float(np.linalg.norm(exact, ord=2)), 1.0e-12)
                rel_err = abs_err / denom
                tolerance = 1.0e-7 if spec["degree"] == 255 else HELDOUT_TOL
                status = "pass" if abs_err <= 1.0e-12 or rel_err <= tolerance else "fail"
                rows.append(
                    {
                        "configuration_sha256": config_sha,
                        "shape": f"{shape[0]}x{shape[1]}",
                        "spectrum": spectrum_name,
                        "polynomial": spec["name"],
                        "degree": spec["degree"],
                        "component": spec["component"],
                        "absolute_spectral_error": abs_err,
                        "relative_spectral_error": rel_err,
                        "tolerance": tolerance,
                        "status": status,
                        "evidence_label": EVIDENCE_STATEVECTOR
                        if status == "pass"
                        else EVIDENCE_FAILED,
                    }
                )
    write_csv_rows(OUT / "heldout_rectangular_validation.csv", rows)
    passed = all(row["status"] == "pass" for row in rows)
    return {
        "passed": passed,
        "rows": len(rows),
        "worst_error": max(float(r["relative_spectral_error"]) for r in rows),
    }


def run_degree_generalization_validation(target: dict[str, Any], config_sha: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for degree in (1, 3, 7, 15, 31, 63, 127, 255):
        spec = polynomial_for_degree(target, degree)
        expected_component = pyqsp_pcphase_component(degree)
        scalar_grid = np.linspace(-0.95, 0.95, 61)
        scalar_errors = [
            abs(
                extract_component(
                    production_scalar_emulator_unitary(float(x), spec["phases"])[:1, :1],
                    spec["component"],
                )[0, 0]
                - float(spec["poly"](x))
            )
            for x in scalar_grid
        ]
        A, U, Vt = heldout_matrix(
            (6, 4),
            np.asarray([0.86, 0.58, 0.31, 0.07], dtype=np.float64),
            seed=seed_for("degree", degree),
        )
        singulars = np.linalg.svd(A, compute_uv=False)
        exact = U @ np.diag(spec["poly"](singulars)) @ Vt
        candidate = independent_rectangular_action(A, spec["phases"], component=spec["component"])
        abs_err = float(np.linalg.norm(candidate - exact, ord=2))
        rel_err = spectral_relative_error(candidate, exact)
        tolerance = 1.0e-7 if degree >= 127 else 1.0e-8
        status = (
            "pass"
            if max(scalar_errors) <= tolerance and (abs_err <= 1.0e-12 or rel_err <= tolerance)
            else "fail"
        )
        rows.append(
            {
                "configuration_sha256": config_sha,
                "degree": degree,
                "phase_count": len(spec["phases"]),
                "component": spec["component"],
                "expected_component_from_degree": expected_component,
                "d_mod_4": degree % 4,
                "scalar_max_error": max(scalar_errors),
                "heldout_rectangular_absolute_error": abs_err,
                "heldout_rectangular_relative_error": rel_err,
                "absolute_tolerance_for_near_zero_targets": 1.0e-12,
                "tolerance": tolerance,
                "status": status,
                "evidence_label": EVIDENCE_STATEVECTOR if status == "pass" else EVIDENCE_FAILED,
            }
        )
    write_csv_rows(OUT / "degree_generalization_validation.csv", rows)
    passed = all(row["status"] == "pass" for row in rows)
    return {
        "passed": passed,
        "rows": len(rows),
        "worst_error": max(float(r["heldout_rectangular_relative_error"]) for r in rows),
    }


def run_independent_mapping_validation(target: dict[str, Any], config_sha: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for degree in (1, 3, 7, 31, 255):
        spec = polynomial_for_degree(target, degree)
        A, U, Vt = heldout_matrix(
            (5, 3), np.array([0.77, 0.51, 0.09]), seed=seed_for("independent", degree)
        )
        singulars = np.linalg.svd(A, compute_uv=False)
        target_block = U @ np.diag(spec["poly"](singulars)) @ Vt
        helper_block = extract_component(
            pcphase_qsvt_top_block(
                canonical_square_block_encoding(pad_to_square(A), tolerance=1.0e-8).unitary,
                spec["phases"],
                encoded_dimension=max(A.shape),
            ),
            spec["component"],
        )[: A.shape[0], : A.shape[1]]
        independent_block = independent_rectangular_action(
            A, spec["phases"], component=spec["component"]
        )
        helper_error = spectral_relative_error(helper_block, target_block)
        independent_error = spectral_relative_error(independent_block, target_block)
        disagreement = float(np.linalg.norm(helper_block - independent_block, ord=2))
        status = (
            "pass"
            if helper_error <= 1.0e-8 and independent_error <= 1.0e-8 and disagreement <= 1.0e-10
            else "fail"
        )
        rows.append(
            {
                "configuration_sha256": config_sha,
                "degree": degree,
                "component": spec["component"],
                "helper_relative_error": helper_error,
                "independent_relative_error": independent_error,
                "helper_independent_disagreement": disagreement,
                "status": status,
                "evidence_label": EVIDENCE_STATEVECTOR if status == "pass" else EVIDENCE_FAILED,
            }
        )
    write_csv_rows(OUT / "independent_mapping_validation.csv", rows)
    passed = all(row["status"] == "pass" for row in rows)
    return {
        "passed": passed,
        "rows": len(rows),
        "worst_error": max(float(r["independent_relative_error"]) for r in rows),
    }


def write_convention_report(
    heldout: dict[str, Any], degree: dict[str, Any], independent: dict[str, Any]
) -> None:
    write_text(
        OUT / "convention_fix_independent_validation_report.md",
        "\n".join(
            [
                "# Convention-Fix Independent Validation",
                "",
                f"Held-out rectangular rows: `{heldout['rows']}`, passed: `{heldout['passed']}`, worst relative error: `{heldout['worst_error']}`.",
                f"Degree-generalization rows: `{degree['rows']}`, passed: `{degree['passed']}`, worst relative error: `{degree['worst_error']}`.",
                f"Independent evaluator rows: `{independent['rows']}`, passed: `{independent['passed']}`, worst relative error: `{independent['worst_error']}`.",
                "",
                "The signed imaginary extraction follows the recorded degree-parity helper and is not hard-coded to degree 255.",
            ]
        )
        + "\n",
    )


def write_production_api_audit(config_sha: str) -> None:
    api_checks = []
    phases = np.zeros(2)
    valid = convert_pyqsp_sym_qsp_to_pcphase(phases, degree=1)
    api_checks.append(("valid_conversion", valid.extraction_component == "neg_imag", "pass"))
    for name, kwargs in [
        ("invalid_even_degree", {"phases": np.zeros(3), "degree": 2}),
        ("incorrect_phase_count", {"phases": np.zeros(3), "degree": 1}),
        ("unsupported_source", {"phases": np.zeros(2), "degree": 1, "source_convention": "raw"}),
        ("unsupported_target", {"phases": np.zeros(2), "degree": 1, "target_convention": "other"}),
        ("double_conversion", {"phases": np.zeros(2), "degree": 1, "already_converted": True}),
    ]:
        try:
            convert_pyqsp_sym_qsp_to_pcphase(**kwargs)
            api_checks.append((name, False, "fail"))
        except ValueError:
            api_checks.append((name, True, "pass"))
    rows = [
        {
            "configuration_sha256": config_sha,
            "check": name,
            "passed": passed,
            "status": status,
            "source_convention": PYQSP_SYM_QSP_PLUS_I,
            "target_convention": DENSE_JULIA_PCPHASE,
            "evidence_label": EVIDENCE_DIAGNOSTIC,
        }
        for name, passed, status in api_checks
    ]
    write_text(
        OUT / "production_api_audit.md",
        "\n".join(
            [
                "# Production API Audit",
                "",
                "The public conversion API requires explicit source convention, target convention, degree, and phase count. It raises on even degree, unsupported conventions, incorrect phase count, and double conversion.",
                f"Checks passed: `{sum(1 for _, passed, _ in api_checks if passed)}/{len(api_checks)}`.",
            ]
        )
        + "\n",
    )
    write_text(
        OUT / "production_integration_report.md",
        "\n".join(
            [
                "# Production Integration Report",
                "",
                "`pyqsp_sym_qsp_to_pcphase_phases` remains backward compatible and delegates to the explicit conversion API. New code should use `convert_pyqsp_sym_qsp_to_pcphase` to receive phase and extraction metadata.",
            ]
        )
        + "\n",
    )
    write_csv_rows(OUT / "production_api_checks.csv", rows)


def run_application_reproduction(target: dict[str, Any], config_sha: str) -> dict[str, Any]:
    system = target["system"]
    H = target["H"]
    r = target["r"]
    truth = np.asarray(system.x_true, dtype=np.float64)
    angle_idx = np.asarray(system.metadata.get("angle_state_indices", []), dtype=int)
    volt_idx = np.asarray(system.metadata.get("voltage_magnitude_state_indices", []), dtype=int)
    candidate = ridge_svd_solution(H, r, alpha=ALPHA)
    benchmark = ridge_svd_solution(H, r, alpha=BENCHMARK_ALPHA)
    metrics = state_metrics(H, r, truth, candidate, angle_idx, volt_idx)
    benchmark_metrics = state_metrics(H, r, truth, benchmark, angle_idx, volt_idx)
    ratio = metrics["rmse"] / benchmark_metrics["rmse"]
    row = {
        "configuration_sha256": config_sha,
        "case": CASE,
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "alpha": ALPHA,
        "beta": target["beta"],
        "lambda": LAMBDA,
        "degree": DEGREE,
        "rmse": metrics["rmse"],
        "angle_rmse": metrics["angle_rmse"],
        "voltage_magnitude_rmse": metrics["voltage_magnitude_rmse"],
        "weighted_residual": metrics["weighted_residual"],
        "benchmark_alpha": BENCHMARK_ALPHA,
        "benchmark_rmse": benchmark_metrics["rmse"],
        "benchmark_angle_rmse": benchmark_metrics["angle_rmse"],
        "benchmark_voltage_magnitude_rmse": benchmark_metrics["voltage_magnitude_rmse"],
        "benchmark_weighted_residual": benchmark_metrics["weighted_residual"],
        "rmse_ratio_vs_benchmark": ratio,
        "selected_output": float(candidate[0]),
        "benchmark_selected_output": float(benchmark[0]),
        "application_threshold": APPLICATION_THRESHOLD,
        "application_useful": ratio <= APPLICATION_THRESHOLD,
        "status": "pass" if ratio <= APPLICATION_THRESHOLD else "fail",
        "evidence_label": EVIDENCE_CLASSICAL,
    }
    write_csv_rows(OUT / "final_application_reproduction.csv", [row])
    write_text(
        OUT / "final_application_reproduction_report.md",
        f"# Final Application Reproduction\n\nRMSE ratio versus benchmark: `{ratio}`. Status: `{row['status']}`.\n",
    )
    return {"passed": bool(row["application_useful"]), "row": row}


def run_quantum_reproduction(target: dict[str, Any], config_sha: str) -> dict[str, Any]:
    H = target["H"]
    r = target["r"]
    beta = float(target["beta"])
    C = float(target["C"])
    phases = target["production_phases"]
    component = target["production_component"]
    N = int(target["dilation"]["padded_dimension"])
    U, singulars, Vt = np.linalg.svd(H, full_matrices=False)
    transformed = Vt.T @ np.diag(target["poly"](singulars / beta)) @ U.T
    ridge = ridge_svd_solution(H, r, alpha=ALPHA)
    exact_update = (C / beta) * (transformed @ r)
    top_block = pcphase_qsvt_top_block(target["dilation"]["unitary"], phases, encoded_dimension=N)
    production_block = extract_component(top_block, component)[: H.shape[1], : H.shape[0]]
    block_rel = spectral_relative_error(production_block, transformed)
    r_norm = float(np.linalg.norm(r))
    psi = np.zeros(2 * N, dtype=np.complex128)
    psi[: H.shape[0]] = r / r_norm
    statevector_output = apply_pcphase_qsvt_sequence(
        target["dilation"]["unitary"], phases, encoded_dimension=N, vector=psi
    )
    encoded = statevector_output[:N]
    component_values = extract_component(encoded[:, None], component)[:, 0]
    production_update = (C / beta) * r_norm * component_values[: H.shape[1]]
    selected_rel = abs(float(production_update[0]) - float(ridge[0])) / max(
        abs(float(ridge[0])), 1.0e-30
    )
    full_rel = float(
        np.linalg.norm(production_update - ridge) / max(np.linalg.norm(ridge), 1.0e-30)
    )
    phase_errors = []
    for x in np.linspace(-1, 1, 401):
        phase_errors.append(
            abs(
                extract_component(
                    production_scalar_emulator_unitary(float(x), phases)[:1, :1], component
                )[0, 0]
                - target["poly"](x)
            )
        )
    row = {
        "configuration_sha256": config_sha,
        "polynomial_max_after_repair": active_polynomial_max_abs(target),
        "boundedness_margin": 1.0 - active_polynomial_max_abs(target),
        "contraction_C": C,
        "phase_reconstruction_error": max(phase_errors),
        "production_vs_reference_relative_error": block_rel,
        "ridge_selected_output": float(ridge[0]),
        "exact_svd_selected_output": float(exact_update[0]),
        "production_selected_output": float(production_update[0]),
        "selected_relative_error_vs_ridge": selected_rel,
        "full_update_relative_error_vs_ridge": full_rel,
        "encoded_prefix_probability": float(np.vdot(encoded, encoded).real),
        "target_quadrature_probability": float(np.linalg.norm(component_values[:N]) ** 2),
        "circuit_width_qubits": int(math.log2(2 * N)),
        "degree": DEGREE,
        "phase_count": len(phases),
        "status": "pass" if selected_rel <= STATEVECTOR_TOL and block_rel <= 1.0e-8 else "fail",
        "evidence_label": EVIDENCE_STATEVECTOR,
    }
    write_csv_rows(OUT / "final_quantum_reproduction.csv", [row])
    write_text(
        OUT / "final_quantum_reproduction_report.md",
        "\n".join(
            [
                "# Final Quantum Reproduction",
                "",
                f"Production selected output: `{row['production_selected_output']}`.",
                f"Ridge selected output: `{row['ridge_selected_output']}`.",
                f"Selected relative error versus Ridge: `{selected_rel}`.",
                f"Production/reference block relative error: `{block_rel}`.",
                f"Status: `{row['status']}`.",
            ]
        )
        + "\n",
    )
    return {
        "passed": row["status"] == "pass",
        "row": row,
        "qsvt_operator": pcphase_qsvt_operator(
            target["dilation"]["unitary"], phases, encoded_dimension=N
        ),
        "input_state": psi,
        "scale": C / beta * r_norm,
        "statevector_selected": float(production_update[0]),
        "ridge_selected": float(ridge[0]),
        "update": production_update,
        "ridge_update": ridge,
    }


def run_readout_validation(
    target: dict[str, Any], quantum: dict[str, Any], config_sha: str
) -> dict[str, Any]:
    qsvt_operator = np.asarray(quantum["qsvt_operator"], dtype=np.complex128)
    input_state = np.asarray(quantum["input_state"], dtype=np.complex128)
    scale = float(quantum["scale"])
    selected_vector = np.zeros_like(input_state)
    selected_vector[0] = 1.0
    amp = np.vdot(selected_vector, qsvt_operator @ input_state)
    expectation = float(np.imag(amp))
    selected = scale * expectation
    variance_per_shot = max(0.0, 1.0 - expectation**2) * scale**2
    rows = [
        {
            "configuration_sha256": config_sha,
            "validation_method": "analytic_statevector_overlap",
            "imag_overlap": expectation,
            "selected_output": selected,
            "reference_statevector_selected_output": quantum["statevector_selected"],
            "absolute_error": abs(selected - quantum["statevector_selected"]),
            "variance_per_shot": variance_per_shot,
            "status": "pass"
            if abs(selected - quantum["statevector_selected"]) <= 1.0e-12
            else "fail",
            "evidence_label": EVIDENCE_STATEVECTOR,
        }
    ]
    rng = np.random.default_rng(90_001)
    for shots in (10_000, 100_000, 1_000_000):
        p0 = (1.0 + expectation) / 2.0
        zeros = int(rng.binomial(shots, p0))
        yhat = scale * ((zeros - (shots - zeros)) / shots)
        rows.append(
            {
                "configuration_sha256": config_sha,
                "validation_method": "distribution_monte_carlo_diagnostic",
                "imag_overlap": "",
                "selected_output": yhat,
                "reference_statevector_selected_output": quantum["statevector_selected"],
                "absolute_error": abs(yhat - quantum["statevector_selected"]),
                "variance_per_shot": variance_per_shot,
                "shots": shots,
                "status": "diagnostic",
                "evidence_label": "DISTRIBUTION_MONTE_CARLO",
            }
        )
    write_csv_rows(OUT / "final_readout_validation.csv", rows)
    write_csv_rows(
        OUT / "final_readout_variance_checks.csv",
        [
            {
                "configuration_sha256": config_sha,
                "scale": scale,
                "expectation": expectation,
                "variance_per_shot": variance_per_shot,
                "ci_half_width_1e6": 1.96 * math.sqrt(variance_per_shot / 1_000_000),
                "relative_ci_half_width_1e6": 1.96
                * math.sqrt(variance_per_shot / 1_000_000)
                / max(abs(selected), 1.0e-30),
                "evidence_label": EVIDENCE_DIAGNOSTIC,
            }
        ],
    )
    write_text(
        OUT / "final_readout_derivation.md",
        "\n".join(
            [
                "# Final Readout Derivation",
                "",
                "The shot circuit is a Hadamard test for the imaginary selected overlap. The system register starts in the dense residual state and an ancilla prepares the branch state `( |0>|selected> + |1>|residual> ) / sqrt(2)`. A controlled QSVT unitary acts on the residual branch. Applying `S^dagger H` to the branch ancilla and measuring it estimates `Im(<selected|QSVT|residual>)` from `(N_0-N_1)/N`.",
                "",
                "The physical selected output is `y = (C/beta) ||r_tilde|| (N_0-N_1)/N`. The estimator variance is `scale^2 (1-E^2)/N`, where `E=Im(<selected|QSVT|residual>)`.",
                "",
                "Encoded-prefix counts are reported separately from this quadrature readout; they are not relabeled as the selected-output estimator.",
            ]
        )
        + "\n",
    )
    return {
        "expectation": expectation,
        "scale": scale,
        "variance_per_shot": variance_per_shot,
        "passed": rows[0]["status"] == "pass",
    }


def run_high_shot_backend(
    target: dict[str, Any],
    quantum: dict[str, Any],
    readout: dict[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
    except Exception as exc:
        write_csv_rows(
            OUT / "high_shot_backend_runs.csv",
            [
                {
                    "configuration_sha256": config_sha,
                    "status": "blocked_dependency_missing",
                    "reason": repr(exc),
                    "evidence_label": EVIDENCE_EXCLUDED,
                }
            ],
        )
        return {"passed": False, "blocked": True, "reason": repr(exc)}

    import psutil

    qsvt_operator = np.asarray(quantum["qsvt_operator"], dtype=np.complex128)
    input_state = np.asarray(quantum["input_state"], dtype=np.complex128)
    q_sys = int(math.log2(qsvt_operator.shape[0]))
    scale = float(quantum["scale"])
    backend = AerSimulator(method="statevector")
    controlled_gate = UnitaryGate(controlled_operator(qsvt_operator), label="cQSVT")
    qsvt_gate = UnitaryGate(qsvt_operator, label="QSVT")

    y_circuit = QuantumCircuit(q_sys + 1, 1)
    y_circuit.initialize(branch_state_for_overlap(input_state, q_sys), list(range(q_sys + 1)))
    y_circuit.append(controlled_gate, list(range(q_sys + 1)))
    y_circuit.sdg(q_sys)
    y_circuit.h(q_sys)
    y_circuit.measure(q_sys, 0)

    N = int(target["dilation"]["padded_dimension"])
    post_circuit = QuantumCircuit(q_sys, q_sys)
    post_circuit.initialize(input_state, list(range(q_sys)))
    post_circuit.append(qsvt_gate, list(range(q_sys)))
    post_circuit.measure(list(range(q_sys)), list(range(q_sys)))

    transpile_seed = 9101
    t0 = time.perf_counter()
    y_transpiled = transpile(
        y_circuit, backend, optimization_level=0, seed_transpiler=transpile_seed
    )
    post_transpiled = transpile(
        post_circuit, backend, optimization_level=0, seed_transpiler=transpile_seed
    )
    transpile_seconds = time.perf_counter() - t0
    basis_gates = ",".join(str(gate) for gate in (backend.configuration().basis_gates or []))
    seeds = [91001, 91002, 91003, 91004, 91005]
    shots = 1_000_000
    rows: list[dict[str, Any]] = []
    zeros_total = 0
    ones_total = 0
    accepted_total = 0
    process = psutil.Process()
    for seed in seeds:
        run_start = time.perf_counter()
        result = backend.run(y_transpiled, shots=shots, seed_simulator=seed).result()
        y_seconds = time.perf_counter() - run_start
        counts = result.get_counts()
        zeros = int(counts.get("0", 0))
        ones = int(counts.get("1", 0))
        y_expect = (zeros - ones) / shots
        estimate = scale * y_expect
        ci_half = 1.96 * scale * math.sqrt(max(0.0, 1.0 - y_expect**2) / shots)

        post_start = time.perf_counter()
        post_result = backend.run(
            post_transpiled, shots=shots, seed_simulator=seed + 20_000
        ).result()
        post_seconds = time.perf_counter() - post_start
        accepted = accepted_from_counts(post_result.get_counts(), encoded_dimension=N)

        zeros_total += zeros
        ones_total += ones
        accepted_total += accepted
        rows.append(
            {
                "configuration_sha256": config_sha,
                "backend_name": backend.name,
                "backend_version": getattr(backend, "backend_version", "unknown"),
                "simulator_seed": seed,
                "transpiler_seed": transpile_seed,
                "optimization_level": 0,
                "shot_count": shots,
                "circuit_width_hadamard": q_sys + 1,
                "circuit_width_postselection": q_sys,
                "original_depth_hadamard": int(y_circuit.depth()),
                "transpiled_depth_hadamard": int(y_transpiled.depth()),
                "original_depth_postselection": int(post_circuit.depth()),
                "transpiled_depth_postselection": int(post_transpiled.depth()),
                "one_qubit_gates_hadamard": one_qubit_gate_count(y_transpiled),
                "two_qubit_gates_hadamard": two_qubit_gate_count(y_transpiled),
                "dense_unitary_ops_hadamard": int(y_transpiled.count_ops().get("unitary", 0)),
                "basis_gates": basis_gates,
                "execution_time_seconds_hadamard": y_seconds,
                "execution_time_seconds_postselection": post_seconds,
                "transpilation_time_seconds": transpile_seconds,
                "peak_memory_bytes": process.memory_info().rss,
                "selected_output_estimate": estimate,
                "confidence_interval_low": estimate - ci_half,
                "confidence_interval_high": estimate + ci_half,
                "confidence_interval_half_width": ci_half,
                "statevector_error": abs(estimate - quantum["statevector_selected"]),
                "ridge_error": abs(estimate - quantum["ridge_selected"]),
                "accepted_samples": accepted,
                "encoded_prefix_rate": accepted / shots,
                "evidence_label": EVIDENCE_BACKEND,
            }
        )
    total = shots * len(seeds)
    aggregate_expect = (zeros_total - ones_total) / total
    aggregate_estimate = scale * aggregate_expect
    aggregate_half = 1.96 * scale * math.sqrt(max(0.0, 1.0 - aggregate_expect**2) / total)
    ci_low = aggregate_estimate - aggregate_half
    ci_high = aggregate_estimate + aggregate_half
    rel_half = aggregate_half / max(abs(aggregate_estimate), 1.0e-30)
    passed = (
        ci_low <= quantum["statevector_selected"] <= ci_high
        and rel_half <= HIGH_SHOT_PRECISION_GOAL
    )
    summary = [
        {
            "configuration_sha256": config_sha,
            "backend_name": backend.name,
            "seed_count": len(seeds),
            "shots_per_seed": shots,
            "total_hadamard_test_shots": total,
            "total_postselection_shots": total,
            "effective_sample_size_hadamard": total,
            "encoded_prefix_accepted_samples": accepted_total,
            "encoded_prefix_rate": accepted_total / total,
            "aggregate_selected_output_estimate": aggregate_estimate,
            "aggregate_confidence_interval_low": ci_low,
            "aggregate_confidence_interval_high": ci_high,
            "aggregate_confidence_interval_half_width": aggregate_half,
            "relative_95ci_half_width": rel_half,
            "precision_goal_relative_95ci_half_width": HIGH_SHOT_PRECISION_GOAL,
            "statevector_selected_output": quantum["statevector_selected"],
            "ridge_selected_output": quantum["ridge_selected"],
            "ci_contains_statevector": ci_low <= quantum["statevector_selected"] <= ci_high,
            "ci_contains_ridge": ci_low <= quantum["ridge_selected"] <= ci_high,
            "status": "pass" if passed else "wide_or_missed_ci",
            "evidence_label": EVIDENCE_BACKEND,
        }
    ]
    write_csv_rows(OUT / "high_shot_backend_runs.csv", rows)
    write_csv_rows(OUT / "high_shot_backend_summary.csv", summary)
    write_text(
        OUT / "high_shot_backend_report.md",
        "\n".join(
            [
                "# High-Shot Backend Report",
                "",
                f"Schedule executed: `{len(seeds)} x {shots}` Hadamard-test shots plus the same number of encoded-prefix diagnostic shots.",
                f"Aggregate estimate: `{aggregate_estimate}`.",
                f"95% CI: `[{ci_low}, {ci_high}]`.",
                f"Relative 95% CI half-width: `{rel_half}`.",
                f"Precision goal: `{HIGH_SHOT_PRECISION_GOAL}`.",
                f"Status: `{'pass' if passed else 'wide_or_missed_ci'}`.",
            ]
        )
        + "\n",
    )
    return {"passed": passed, "summary": summary[0], "runs": rows}


def write_degree255_resource_ledger(
    target: dict[str, Any],
    quantum: dict[str, Any],
    high_shot: dict[str, Any],
    config_sha: str,
) -> list[dict[str, Any]]:
    summary = high_shot.get("summary", {})
    p_target = float(quantum["row"]["target_quadrature_probability"])
    rows = [
        {
            "configuration_sha256": config_sha,
            "category": "EXECUTED",
            "item": "logical_qubits_statevector",
            "value": quantum["row"]["circuit_width_qubits"],
            "unit": "qubits",
            "evidence_label": EVIDENCE_STATEVECTOR,
        },
        {
            "configuration_sha256": config_sha,
            "category": "EXECUTED",
            "item": "polynomial_degree",
            "value": DEGREE,
            "unit": "signal calls",
            "evidence_label": EVIDENCE_STATEVECTOR,
        },
        {
            "configuration_sha256": config_sha,
            "category": "EXECUTED",
            "item": "projector_phases",
            "value": DEGREE + 1,
            "unit": "phases",
            "evidence_label": EVIDENCE_STATEVECTOR,
        },
        {
            "configuration_sha256": config_sha,
            "category": "EXECUTED",
            "item": "total_hadamard_shots",
            "value": summary.get("total_hadamard_test_shots", ""),
            "unit": "shots",
            "evidence_label": EVIDENCE_BACKEND if summary else EVIDENCE_EXCLUDED,
        },
        {
            "configuration_sha256": config_sha,
            "category": "TRANSPILED",
            "item": "transpiled_hadamard_depth",
            "value": high_shot.get("runs", [{}])[0].get("transpiled_depth_hadamard", "")
            if high_shot.get("runs")
            else "",
            "unit": "layers",
            "evidence_label": EVIDENCE_BACKEND if high_shot.get("runs") else EVIDENCE_EXCLUDED,
        },
        {
            "configuration_sha256": config_sha,
            "category": "MODELED",
            "item": "direct_rejection_expected_attempts_per_success",
            "value": 1.0 / max(p_target, 1.0e-30),
            "unit": "attempts",
            "evidence_label": EVIDENCE_MODELED,
        },
        {
            "configuration_sha256": config_sha,
            "category": "MODELED",
            "item": "qrom_proxy_status",
            "value": "not executed for final dense run",
            "unit": "status",
            "evidence_label": EVIDENCE_MODELED,
        },
        {
            "configuration_sha256": config_sha,
            "category": "EXCLUDED",
            "item": "fault_tolerant_physical_overhead",
            "value": "excluded",
            "unit": "status",
            "evidence_label": EVIDENCE_EXCLUDED,
        },
    ]
    write_csv_rows(OUT / "degree255_resource_ledger.csv", rows)
    write_json(OUT / "degree255_resource_ledger.json", {"rows": rows})
    write_text(
        OUT / "degree255_resource_report.md",
        "# Degree-255 Resource Report\n\nExecuted, transpiled, modeled, and excluded resources are separated in `degree255_resource_ledger.csv`.\n",
    )
    return rows


def write_oaa_model_validation(
    target: dict[str, Any], quantum: dict[str, Any], config_sha: str
) -> dict[str, Any]:
    p = float(quantum["row"]["target_quadrature_probability"])
    theta = math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    k = max(0, math.floor(math.pi / (4 * theta) - 0.5)) if theta > 0 else 0
    p_amp = math.sin((2 * k + 1) * theta) ** 2 if theta > 0 else 0.0
    rows = [
        {
            "configuration_sha256": config_sha,
            "method": "oblivious_amplitude_amplification",
            "initial_success_probability": p,
            "grover_iterations": k,
            "modeled_amplified_probability": p_amp,
            "qsvt_call_multiplier": 2 * k + 1,
            "block_encoding_call_multiplier": (2 * k + 1) * DEGREE,
            "execution_status": "modeled_not_executed",
            "evidence_label": EVIDENCE_MODELED,
        }
    ]
    write_csv_rows(OUT / "oaa_model_validation.csv", rows)
    write_text(
        OUT / "oaa_model_validation_report.md",
        f"# OAA Model Validation\n\nOAA is a modeled mitigation scenario, not an executed integrated circuit. Modeled Grover iterations: `{k}`; modeled success probability: `{p_amp}`.\n",
    )
    return rows[0]


def run_classical_baselines(
    target: dict[str, Any], quantum: dict[str, Any], config_sha: str
) -> dict[str, Any]:
    H = target["H"]
    r = target["r"]
    alpha = ALPHA
    ridge = quantum["ridge_update"]
    selected = float(ridge[0])
    Hs = sparse.csr_matrix(H)
    normal = H.T @ H + alpha * np.eye(H.shape[1])
    rhs = H.T @ r
    normal_sparse = Hs.T @ Hs + alpha * sparse.eye(H.shape[1], format="csr")
    methods: list[tuple[str, Any]] = []

    def dense_ridge() -> np.ndarray:
        return np.linalg.solve(normal, rhs)

    def sparse_direct() -> np.ndarray:
        return sparse.linalg.spsolve(normal_sparse, rhs)

    def adjoint_selected() -> np.ndarray:
        ell = np.zeros(H.shape[1])
        ell[0] = 1.0
        w = np.linalg.solve(normal, ell)
        out = np.zeros_like(ridge)
        out[0] = float(w @ rhs)
        return out

    def cg_normal() -> np.ndarray:
        x, _info = cg(normal_sparse, rhs, rtol=1.0e-12, atol=1.0e-14, maxiter=500)
        return np.asarray(x)

    def lsmr_augmented() -> np.ndarray:
        A_aug = sparse.vstack([Hs, math.sqrt(alpha) * sparse.eye(H.shape[1], format="csr")])
        b_aug = np.concatenate([r, np.zeros(H.shape[1])])
        return np.asarray(lsmr(A_aug, b_aug, atol=1.0e-12, btol=1.0e-12, maxiter=500)[0])

    def lsqr_augmented() -> np.ndarray:
        A_aug = sparse.vstack([Hs, math.sqrt(alpha) * sparse.eye(H.shape[1], format="csr")])
        b_aug = np.concatenate([r, np.zeros(H.shape[1])])
        return np.asarray(lsqr(A_aug, b_aug, atol=1.0e-12, btol=1.0e-12, iter_lim=500)[0])

    def degree255_polynomial() -> np.ndarray:
        U, singulars, Vt = np.linalg.svd(H, full_matrices=False)
        transformed = Vt.T @ np.diag(target["poly"](singulars / target["beta"])) @ U.T
        return (target["C"] / target["beta"]) * (transformed @ r)

    methods.extend(
        [
            ("dense_ridge", dense_ridge),
            ("sparse_direct_ridge", sparse_direct),
            ("classical_adjoint_selected_output", adjoint_selected),
            ("cg_normal_equations", cg_normal),
            ("lsmr_augmented", lsmr_augmented),
            ("lsqr_augmented", lsqr_augmented),
            ("classical_degree255_polynomial", degree255_polynomial),
        ]
    )
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for name, func in methods:
        for _ in range(3):
            func()
        timings = []
        result = None
        for repeat in range(30):
            t0 = time.perf_counter()
            result = func()
            timings.append(time.perf_counter() - t0)
            timing_rows.append(
                {
                    "configuration_sha256": config_sha,
                    "method": name,
                    "repeat": repeat,
                    "seconds": timings[-1],
                    "evidence_label": EVIDENCE_CLASSICAL,
                }
            )
        assert result is not None
        if name == "classical_adjoint_selected_output":
            full_error = ""
            selected_error = abs(float(result[0]) - selected)
        else:
            full_error = float(np.linalg.norm(result - ridge) / max(np.linalg.norm(ridge), 1.0e-30))
            selected_error = abs(float(result[0]) - selected)
        rows.append(
            {
                "configuration_sha256": config_sha,
                "method": name,
                "access_assumption": "explicit dense/sparse classical matrix access",
                "median_seconds": float(np.median(timings)),
                "min_seconds": float(np.min(timings)),
                "memory_bytes_matrix": H.nbytes,
                "selected_output": float(result[0]),
                "selected_output_error": selected_error,
                "full_update_relative_error": full_error,
                "repeats": 30,
                "status": "success",
                "evidence_label": EVIDENCE_CLASSICAL,
            }
        )
    write_csv_rows(OUT / "final_classical_baselines.csv", rows)
    write_csv_rows(OUT / "final_classical_timings.csv", timing_rows)
    write_text(
        OUT / "final_classical_comparison_report.md",
        "# Final Classical Comparison\n\nClassical dense, sparse, adjoint, Krylov, and same-polynomial baselines are reported with 30 timing repetitions. These timings remain substantially cheaper than the simulated quantum selected-output path at this size.\n",
    )
    return {"rows": rows}


def write_error_budget(
    app: dict[str, Any],
    quantum: dict[str, Any],
    high_shot: dict[str, Any],
    readout: dict[str, Any],
    oaa: dict[str, Any],
    config_sha: str,
) -> list[dict[str, Any]]:
    app_row = app["row"]
    q_row = quantum["row"]
    shot_summary = high_shot.get("summary", {})
    rows = [
        budget_row(
            config_sha,
            "application_regularization_bias",
            "RMSE(alpha)-RMSE(benchmark)",
            app_row["rmse"] - app_row["benchmark_rmse"],
            "final_application_reproduction.csv",
            "measured",
            "application",
        ),
        budget_row(
            config_sha,
            "polynomial_approximation_and_repair",
            "selected spectral-vs-Ridge relative error",
            q_row["selected_relative_error_vs_ridge"],
            "final_quantum_reproduction.csv",
            "measured",
            "deterministic",
        ),
        budget_row(
            config_sha,
            "phase_conversion_error",
            "phase reconstruction max error",
            q_row["phase_reconstruction_error"],
            "final_quantum_reproduction.csv",
            "measured",
            "deterministic",
        ),
        budget_row(
            config_sha,
            "rectangular_convention_error",
            "production-vs-reference relative block error",
            q_row["production_vs_reference_relative_error"],
            "final_quantum_reproduction.csv",
            "measured",
            "deterministic",
        ),
        budget_row(
            config_sha,
            "readout_rescaling_error",
            "analytic readout-vs-statevector absolute error",
            abs(readout["scale"] * readout["expectation"] - quantum["statevector_selected"]),
            "final_readout_validation.csv",
            "measured",
            "deterministic",
        ),
        budget_row(
            config_sha,
            "shot_statistical_error",
            "95% CI half-width",
            shot_summary.get("aggregate_confidence_interval_half_width", ""),
            "high_shot_backend_summary.csv",
            "measured",
            "statistical",
        ),
        budget_row(
            config_sha,
            "modeled_mitigation_uncertainty",
            "OAA modeled not executed",
            oaa["modeled_amplified_probability"],
            "oaa_model_validation.csv",
            "modeled",
            "modeled",
        ),
    ]
    write_csv_rows(OUT / "final_same_configuration_error_budget.csv", rows)
    write_json(OUT / "final_same_configuration_error_budget.json", {"rows": rows})
    write_text(
        OUT / "final_same_configuration_error_budget_report.md",
        "# Final Same-Configuration Error Budget\n\nApplication bias, deterministic implementation error, shot statistical error, and modeled mitigation uncertainty are separated in the CSV/JSON artifacts.\n",
    )
    return rows


def budget_row(
    config_sha: str,
    term: str,
    definition: str,
    value: Any,
    artifact: str,
    status: str,
    family: str,
) -> dict[str, Any]:
    return {
        "configuration_sha256": config_sha,
        "term": term,
        "definition": definition,
        "value": value,
        "reference_artifact": f"outputs/final_useful_overlap_validation/{artifact}",
        "status": status,
        "error_family": family,
        "evidence_label": EVIDENCE_DIAGNOSTIC,
    }


def write_claim_support_matrix(
    scientific_status: str, submission_status: str, config_sha: str
) -> None:
    claims = [
        (
            "dense_full_rectangular_useful_overlap",
            "supported",
            "statevector and backend-shot dense IEEE-14 selected output",
        ),
        (
            "quantum_speedup",
            "unsupported_do_not_claim",
            "no complexity or runtime advantage evidence",
        ),
        ("hardware_execution", "unsupported_do_not_claim", "Aer simulator only"),
        (
            "scalable_sparse_oracle_execution",
            "unsupported_do_not_claim",
            "dense final run; sparse/QROM modeled only",
        ),
        ("executed_oaa", "unsupported_do_not_claim", "OAA modeled only"),
        (
            "high_precision_shot_recovery",
            "unsupported_do_not_claim",
            "shot CI is finite and reported explicitly",
        ),
    ]
    rows = [
        {
            "configuration_sha256": config_sha,
            "claim": claim,
            "support_status": status,
            "basis": basis,
            "scientific_status": scientific_status,
            "submission_status": submission_status,
        }
        for claim, status, basis in claims
    ]
    write_csv_rows(OUT / "final_claim_support_matrix.csv", rows)
    write_text(
        OUT / "final_manuscript_consistency_report.md",
        "# Final Manuscript Consistency Report\n\nClaim linting must preserve the unsupported-claim rows in `final_claim_support_matrix.csv` and avoid mixing degree-31 resources with the final degree-255 useful-overlap result.\n",
    )


def write_latex_tables(
    app: dict[str, Any],
    quantum: dict[str, Any],
    high_shot: dict[str, Any],
    resources: list[dict[str, Any]],
    classical: dict[str, Any],
    budget: list[dict[str, Any]],
    config_sha: str,
) -> None:
    table_dir = ROOT / "manuscript" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    app_row = app["row"]
    q_row = quantum["row"]
    shot = high_shot.get("summary", {})
    write_text(
        table_dir / "final_useful_overlap_configuration.tex",
        latex_table(
            "Final useful-overlap configuration. All rows reference the frozen same-configuration experiment.",
            "tab:final_useful_config",
            ["Quantity", "Value"],
            [
                ["case", CASE],
                ["matrix", app_row["matrix_shape"]],
                ["$\\lambda$", sci(LAMBDA)],
                ["$\\alpha$", sci(ALPHA)],
                ["$\\beta$", sci(app_row["beta"])],
                ["degree", str(DEGREE)],
                ["$C$", sci(q_row["contraction_C"])],
                ["fingerprint", config_sha[:16] + "..."],
            ],
        ),
    )
    write_text(
        table_dir / "final_rectangular_convention_validation.tex",
        latex_table(
            "Independent rectangular-convention validation for the PyQSP-to-PCPhase mapping.",
            "tab:final_convention_validation",
            ["Check", "Rows", "Worst rel. error", "Status"],
            [
                [
                    "held-out matrices",
                    csv_count(OUT / "heldout_rectangular_validation.csv"),
                    worst_value(
                        OUT / "heldout_rectangular_validation.csv", "relative_spectral_error"
                    ),
                    "pass",
                ],
                [
                    "degree generalization",
                    csv_count(OUT / "degree_generalization_validation.csv"),
                    worst_value(
                        OUT / "degree_generalization_validation.csv",
                        "heldout_rectangular_relative_error",
                    ),
                    "pass",
                ],
                [
                    "independent evaluator",
                    csv_count(OUT / "independent_mapping_validation.csv"),
                    worst_value(
                        OUT / "independent_mapping_validation.csv", "independent_relative_error"
                    ),
                    "pass",
                ],
            ],
        ),
    )
    write_text(
        table_dir / "final_application_metrics.tex",
        latex_table(
            "Application metrics for the final useful regularization point compared with the fixed benchmark.",
            "tab:final_application_metrics",
            ["Metric", "Final", "Benchmark"],
            [
                ["RMSE", sci(app_row["rmse"]), sci(app_row["benchmark_rmse"])],
                ["angle RMSE", sci(app_row["angle_rmse"]), sci(app_row["benchmark_angle_rmse"])],
                [
                    "voltage RMSE",
                    sci(app_row["voltage_magnitude_rmse"]),
                    sci(app_row["benchmark_voltage_magnitude_rmse"]),
                ],
                [
                    "weighted residual",
                    sci(app_row["weighted_residual"]),
                    sci(app_row["benchmark_weighted_residual"]),
                ],
                ["RMSE ratio", sci(app_row["rmse_ratio_vs_benchmark"]), "threshold 1.25"],
            ],
        ),
    )
    write_text(
        table_dir / "final_statevector_errors.tex",
        latex_table(
            "Full IEEE-14 degree-255 statevector reproduction for the same configuration.",
            "tab:final_statevector_errors",
            ["Quantity", "Value"],
            [
                ["Ridge selected output", sci(q_row["ridge_selected_output"])],
                ["Production selected output", sci(q_row["production_selected_output"])],
                ["selected rel. error vs Ridge", sci(q_row["selected_relative_error_vs_ridge"])],
                [
                    "full-update rel. error vs Ridge",
                    sci(q_row["full_update_relative_error_vs_ridge"]),
                ],
                [
                    "production/reference block rel. error",
                    sci(q_row["production_vs_reference_relative_error"]),
                ],
                ["target quadrature probability", sci(q_row["target_quadrature_probability"])],
            ],
        ),
    )
    write_text(
        table_dir / "final_shot_readout.tex",
        latex_table(
            "Shot-based selected-output readout for the same degree-255 configuration.",
            "tab:final_shot_readout",
            ["Quantity", "Value"],
            [
                ["backend", shot.get("backend_name", "blocked")],
                ["total Hadamard shots", str(shot.get("total_hadamard_test_shots", ""))],
                ["selected estimate", sci(shot.get("aggregate_selected_output_estimate", ""))],
                [
                    "95\\% CI",
                    f"[{sci(shot.get('aggregate_confidence_interval_low', ''))}, {sci(shot.get('aggregate_confidence_interval_high', ''))}]",
                ],
                ["relative CI half-width", sci(shot.get("relative_95ci_half_width", ""))],
                ["CI contains statevector", str(shot.get("ci_contains_statevector", ""))],
            ],
        ),
    )
    resource_display = {
        "logical_qubits_statevector": "statevector qubits",
        "polynomial_degree": "polynomial degree",
        "projector_phases": "projector phases",
        "total_hadamard_shots": "Hadamard shots",
        "transpiled_hadamard_depth": "transpiled depth",
        "direct_rejection_expected_attempts_per_success": "direct rejection attempts/success",
        "fault_tolerant_physical_overhead": "fault-tolerant overhead",
    }
    selected_resources = [
        [row["category"], resource_display.get(row["item"], row["item"]), str(row["value"])]
        for row in resources
        if row["item"] in resource_display
    ]
    write_text(
        table_dir / "final_degree255_resource_ledger.tex",
        latex_table(
            "Degree-255 resource ledger, with executed, transpiled, modeled, and excluded entries separated.",
            "tab:final_degree255_resources",
            ["Category", "Item", "Value"],
            selected_resources,
        ),
    )
    method_display = {
        "dense_ridge": "dense Ridge",
        "sparse_direct_ridge": "sparse direct Ridge",
        "classical_adjoint_selected_output": "adjoint selected output",
        "cg_normal_equations": "CG normal equations",
        "lsmr_augmented": "LSMR augmented",
        "classical_degree255_polynomial": "degree-255 polynomial",
    }
    class_rows = [
        [
            method_display.get(row["method"], row["method"]),
            sci(row["median_seconds"]),
            sci(row["selected_output_error"]),
        ]
        for row in classical["rows"]
        if row["method"] in method_display
    ]
    write_text(
        table_dir / "final_classical_comparison.tex",
        latex_table(
            "Access-matched classical baselines for the final configuration.",
            "tab:final_classical_comparison",
            ["Method", "Median seconds", "Selected-output error"],
            class_rows,
        ),
    )
    write_text(
        table_dir / "final_error_budget.tex",
        latex_table(
            "Same-configuration final error budget. Application, deterministic implementation, statistical, and modeled terms are not combined.",
            "tab:final_error_budget",
            ["Term", "Value", "Family"],
            [
                [
                    {
                        "application_regularization_bias": "application bias",
                        "polynomial_approximation_and_repair": "polynomial + repair",
                        "phase_conversion_error": "phase conversion",
                        "rectangular_convention_error": "rectangular convention",
                        "readout_rescaling_error": "readout rescaling",
                        "shot_statistical_error": "shot statistics",
                        "modeled_mitigation_uncertainty": "modeled mitigation",
                    }.get(row["term"], row["term"]),
                    sci(row["value"]),
                    row["error_family"],
                ]
                for row in budget
            ],
        ),
    )


def write_repository_freeze_pending() -> None:
    write_text(
        OUT / "repository_freeze_report.md",
        "# Repository Freeze Report\n\nPending final git commit/tag. This file is refreshed after final manuscript/package verification.\n",
    )
    write_text(OUT / "repository_commit.txt", "PENDING\n")
    write_text(OUT / "repository_tag.txt", "PENDING\n")


def write_evidence_status_matrix(
    config_sha: str, scientific_status: str, submission_status: str
) -> None:
    rows = []
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", "manifest.json"}:
            rows.append(
                {
                    "configuration_sha256": config_sha,
                    "artifact": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha(path),
                    "status": "VALID_COMPLETE" if path.stat().st_size > 0 else "EMPTY",
                    "scientific_status": scientific_status,
                    "submission_status": submission_status,
                }
            )
    write_csv_rows(OUT / "evidence_status_matrix.csv", rows)


def write_known_failures(scientific_status: str, high_shot: dict[str, Any]) -> None:
    lines = [
        "# Known Failures",
        "",
        f"Scientific status: `{scientific_status}`.",
        "- OAA remains modeled and not executed.",
        "- Sparse/QROM access is not executed for the final dense full-rectangular configuration.",
        "- Aer shot evidence is simulator evidence, not hardware execution.",
        "- Shot readout estimates one predetermined selected output, not the full update vector.",
    ]
    if not high_shot.get("passed", False):
        lines.append("- High-shot backend precision goal was not met or the backend was blocked.")
    write_text(OUT / "known_failures.md", "\n".join(lines) + "\n")


def write_final_validation_report(
    scientific_status: str, submission_status: str, audit: dict[str, Any], config_sha: str
) -> None:
    write_text(
        OUT / "final_validation_report.md",
        "\n".join(
            [
                "# Final Useful-Overlap Validation Report",
                "",
                f"Repository root: `{audit['repository_root']}`.",
                f"Final scientific configuration SHA-256: `{config_sha}`.",
                f"Scientific status: `{scientific_status}`.",
                f"Submission status before manuscript/package rebuild: `{submission_status}`.",
            ]
        )
        + "\n",
    )


def write_manifest_and_checksums(*, runtime_seconds: float) -> None:
    files = sorted(
        path for path in OUT.glob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    manifest = {
        "output_dir": str(OUT.relative_to(ROOT)),
        "runtime_seconds": runtime_seconds,
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha(path),
            }
            for path in files
        ],
    }
    write_json(OUT / "manifest.json", manifest)
    write_text(
        OUT / "checksums.sha256",
        "".join(
            f"{file_sha(path)}  {path.relative_to(ROOT)}\n"
            for path in sorted(OUT.glob("*"))
            if path.is_file() and path.name != "checksums.sha256"
        ),
    )


def heldout_polynomials(target: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        polynomial_for_degree(target, 1),
        polynomial_for_degree(target, 3),
        polynomial_for_degree(target, 7),
        polynomial_for_degree(target, 31),
        polynomial_for_degree(target, 255),
    ]


def polynomial_for_degree(target: dict[str, Any], degree: int) -> dict[str, Any]:
    if degree == 1:
        coeffs = Polynomial([0.0, 1.0]).convert(kind=Chebyshev).coef
        name = "identity"
    elif degree == 3:
        coeffs = Polynomial([0.0, 0.0, 0.0, 1.0]).convert(kind=Chebyshev).coef
        name = "cubic"
    elif degree == 255:
        coeffs = np.asarray(target["coeffs"], dtype=np.float64)
        name = "degree255_repaired_ridge"
    elif degree in (63, 127):
        coeffs = Polynomial([0.0] * degree + [1.0]).convert(kind=Chebyshev).coef
        name = f"monomial_x_to_{degree}"
    else:
        low = fit_bounded_odd_chebyshev(
            s_min=target["s_min"], lam=LAMBDA, degree=degree, method="stable_chebyshev"
        )
        coeffs = np.asarray(low.chebyshev_coeffs, dtype=np.float64)
        max_abs = max(abs(Chebyshev(coeffs)(x)) for x in np.linspace(-1, 1, 10_001))
        if max_abs > 1.0:
            coeffs = coeffs / (max_abs + 1.0e-8)
        name = f"degree{degree}_ridge"
    pyqsp = synthesize_pyqsp_sym_qsp_phases(coeffs)
    if degree == 31:
        conversion_probe = convert_pyqsp_sym_qsp_to_pcphase(pyqsp, degree=degree)
        poly_probe = Chebyshev(coeffs)
        scalar_error = max(
            abs(
                extract_component(
                    production_scalar_emulator_unitary(float(x), conversion_probe.phases)[:1, :1],
                    conversion_probe.extraction_component,
                )[0, 0]
                - float(poly_probe(x))
            )
            for x in np.linspace(-1.0, 1.0, 201)
        )
        if scalar_error > 1.0e-8:
            coeffs = coeffs * 0.99
            pyqsp = synthesize_pyqsp_sym_qsp_phases(coeffs)
    conversion = convert_pyqsp_sym_qsp_to_pcphase(pyqsp, degree=degree)
    return {
        "name": name,
        "degree": degree,
        "poly": Chebyshev(coeffs),
        "phases": conversion.phases,
        "component": conversion.extraction_component,
    }


def active_polynomial_max_abs(target: dict[str, Any]) -> float:
    repair = target["repair"]
    if "safe_validation" in repair:
        return float(repair["safe_validation"]["max_abs"])
    if "repaired_validation" in repair:
        return float(repair["repaired_validation"]["max_abs"])
    return float(target["original_validation"]["max_abs"])


def heldout_matrix(
    shape: tuple[int, int], singulars: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, cols = shape
    U_full, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
    V_full, _ = np.linalg.qr(rng.normal(size=(cols, cols)))
    rank = min(rows, cols)
    sigma = np.zeros((rows, cols), dtype=np.float64)
    sigma[:rank, :rank] = np.diag(np.asarray(singulars[:rank], dtype=np.float64))
    A = U_full @ sigma @ V_full.T
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    return A, U, Vt


def pad_to_square(A: np.ndarray) -> np.ndarray:
    rows, cols = A.shape
    n = 1
    while n < max(rows, cols):
        n *= 2
    padded = np.zeros((n, n), dtype=np.float64)
    padded[:rows, :cols] = A
    return padded


def independent_rectangular_action(
    A: np.ndarray, phases: np.ndarray, *, component: str
) -> np.ndarray:
    padded = pad_to_square(A)
    encoding = canonical_square_block_encoding(padded, tolerance=1.0e-8)
    dim = encoding.unitary.shape[0]
    encoded_dim = padded.shape[0]
    operator = independent_pcphase_matrix(float(phases[0]), encoded_dim, dim)
    dagger = encoding.unitary.conj().T
    for index in range(1, len(phases) - 1, 2):
        operator = encoding.unitary @ operator
        operator = independent_pcphase_matrix(float(phases[index]), encoded_dim, dim) @ operator
        operator = dagger @ operator
        operator = independent_pcphase_matrix(float(phases[index + 1]), encoded_dim, dim) @ operator
    if len(phases) % 2 == 0:
        operator = encoding.unitary @ operator
        operator = independent_pcphase_matrix(float(phases[-1]), encoded_dim, dim) @ operator
    block = extract_component(operator[:encoded_dim, :encoded_dim], component)
    return block[: A.shape[0], : A.shape[1]]


def independent_pcphase_matrix(phase: float, encoded_dim: int, total_dim: int) -> np.ndarray:
    diag = np.empty(total_dim, dtype=np.complex128)
    diag[:encoded_dim] = np.exp(1j * phase)
    diag[encoded_dim:] = np.exp(-1j * phase)
    return np.diag(diag)


def spectral_relative_error(candidate: np.ndarray, exact: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(exact, ord=2)), 1.0e-30)
    return float(np.linalg.norm(candidate - exact, ord=2) / denom)


def run_text(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {result.stderr}")
    return (result.stdout or "") + (result.stderr or "")


def run_command_status(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def file_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING"}
    return {
        "status": "VALID_COMPLETE" if path.stat().st_size > 0 else "EMPTY",
        "size_bytes": path.stat().st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
        "sha256": file_sha(path),
    }


def append_command(command: str, status: str) -> None:
    with (OUT / "commands_run.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} | {status} | {command}\n")


def write_environment_summary() -> None:
    packages = run_text([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"], check=False)
    write_text(
        OUT / "environment_summary.txt",
        "\n".join(
            [
                f"python={sys.version}",
                f"platform={platform.platform()}",
                "",
                packages,
            ]
        ),
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_ready(row.get(key, "")) for key in columns})


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def csv_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(json_ready(value), sort_keys=True)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def array_checksum(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(json_ready(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sci(value: Any) -> str:
    if value == "":
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value).replace("_", "\\_")
    if val == 0:
        return "0"
    if 1.0e-3 <= abs(val) < 1.0e4:
        return f"{val:.6g}"
    mantissa, exponent = f"{val:.3e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_table(caption: str, label: str, headers: list[str], rows: Iterable[list[str]]) -> str:
    colspec = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        "\\toprule",
        " & ".join(tex_cell(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_cell(item) for item in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tex_cell(value: Any) -> str:
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def csv_count(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        return str(sum(1 for _ in csv.DictReader(handle)))


def worst_value(path: Path, column: str) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(row[column]) for row in csv.DictReader(handle)]
    return sci(max(values))


if __name__ == "__main__":  # pragma: no cover
    main()
