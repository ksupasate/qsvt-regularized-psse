"""Phase 2 harder selected-block QSVT implementation-boundary evidence.

The experiment reuses the selected-output pipeline used by the manuscript's
4x4 and 8x8 correctness anchors.  It adds a raw IEEE-30-derived 16x16 block and
condition-controlled variants of the deterministic IEEE-14-derived 8x8 block.
The variants preserve the selected singular vectors and residual and replace
only the singular-value schedule; they are stress constructions, not raw IEEE
measurement blocks.

Ridge/Tikhonov at the same alpha is always the reference.  The script records
successful and failed rows and makes no speedup, numerical-superiority, full
IEEE-scale execution, sparse-block-encoding, or nonlinear-loop claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.circuit_signed_readout import circuit_signed_readout_rows
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    array_checksum,
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    BlockDemoResult,
    _state_labels_for_cols,
    run_demo_for_block,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_CONFIG = Path("configs/qsvt_phase2_boundary.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/phase2_qsvt_boundary")
BLOCK_SELECTION_RULE = "largest_row_col_norms (deterministic, pre-solve)"
ADJOINT_TIMING_REPEATS = 30
BOUND_TOLERANCE = 2.0e-3
ISOLATED_READOUT_STATUS = "isolated_overlap_assumed_output_state_preparation"

FINAL_STATUSES = {
    "feasible",
    "degree-limited",
    "tolerance-missing",
    "phase-synthesis-failed",
    "statevector-failed",
    "readout-failed",
    "skipped-with-reason",
}

REQUIRED_COLUMNS = [
    "run_id",
    "case_name",
    "block_size",
    "matrix_source",
    "kappa",
    "sigma_min",
    "sigma_max",
    "beta",
    "lambda",
    "alpha",
    "degree_attempted",
    "degree_min_feasible",
    "phase_count",
    "target_error_tolerance",
    "max_polynomial_error",
    "spectrum_point_error",
    "uniform_grid_error",
    "boundedness_pass",
    "parity_pass",
    "phase_synthesis_status",
    "phase_synthesis_error",
    "qsvt_statevector_status",
    "postselection_probability",
    "update_relative_error_vs_matched_ridge",
    "selected_functional_errors",
    "finite_shot_readout_status",
    "finite_shot_shots",
    "finite_shot_mean_relative_error",
    "classical_adjoint_value",
    "classical_adjoint_time",
    "final_status",
    "failure_reason",
]

EXTRA_COLUMNS = [
    "block_id",
    "block_kind",
    "degree_ceiling",
    "selected_rows",
    "selected_cols",
    "block_checksum",
    "residual_checksum",
    "selection_rule",
    "measurement_state_provenance",
    "singular_values",
    "controlled_target_kappa",
    "polynomial_degree_evaluated",
    "physical_recovery_factor_C_over_beta",
    "classical_adjoint_observable",
]


@dataclass(slots=True)
class SelectedBlock:
    block_id: str
    case_name: str
    matrix_source: str
    block_kind: str
    H: np.ndarray
    r: np.ndarray
    selected_rows: np.ndarray
    selected_cols: np.ndarray
    column_labels: list[dict[str, Any]]
    provenance: str
    controlled_target_kappa: float | None = None

    @property
    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.H, compute_uv=False)


@dataclass(frozen=True, slots=True)
class PolynomialDiagnostic:
    degree: int
    uniform_grid_error: float
    spectrum_point_error: float
    max_polynomial_error: float
    boundedness_pass: bool
    parity_pass: bool
    bounded_max_abs: float
    bound_C: float

    @property
    def meets(self) -> bool:
        return self.boundedness_pass and self.parity_pass and np.isfinite(self.max_polynomial_error)


def condition_controlled_variant(H: np.ndarray, target_kappa: float) -> np.ndarray:
    """Preserve singular vectors and sigma_max while imposing a log-spaced spectrum."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("condition-controlled selected block must be square")
    if target_kappa < 1.0:
        raise ValueError("target_kappa must be at least one")
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    replacement = np.geomspace(
        singular_values[0], singular_values[0] / target_kappa, matrix.shape[0]
    )
    return left @ np.diag(replacement) @ right_t


def _build_raw_block(spec: dict[str, Any], seed: int) -> SelectedBlock:
    case = str(spec["case_name"])
    size = int(spec["block_size"])
    source = str(spec.get("case_source", "pypower"))
    system, matrix_source = build_engineering_system(
        {
            "case_name": case,
            "case_source": source,
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H, r, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=size,
        col_count=size,
        policy="largest_row_col_norms",
    )
    return SelectedBlock(
        block_id=f"{case}_{size}x{size}_raw",
        case_name=case,
        matrix_source=str(matrix_source),
        block_kind="raw_ieee_derived_selected_block",
        H=H,
        r=r,
        selected_rows=rows,
        selected_cols=cols,
        column_labels=_state_labels_for_cols(system.metadata, cols),
        provenance=(
            f"Generated {case.upper()} {source.upper()} weighted Jacobian, seed {seed}; "
            f"{BLOCK_SELECTION_RULE}; generated benchmark measurements, not field records"
        ),
    )


def build_selected_blocks(config: dict[str, Any]) -> list[SelectedBlock]:
    """Build raw selected blocks and disclosed condition-controlled stress variants."""

    seed = int(config["seed"])
    raw = [_build_raw_block(spec, seed) for spec in config["raw_blocks"]]
    controlled_spec = config["condition_controlled_blocks"]
    base = _build_raw_block(
        {
            "case_name": controlled_spec["base_case_name"],
            "case_source": controlled_spec.get("base_case_source", "pypower"),
            "block_size": controlled_spec["base_block_size"],
        },
        seed,
    )
    blocks = list(raw)
    for target in controlled_spec["target_kappas"]:
        target_kappa = float(target)
        exponent = round(math.log10(target_kappa))
        blocks.append(
            SelectedBlock(
                block_id=f"{base.case_name}_{base.H.shape[0]}x{base.H.shape[1]}_condition_controlled_k1e{exponent}",
                case_name=base.case_name,
                matrix_source=base.matrix_source,
                block_kind="ieee_derived_condition_controlled_stress_block",
                H=condition_controlled_variant(base.H, target_kappa),
                r=base.r.copy(),
                selected_rows=base.selected_rows.copy(),
                selected_cols=base.selected_cols.copy(),
                column_labels=list(base.column_labels),
                controlled_target_kappa=target_kappa,
                provenance=(
                    f"Derived from {base.block_id}; same selected rows, columns, singular "
                    f"vectors, sigma_max, and residual; log-spaced singular values impose "
                    f"kappa={target_kappa:.0f}; controlled stress block, not a raw IEEE block"
                ),
            )
        )
    return blocks


def _diagnose_polynomial(block: SelectedBlock, lam: float, degree: int) -> PolynomialDiagnostic:
    singular_values = block.singular_values
    beta = float(singular_values[0])
    alpha = float(lam) * beta**2
    domain_min = float(np.clip(0.9 * singular_values[-1] / beta, 1.0e-4, 0.999))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = fit_codesigned_bounded_polynomial(
            beta=beta,
            alpha=alpha,
            domain_min=domain_min,
            domain_max=1.0,
            degree=int(degree),
            margin=1.05,
        )
    coefficients = np.asarray(target.coefficients, dtype=np.float64)
    parity_pass = bool(
        np.all(np.isfinite(coefficients))
        and np.max(np.abs(coefficients[0::2]), initial=0.0) <= 1.0e-10
    )
    try:
        validation = validate_qsvt_polynomial(
            coefficients, parity="odd", bound_tolerance=BOUND_TOLERANCE
        )
        boundedness_pass = True
        bounded_max_abs = float(validation["max_abs_on_unit_interval"])
    except Exception:
        boundedness_pass = False
        bounded_max_abs = float(target.bounded_max_abs)
    normalized = singular_values / beta
    target_at_spectrum = (normalized / (normalized**2 + float(lam))) / float(target.bound_C)
    try:
        spectrum_error = float(np.max(np.abs(target.polynomial(normalized) - target_at_spectrum)))
    except Exception:
        spectrum_error = math.nan
    uniform_error = float(target.fit_max_abs_error)
    maximum = (
        float(max(uniform_error, spectrum_error))
        if np.isfinite(uniform_error) and np.isfinite(spectrum_error)
        else math.nan
    )
    return PolynomialDiagnostic(
        degree=int(degree),
        uniform_grid_error=uniform_error,
        spectrum_point_error=spectrum_error,
        max_polynomial_error=maximum,
        boundedness_pass=boundedness_pass,
        parity_pass=parity_pass,
        bounded_max_abs=bounded_max_abs,
        bound_C=float(target.bound_C),
    )


def _classical_adjoint(result: BlockDemoResult, alpha: float) -> tuple[str, float, float]:
    observable = next(obs for obs in result.observables if obs.sign_aware_required)
    timings: list[float] = []
    value = math.nan
    for _ in range(ADJOINT_TIMING_REPEATS):
        start = time.perf_counter()
        gram = result.H_block.T @ result.H_block + float(alpha) * np.eye(result.H_block.shape[1])
        rhs = result.H_block.T @ result.r_block
        value = float(np.linalg.solve(gram, observable.vector) @ rhs)
        timings.append(time.perf_counter() - start)
    return observable.observable_id, value, float(np.median(timings))


def _selected_functional_errors(result: BlockDemoResult) -> str:
    errors = {
        str(row["observable_id"]): float(row["relative_error"])
        for row in result.observable_rows
        if np.isfinite(float(row["relative_error"]))
    }
    return json.dumps(errors, sort_keys=True, separators=(",", ":"))


def _finite_shot_summary(result: BlockDemoResult, shots: int, seed: int) -> tuple[str, int, float]:
    if result.output_state is None or result.status_label != "pass":
        return "readout-failed", int(shots), math.nan
    signed = [
        {
            "observable_id": obs.observable_id,
            "observable_type": obs.observable_type,
            "vector": obs.vector,
        }
        for obs in result.observables
        if obs.sign_aware_required
    ]
    ridge_reference = {
        obs.observable_id: obs.exact_value(result.ridge_update) for obs in result.observables
    }
    rows = circuit_signed_readout_rows(
        observables=signed,
        output_state=result.output_state,
        physical_recovery_scale=result.physical_recovery_scale,
        ridge_reference=ridge_reference,
        shots_grid=(int(shots),),
        seed=int(seed),
    )
    errors = [
        abs(float(row["physical_signed_value_estimate"]) - float(row["ridge_reference_value"]))
        / max(abs(float(row["ridge_reference_value"])), 1.0e-15)
        for row in rows
    ]
    return ISOLATED_READOUT_STATUS, int(shots), float(np.mean(errors))


def _run_full_pipeline(
    block: SelectedBlock,
    lam: float,
    degree: int,
    tolerance: float,
    phase_cache_dir: Path,
) -> BlockDemoResult:
    beta = float(block.singular_values[0])
    return run_demo_for_block(
        case=block.case_name,
        matrix_source=block.matrix_source,
        H_block=block.H,
        r_block=block.r,
        selected_rows=block.selected_rows,
        selected_cols=block.selected_cols,
        column_labels=block.column_labels,
        alpha=float(lam) * beta**2,
        degree=int(degree),
        angle_solver="iterative",
        margin=1.05,
        domain_low_factor=0.9,
        pass_relative_tolerance=float(tolerance),
        phase_cache_dir=phase_cache_dir,
    )


def _base_row(block: SelectedBlock, lam: float, degree_ceiling: int) -> dict[str, Any]:
    singular_values = block.singular_values
    beta = float(singular_values[0])
    return {
        "run_id": f"phase2_{block.block_id}_lambda{lam:.4g}_dmax{degree_ceiling}",
        "case_name": block.case_name,
        "block_size": int(block.H.shape[0]),
        "matrix_source": block.matrix_source,
        "kappa": float(singular_values[0] / singular_values[-1]),
        "sigma_min": float(singular_values[-1]),
        "sigma_max": float(singular_values[0]),
        "beta": beta,
        "lambda": float(lam),
        "alpha": float(lam) * beta**2,
        "block_id": block.block_id,
        "block_kind": block.block_kind,
        "degree_ceiling": int(degree_ceiling),
        "selected_rows": " ".join(str(int(value)) for value in block.selected_rows),
        "selected_cols": " ".join(str(int(value)) for value in block.selected_cols),
        "block_checksum": array_checksum(block.H),
        "residual_checksum": array_checksum(block.r),
        "selection_rule": BLOCK_SELECTION_RULE,
        "measurement_state_provenance": block.provenance,
        "singular_values": " ".join(f"{value:.12g}" for value in singular_values),
        "controlled_target_kappa": block.controlled_target_kappa,
    }


def _legacy_representative_rows() -> list[dict[str, Any]]:
    path = Path("outputs/qsvt_selected_workload_extension/selected_workload_results.csv")
    if not path.is_file():
        return []
    old = pd.read_csv(path).drop_duplicates("workload_id")
    wanted = {
        "anchor_4x4_benchmark_alpha_d31": "existing primary correctness anchor",
        "8x8_benchmark_alpha_d31": "existing benchmark-alpha boundary row",
        "8x8_codesigned_lambda_matched_d31": "existing lambda-matched secondary anchor",
    }
    rows: list[dict[str, Any]] = []
    for workload, label in wanted.items():
        matches = old.loc[old["workload_id"] == workload]
        if matches.empty:
            continue
        source = matches.iloc[0]
        status = "feasible" if source["status"] == "feasible" else "degree-limited"
        rows.append(
            {
                "run_id": str(source["workload_id"]),
                "case_name": str(source["case"]),
                "block_size": int(str(source["block_size"]).split("x")[0]),
                "matrix_source": str(
                    source.get("measurement_state_provenance", "existing Phase 1 output")
                ),
                "kappa": float(source["kappa_block"]),
                "sigma_min": math.nan,
                "sigma_max": float(source["beta"]),
                "beta": float(source["beta"]),
                "lambda": float(source["lambda_alpha_over_beta2"]),
                "alpha": float(source["alpha"]),
                "degree_attempted": int(source["degree_attempted"]),
                "degree_min_feasible": int(source["degree_attempted"])
                if status == "feasible"
                else math.nan,
                "phase_count": int(source["phase_count"]),
                "target_error_tolerance": float(source["pass_relative_tolerance"]),
                "max_polynomial_error": float(source["target_fit_error"]),
                "spectrum_point_error": float(source["actual_singular_value_error"]),
                "uniform_grid_error": float(source["target_fit_error"]),
                "boundedness_pass": str(source["phase_synthesis_status"]) == "completed",
                "parity_pass": True,
                "phase_synthesis_status": str(source["phase_synthesis_status"]),
                "phase_synthesis_error": "not_applicable",
                "qsvt_statevector_status": "completed"
                if np.isfinite(source["update_relative_error_vs_ridge"])
                else "not_completed",
                "postselection_probability": float(source["postselection_probability"]),
                "update_relative_error_vs_matched_ridge": float(
                    source["update_relative_error_vs_ridge"]
                ),
                "selected_functional_errors": "see outputs/qsvt_selected_workload_extension",
                "finite_shot_readout_status": ISOLATED_READOUT_STATUS
                if status == "feasible"
                else "not_run",
                "finite_shot_shots": float(source["finite_shot_shots"]),
                "finite_shot_mean_relative_error": float(source["finite_shot_relative_error"]),
                "classical_adjoint_value": float(source["classical_adjoint_value"]),
                "classical_adjoint_time": float(source["classical_adjoint_median_seconds"]),
                "final_status": status,
                "failure_reason": "none" if status == "feasible" else label,
                "block_id": label,
                "block_kind": "existing_phase1_evidence",
                "degree_ceiling": int(source["degree_attempted"]),
                "selected_rows": str(source["selected_rows"]),
                "selected_cols": str(source["selected_cols"]),
                "block_checksum": str(source["block_checksum"]),
                "residual_checksum": "see_existing_output",
                "selection_rule": str(source["block_selection_rule"]),
                "measurement_state_provenance": str(source["measurement_state_provenance"]),
                "singular_values": str(source["singular_values"]),
                "controlled_target_kappa": math.nan,
                "polynomial_degree_evaluated": int(source["degree_attempted"]),
                "physical_recovery_factor_C_over_beta": math.nan,
                "classical_adjoint_observable": str(source["observable_id"]),
            }
        )
    return rows


def run_phase2_boundary(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_directory(Path(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")
    tolerance = float(config["target_error_tolerance"])
    ceilings = [int(value) for value in config["degree_ceilings"]]
    max_degree = max(ceilings)
    phase_max = int(config["phase_synthesis_max_degree"])
    lambdas = [float(value) for value in config["lambdas"]]
    blocks = build_selected_blocks(config)
    readout_config = config["finite_shot_readout"]

    rows: list[dict[str, Any]] = []
    diagnostics_cache: dict[tuple[str, float, int], PolynomialDiagnostic] = {}
    full_cache: dict[tuple[str, float, int], BlockDemoResult | Exception] = {}

    for block in blocks:
        for lam in lambdas:
            first_feasible_degree: int | None = None
            for degree in range(7, max_degree + 1, 2):
                diagnostic = _diagnose_polynomial(block, lam, degree)
                diagnostics_cache[(block.block_id, lam, degree)] = diagnostic
                if diagnostic.meets and diagnostic.max_polynomial_error <= tolerance:
                    first_feasible_degree = degree
                    break

            pipeline_result: BlockDemoResult | Exception | None = None
            if first_feasible_degree is not None and first_feasible_degree <= phase_max:
                cache_key = (block.block_id, lam, first_feasible_degree)
                try:
                    pipeline_result = _run_full_pipeline(
                        block,
                        lam,
                        first_feasible_degree,
                        tolerance,
                        phase_cache_dir,
                    )
                except Exception as exc:  # an explicit failed row is preferable to data loss
                    pipeline_result = exc
                full_cache[cache_key] = pipeline_result

            for ceiling in ceilings:
                row = _base_row(block, lam, ceiling)
                eligible = (
                    first_feasible_degree
                    if first_feasible_degree is not None and first_feasible_degree <= ceiling
                    else None
                )
                diagnostic_degree = eligible if eligible is not None else ceiling
                diagnostic = diagnostics_cache.get((block.block_id, lam, diagnostic_degree))
                if diagnostic is None:
                    diagnostic = _diagnose_polynomial(block, lam, diagnostic_degree)
                    diagnostics_cache[(block.block_id, lam, diagnostic_degree)] = diagnostic
                row.update(
                    {
                        "degree_attempted": int(diagnostic_degree),
                        "degree_min_feasible": eligible,
                        "phase_count": 0,
                        "target_error_tolerance": tolerance,
                        "max_polynomial_error": diagnostic.max_polynomial_error,
                        "spectrum_point_error": diagnostic.spectrum_point_error,
                        "uniform_grid_error": diagnostic.uniform_grid_error,
                        "boundedness_pass": diagnostic.boundedness_pass,
                        "parity_pass": diagnostic.parity_pass,
                        "phase_synthesis_status": "not_attempted",
                        "phase_synthesis_error": "not_applicable",
                        "qsvt_statevector_status": "not_attempted",
                        "postselection_probability": math.nan,
                        "update_relative_error_vs_matched_ridge": math.nan,
                        "selected_functional_errors": "not_run",
                        "finite_shot_readout_status": "not_run",
                        "finite_shot_shots": 0,
                        "finite_shot_mean_relative_error": math.nan,
                        "classical_adjoint_value": math.nan,
                        "classical_adjoint_time": math.nan,
                        "final_status": "degree-limited",
                        "failure_reason": (
                            "no polynomial in the tested odd-degree grid met the 1e-2 "
                            "criterion within this ceiling"
                        ),
                        "polynomial_degree_evaluated": int(diagnostic.degree),
                        "physical_recovery_factor_C_over_beta": diagnostic.bound_C
                        / float(block.singular_values[0]),
                        "classical_adjoint_observable": "not_run",
                    }
                )

                if eligible is None:
                    if not diagnostic.boundedness_pass or not diagnostic.parity_pass:
                        row["final_status"] = "tolerance-missing"
                        row["failure_reason"] = (
                            f"degree {diagnostic.degree} failed boundedness/parity and no "
                            "lower tested odd degree met the target"
                        )
                elif eligible > phase_max:
                    row["final_status"] = "phase-synthesis-failed"
                    row["phase_synthesis_status"] = "not_attempted_above_validated_ceiling"
                    row["phase_synthesis_error"] = (
                        f"minimum passing polynomial degree {eligible} exceeds validated "
                        f"phase-synthesis ceiling {phase_max}"
                    )
                    row["failure_reason"] = row["phase_synthesis_error"]
                elif isinstance(pipeline_result, Exception):
                    row["final_status"] = "statevector-failed"
                    row["phase_synthesis_status"] = "pipeline_exception"
                    row["phase_synthesis_error"] = (
                        f"{type(pipeline_result).__name__}: {pipeline_result}"
                    )
                    row["qsvt_statevector_status"] = "failed"
                    row["failure_reason"] = row["phase_synthesis_error"]
                elif isinstance(pipeline_result, BlockDemoResult):
                    common = pipeline_result.row_common
                    row["phase_count"] = int(common["phase_count"])
                    row["phase_synthesis_status"] = str(common["phase_synthesis_status"])
                    row["phase_synthesis_error"] = str(
                        pipeline_result.pipeline_metadata.get("phase_synthesis_failure", "")
                        or "none"
                    )
                    row["qsvt_statevector_status"] = (
                        "completed" if pipeline_result.output_state is not None else "failed"
                    )
                    row["postselection_probability"] = float(common["postselection_probability"])
                    row["update_relative_error_vs_matched_ridge"] = float(
                        common["update_relative_error_vs_ridge"]
                    )
                    row["selected_functional_errors"] = _selected_functional_errors(pipeline_result)
                    obs_id, adjoint_value, adjoint_time = _classical_adjoint(
                        pipeline_result, float(row["alpha"])
                    )
                    row["classical_adjoint_observable"] = obs_id
                    row["classical_adjoint_value"] = adjoint_value
                    row["classical_adjoint_time"] = adjoint_time
                    if (
                        common["phase_synthesis_status"] == "completed"
                        and np.isfinite(row["update_relative_error_vs_matched_ridge"])
                        and row["update_relative_error_vs_matched_ridge"] <= tolerance
                    ):
                        row["final_status"] = "feasible"
                        row["failure_reason"] = "none"
                    elif common["phase_synthesis_status"] != "completed":
                        row["final_status"] = "phase-synthesis-failed"
                        row["failure_reason"] = row["phase_synthesis_error"]
                    else:
                        row["final_status"] = "statevector-failed"
                        row["failure_reason"] = (
                            "phase synthesis completed but the matched-Ridge update error "
                            "exceeded 1e-2"
                        )

                    should_read = (
                        bool(readout_config.get("enabled", False))
                        and block.block_id == str(readout_config["block_id"])
                        and math.isclose(
                            lam, float(readout_config["lambda"]), rel_tol=0.0, abs_tol=1.0e-12
                        )
                        and row["final_status"] == "feasible"
                    )
                    if should_read:
                        status, shots, mean_error = _finite_shot_summary(
                            pipeline_result,
                            int(readout_config["shots"]),
                            int(readout_config["seed"]),
                        )
                        row["finite_shot_readout_status"] = status
                        row["finite_shot_shots"] = shots
                        row["finite_shot_mean_relative_error"] = mean_error
                        if status != ISOLATED_READOUT_STATUS:
                            row["final_status"] = "readout-failed"
                            row["failure_reason"] = "finite-shot signed-functional readout failed"

                if row["final_status"] not in FINAL_STATUSES:
                    raise RuntimeError(f"invalid final status: {row['final_status']}")
                rows.append(row)

    columns = REQUIRED_COLUMNS + EXTRA_COLUMNS
    frame = pd.DataFrame(rows, columns=columns)
    all_csv = output_dir / "all_attempts.csv"
    all_json = output_dir / "all_attempts.json"
    frame.to_csv(all_csv, index=False, na_rep="not_applicable")
    frame.to_json(all_json, orient="records", indent=2)

    successes = frame.loc[frame["final_status"] == "feasible"].copy()
    failures = frame.loc[frame["final_status"] != "feasible"].copy()
    successes.to_csv(output_dir / "successful_rows.csv", index=False, na_rep="not_applicable")
    failures.to_csv(output_dir / "failed_rows.csv", index=False, na_rep="not_applicable")

    representative = pd.DataFrame(_legacy_representative_rows(), columns=columns)
    new_representative_ids = [
        ("ieee30_16x16_raw", 6.9e-2, 45),
        ("ieee14_8x8_condition_controlled_k1e4", 6.9e-2, 45),
        ("ieee14_8x8_raw", 1.0e-2, 45),
        ("ieee14_8x8_condition_controlled_k1e3", 1.0e-2, 201),
    ]
    selected_new = []
    for block_id, lam, ceiling in new_representative_ids:
        match = frame.loc[
            (frame["block_id"] == block_id)
            & np.isclose(frame["lambda"].astype(float), lam)
            & (frame["degree_ceiling"] == ceiling)
        ]
        if not match.empty:
            selected_new.append(match.iloc[0].to_dict())
    representative = pd.concat(
        [representative, pd.DataFrame(selected_new, columns=columns)], ignore_index=True
    )
    representative.to_csv(
        output_dir / "representative_rows.csv", index=False, na_rep="not_applicable"
    )

    metadata = []
    for block in blocks:
        singular_values = block.singular_values
        metadata.append(
            {
                "block_id": block.block_id,
                "case_name": block.case_name,
                "matrix_source": block.matrix_source,
                "block_kind": block.block_kind,
                "block_size": int(block.H.shape[0]),
                "selected_row_indices": [int(value) for value in block.selected_rows],
                "selected_column_indices": [int(value) for value in block.selected_cols],
                "selection_rule": BLOCK_SELECTION_RULE,
                "measurement_state_provenance": block.provenance,
                "singular_values": [float(value) for value in singular_values],
                "sigma_min": float(singular_values[-1]),
                "sigma_max": float(singular_values[0]),
                "beta": float(singular_values[0]),
                "kappa": float(singular_values[0] / singular_values[-1]),
                "controlled_target_kappa": block.controlled_target_kappa,
                "block_checksum": array_checksum(block.H),
                "residual_checksum": array_checksum(block.r),
                "lambda_alpha_pairs": [
                    {
                        "lambda": lam,
                        "alpha": float(lam) * float(singular_values[0]) ** 2,
                    }
                    for lam in lambdas
                ],
            }
        )
    write_json(output_dir / "selected_block_metadata.json", metadata)

    harder_success = successes.loc[
        (successes["kappa"] >= 100.0) | (successes["block_size"] >= 16)
    ].sort_values(["lambda", "kappa", "block_size"], ascending=[True, False, False])
    target_regime = frame.loc[(frame["kappa"] >= 100.0) & (frame["lambda"] <= 1.0e-2)]
    summary = {
        "phase": 2,
        "target_error_tolerance": tolerance,
        "degree_ceilings": ceilings,
        "phase_synthesis_max_degree": phase_max,
        "lambda_values": lambdas,
        "block_count": len(blocks),
        "attempt_row_count": len(frame),
        "successful_row_count": len(successes),
        "failed_row_count": len(failures),
        "status_counts": {
            str(key): int(value) for key, value in frame["final_status"].value_counts().items()
        },
        "harder_success_exists": bool(not harder_success.empty),
        "best_harder_success_run_id": (
            str(harder_success.iloc[0]["run_id"]) if not harder_success.empty else None
        ),
        "success_at_kappa_ge_1e2_lambda_le_1e2": bool(
            (target_regime["final_status"] == "feasible").any()
        ),
        "negative_boundary_statement": (
            "For the tested raw and explicitly condition-controlled IEEE-derived selected "
            "blocks with kappa >= 1e2 and lambda <= 1e-2, no row was both accurate and "
            "phase-synthesizable within the tested polynomial and phase-synthesis limits."
        ),
        "finite_shot_readout_workload_count": int(
            frame.loc[frame["finite_shot_readout_status"] == ISOLATED_READOUT_STATUS]
            .drop_duplicates(["block_id", "lambda", "degree_attempted"])
            .shape[0]
        ),
        "claim_boundary": (
            "Selected-submatrix implementation-boundary evidence for the matched "
            "Ridge/Tikhonov "
            "filter; no quantum speedup, numerical superiority, full IEEE-scale execution, "
            "scalable residual loading, full sparse block encoding, or nonlinear-loop QSVT."
        ),
    }
    write_json(output_dir / "phase2_summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "artifact_name": "phase2_qsvt_boundary",
            "seed_provenance": {
                "status": "recorded",
                "seeds": {
                    "system_seed": int(config["seed"]),
                    "isolated_readout_seed": int(readout_config["seed"]),
                },
            },
            "source_config": "configs/qsvt_phase2_boundary.yaml",
            "regeneration_command": (
                ".venv/bin/python scripts/run_phase2_qsvt_boundary.py "
                "--config configs/qsvt_phase2_boundary.yaml"
            ),
            "outputs": [
                "outputs/phase2_qsvt_boundary/all_attempts.csv",
                "outputs/phase2_qsvt_boundary/all_attempts.json",
                "outputs/phase2_qsvt_boundary/representative_rows.csv",
                "outputs/phase2_qsvt_boundary/selected_block_metadata.json",
                "outputs/phase2_qsvt_boundary/phase2_summary.json",
            ],
            "claim_boundary": summary["claim_boundary"],
        },
    )

    readme = f"""# Phase 2 harder selected-block QSVT boundary evidence

This directory is generated by:

```bash
.venv/bin/python scripts/run_phase2_qsvt_boundary.py --config configs/qsvt_phase2_boundary.yaml
```

## Selection and provenance

Raw blocks use the deterministic, pre-solve `largest_row_col_norms` rule on generated
PYPOWER weighted Jacobians (seed {config["seed"]}). The raw blocks are IEEE-14 8x8 and
IEEE-30 16x16. The two condition-controlled 8x8 stress blocks retain the IEEE-14
selection, residual, singular vectors, and sigma_max, but replace the singular values
with a log-spaced schedule at kappa 1e3 or 1e4. They are not raw IEEE measurement blocks.

## Sweep

- lambda: {lambdas}
- degree ceilings: {ceilings}
- target approximation and matched-update tolerance: {tolerance}
- phase synthesis is attempted only through the validated degree-{phase_max} ceiling
- finite-shot readout: {readout_config["shots"]} shots for one disclosed harder feasible row

For each `(block, lambda, degree ceiling)` row, the script searches odd polynomial
degrees from 7 through the ceiling and records the minimum degree meeting boundedness,
parity, uniform-grid error, and block-spectrum error. Phase synthesis and dense
statevector execution are then attempted when that degree does not exceed {phase_max}.

## Status definitions

- `feasible`: polynomial criterion, phase synthesis, and matched-Ridge statevector update
  error all pass at the stated tolerance.
- `degree-limited`: no tested odd degree within the ceiling meets the error criterion.
- `tolerance-missing`: the ceiling polynomial fails boundedness/parity and no lower tested
  degree meets the criterion.
- `phase-synthesis-failed`: an accurate polynomial exists, but synthesis fails or lies
  above the validated synthesis ceiling.
- `statevector-failed`: phases complete but circuit execution fails or misses the update
  criterion.
- `readout-failed`: the designated finite-shot readout does not complete.
- `skipped-with-reason`: an attempt is intentionally omitted with an explicit reason.

`all_attempts.csv/json` retain every ceiling row. `successful_rows.csv` and
`failed_rows.csv` partition those rows. `representative_rows.csv` adds the three existing
anchor/boundary rows used for manuscript context. `selected_block_metadata.json` records
all block provenance and spectra.

The reference is Ridge/Tikhonov at the same alpha. These outputs are component-level
boundary evidence, not a quantum-speedup result, not a full sparse block encoding, and
not full IEEE-scale or nonlinear-loop quantum execution.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"frame": frame, "representative": representative, "summary": summary}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Phase 2 config must be a mapping")
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    result = run_phase2_boundary(config)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
