"""Task D: larger-matrix QSVT boundary / failure diagnosis.

Ingests existing QSVT polynomial/matrix/gate diagnostics and classifies each
configuration into a fixed controlled vocabulary - ``pass``, ``degree_limited``,
``tolerance_missed``, ``phase_unavailable``, ``output_missing``, ``skipped`` -
so the manuscript can state *where* the QSVT-compatible pathway succeeds, misses
tolerance, or stops, instead of implying a uniform success.

This module does not tune anything to pass and does not re-run experiments. It
reads artifacts already on disk, computes ``pass_tolerance`` from the recorded
error vs. the requested (or, for the fixed-degree multi-case diagnostic, the
documented reference) tolerance, and records a conservative likely cause. A
missing source is reported as ``output_missing`` rather than silently dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper._common import read_csv
from robust_qsvt_se.paper.tqe_revision_support_common import (
    REVISION_OUTPUT_ROOT,
    find_forbidden,
    write_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import DEFAULT_EPSILON
from robust_qsvt_se.utils.io import ensure_directory

BOUNDARY_DIR = REVISION_OUTPUT_ROOT / "qsvt_boundary"

BOUNDARY_TYPES = frozenset(
    {
        "pass",
        "degree_limited",
        "tolerance_missed",
        "phase_unavailable",
        "output_missing",
        "skipped",
    }
)

# A degree at/above this is treated as "already high": a further miss is reported
# as degree/synthesis-limited rather than as a simple tolerance miss.
HIGH_DEGREE_THRESHOLD = 51

BOUNDARY_COLUMNS = [
    "case_or_block",
    "matrix_shape",
    "alpha",
    "degree",
    "target_tolerance",
    "tolerance_source",
    "actual_max_error",
    "pass_tolerance",
    "sigma_min",
    "sigma_max",
    "kappa",
    "normalized_alpha",
    "boundary_type",
    "likely_cause",
    "safe_interpretation",
    "unsafe_interpretation",
    "evaluation_stage",
    "phase_status",
    "source_artifact",
]

# (safe_interpretation, unsafe_interpretation) per boundary type. The unsafe text
# is a do-not-use list, so it intentionally contains phrasing to avoid.
INTERPRETATIONS: dict[str, tuple[str, str]] = {
    "pass": (
        "QSVT-compatible target met the configured tolerance for this controlled benchmark "
        "configuration; matched-alpha implementation-pathway evidence, not a speedup claim.",
        "QSVT outperforms Ridge or demonstrates quantum speedup.",
    ),
    "degree_limited": (
        "Target tolerance not met at the configured (already high) polynomial degree; the "
        "regularized inverse-like target is degree/synthesis-limited for this ill-conditioned "
        "configuration - a reported boundary, not a corrected result.",
        "QSVT is broken, or QSVT beats Ridge once the degree is tuned.",
    ),
    "tolerance_missed": (
        "Target tolerance not met at the configured degree; a higher degree or a co-designed "
        "target may be required. Reported as a boundary, not tuned to pass.",
        "QSVT failure implies Ridge superiority or that no quantum pathway exists.",
    ),
    "phase_unavailable": (
        "Polynomial target met tolerance, but the end-to-end QSVT pathway is bounded by "
        "phase-factor synthesis, which is validated only for selected configurations; "
        "polynomial-fallback evidence applies here.",
        "Full QSVT circuit executed on hardware with full-vector readout solved.",
    ),
    "skipped": (
        "Configuration skipped before evaluation (for example the requested block exceeds the "
        "matrix shape, or a circuit-object-only budget); recorded as skipped, not as success "
        "or failure.",
        "A negative result was hidden or omitted.",
    ),
    "output_missing": (
        "Source diagnostic output is absent on disk; recorded as missing rather than inferred "
        "or assumed to pass.",
        "Assume the configuration was run and passed.",
    ),
}


def derive_boundary(record: dict[str, Any]) -> tuple[str, Any, str]:
    """Return ``(boundary_type, pass_tolerance, likely_cause)`` for a normalized row.

    Pure and side-effect free so it can be unit-tested against controlled inputs.
    """

    err = record.get("actual_max_error")
    tol = record.get("target_tolerance")
    degree = record.get("degree")
    stage = record.get("evaluation_stage", "")
    phase = record.get("phase_status", "unknown")
    phase_in_scope = bool(record.get("phase_in_scope", False))
    run = str(record.get("run_status_raw") or "").lower()
    circuit = str(record.get("circuit_status_raw") or "").lower()
    cond_note = _condition_note(
        record.get("kappa"), record.get("sigma_min"), record.get("normalized_alpha")
    )

    if err is None:
        if stage == "gate_level":
            if "complete" in circuit:
                return (
                    "pass",
                    True,
                    "Gate-level QSVT circuit instantiated and simulated; validates circuit "
                    "instantiation for this selected subproblem, not a target-tolerance gate.",
                )
            return (
                "skipped",
                "unknown",
                f"Gate-level audit skipped before simulation ({run or circuit or 'budget'}).",
            )
        return (
            "skipped",
            "unknown",
            f"Configuration skipped before polynomial evaluation "
            f"({run or 'no error metric available'}).",
        )

    pass_tolerance = (tol is not None) and (err <= tol)
    if pass_tolerance:
        if phase_in_scope and phase in ("not_attempted", "failed"):
            cause = (
                f"Polynomial approximation met tolerance (max error {err:.3e} <= {tol:.3e}); "
                f"phase-factor synthesis {phase} for this configuration, so only "
                "polynomial-fallback evidence applies."
            )
            return "phase_unavailable", True, cause + cond_note
        return (
            "pass",
            True,
            f"Polynomial approximation met tolerance (max error {err:.3e} <= {tol:.3e})."
            + cond_note,
        )

    base = (
        f"Target tolerance {tol:.3e} missed (max error {err:.3e})"
        if tol is not None
        else f"No requested tolerance; max error {err:.3e}"
    )
    if degree is not None and int(degree) >= HIGH_DEGREE_THRESHOLD:
        return (
            "degree_limited",
            False,
            f"{base} at already-high degree {int(degree)}; degree/synthesis-limited." + cond_note,
        )
    suffix = f" at degree {int(degree)}" if degree is not None else ""
    return (
        "tolerance_missed",
        False if tol is not None else "unknown",
        f"{base}{suffix}; higher degree or co-designed target may be required." + cond_note,
    )


def run_boundary_diagnostics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", BOUNDARY_DIR)))

    records: list[dict[str, Any]] = []
    records += _ingest_multicase(resolved)
    records += _ingest_larger_qsvt(resolved)
    records += _ingest_gate_coverage(resolved)
    records += _ingest_precision_sweep(resolved)

    rows = [_finalize_row(record) for record in records]
    frame = pd.DataFrame(rows, columns=BOUNDARY_COLUMNS)
    _self_check_safe_fields(frame)

    boundary_csv = output_dir / "qsvt_boundary_diagnostics.csv"
    summary_md = output_dir / "qsvt_boundary_diagnostics.md"
    frame.to_csv(boundary_csv, index=False)
    summary_md.write_text(_summary_markdown(frame), encoding="utf-8")

    artifacts = {
        "qsvt_boundary_diagnostics_csv": boundary_csv,
        "qsvt_boundary_diagnostics_md": summary_md,
    }
    manifest = write_manifest(
        output_dir=output_dir,
        artifact_name="qsvt_boundary_diagnostics",
        description=(
            "Larger-matrix QSVT boundary/failure diagnosis over existing diagnostics. "
            "No tuning; missing outputs are reported, not hidden."
        ),
        artifacts=artifacts,
        extra={
            "controlled_vocabulary": sorted(BOUNDARY_TYPES),
            "boundary_type_counts": frame["boundary_type"].value_counts().to_dict(),
            "high_degree_threshold": HIGH_DEGREE_THRESHOLD,
            "reference_tolerance_default_epsilon": float(DEFAULT_EPSILON),
            "tuned_to_pass": False,
        },
        manifest_name="qsvt_boundary_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "frame": frame, "artifacts": artifacts}


def _ingest_multicase(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolved.get(
        "multicase_csv",
        "outputs/qsvt_multicase_approximation_diagnostics/multicase_approximation_summary.csv",
    )
    frame = read_csv(path)
    if frame.empty:
        return [_missing_record(str(path), "full_matrix_polynomial")]
    records = []
    for _, row in frame.iterrows():
        sigma_max = _f(row.get("sigma_max"))
        records.append(
            {
                "source_artifact": str(path),
                "case_or_block": str(row.get("case_name", "")),
                "matrix_shape": f"{_i(row.get('m'))}x{_i(row.get('n'))}",
                "alpha": _f(row.get("alpha")),
                "degree": _i(row.get("degree")),
                # The fixed-degree multi-case run requested no tolerance; compare against the
                # documented repository reference epsilon and disclose the source.
                "target_tolerance": float(DEFAULT_EPSILON),
                "tolerance_source": "repo_reference_default_epsilon (run requested none)",
                "actual_max_error": _f(row.get("max_pointwise_error")),
                "sigma_min": _f(row.get("sigma_min")),
                "sigma_max": sigma_max,
                "kappa": _f(row.get("kappa")),
                "normalized_alpha": _normalized_alpha(_f(row.get("alpha")), sigma_max),
                "evaluation_stage": "full_matrix_polynomial",
                "phase_status": "not_in_scope",
                "phase_in_scope": False,
                "run_status_raw": str(row.get("status", "")),
                "circuit_status_raw": "",
            }
        )
    return records


def _ingest_larger_qsvt(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolved.get("larger_qsvt_csv") or _latest_larger_qsvt_csv()
    if path is None:
        return [
            _missing_record(
                "results/tqe_revision_evidence/*/larger_qsvt_matrix_validation.csv",
                "selected_block_polynomial",
            )
        ]
    frame = read_csv(path)
    if frame.empty:
        return [_missing_record(str(path), "selected_block_polynomial")]
    records = []
    for _, row in frame.iterrows():
        sigma_max = _f(row.get("sigma_max"))
        failure = str(row.get("failure_reason") or "")
        records.append(
            {
                "source_artifact": str(path),
                "case_or_block": f"{row.get('case_name', '')}_{row.get('block_shape', '')}",
                "matrix_shape": str(row.get("block_shape", "")),
                "alpha": _f(row.get("alpha")),
                "degree": _i(row.get("degree")),
                "target_tolerance": _f(row.get("target_epsilon")),
                "tolerance_source": "run_requested",
                "actual_max_error": _f(row.get("grid_max_error")),
                "sigma_min": _f(row.get("sigma_min")),
                "sigma_max": sigma_max,
                "kappa": _f(row.get("condition_number")),
                "normalized_alpha": _normalized_alpha(_f(row.get("alpha")), sigma_max),
                "evaluation_stage": "selected_block_polynomial",
                "phase_status": _normalize_phase(row.get("phase_synthesis_status"), True),
                "phase_in_scope": True,
                "run_status_raw": failure,
                "circuit_status_raw": "",
            }
        )
    return records


def _ingest_gate_coverage(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolved.get(
        "gate_coverage_csv",
        "outputs/tqe_qsvt_additional_experiments/full_gate_level_qsvt_coverage/"
        "full_gate_level_qsvt_coverage_results.csv",
    )
    frame = read_csv(path)
    if frame.empty:
        return [_missing_record(str(path), "gate_level")]
    records = []
    for _, row in frame.iterrows():
        size = _i(row.get("subproblem_size"))
        records.append(
            {
                "source_artifact": str(path),
                "case_or_block": f"{row.get('case_name', '')}_{size}x{size}",
                "matrix_shape": f"{size}x{size}",
                "alpha": _f(row.get("alpha")),
                "degree": _i(row.get("degree")),
                "target_tolerance": _f(row.get("epsilon_target")),
                "tolerance_source": "run_requested",
                # The gate-coverage file records circuit/update errors, not a polynomial grid
                # error, so no tolerance-grid error is asserted here.
                "actual_max_error": None,
                "sigma_min": None,
                "sigma_max": None,
                "kappa": None,
                "normalized_alpha": None,
                "evaluation_stage": "gate_level",
                "phase_status": _normalize_phase(row.get("phase_synthesis_status"), True),
                "phase_in_scope": True,
                "run_status_raw": str(row.get("failure_or_skip_reason") or ""),
                "circuit_status_raw": str(row.get("qsvt_circuit_status") or ""),
            }
        )
    return records


def _ingest_precision_sweep(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolved.get(
        "precision_sweep_csv",
        "outputs/tqe_qsvt_additional_experiments/degree_alpha_precision_sweep/"
        "degree_alpha_precision_sweep_results.csv",
    )
    frame = read_csv(path)
    if frame.empty:
        return [_missing_record(str(path), "degree_alpha_sweep")]
    representative = _precision_representatives(frame)
    records = []
    for _, row in representative.iterrows():
        sigma_max = _f(row.get("sigma_max"))
        size = _i(row.get("subproblem_size"))
        records.append(
            {
                "source_artifact": str(path),
                "case_or_block": f"{row.get('case_name', '')}_{size}x{size}",
                "matrix_shape": str(row.get("matrix_shape", f"{size}x{size}")),
                "alpha": _f(row.get("alpha")),
                "degree": _i(row.get("degree")),
                "target_tolerance": _f(row.get("epsilon_target")),
                "tolerance_source": "run_requested",
                "actual_max_error": _f(
                    row.get("max_approximation_error_on_actual_singular_values")
                ),
                "sigma_min": _f(row.get("sigma_min")),
                "sigma_max": sigma_max,
                "kappa": _f(row.get("condition_number")),
                "normalized_alpha": _normalized_alpha(_f(row.get("alpha")), sigma_max),
                "evaluation_stage": "degree_alpha_sweep",
                "phase_status": _normalize_phase(row.get("phase_synthesis_status"), True),
                "phase_in_scope": True,
                "run_status_raw": str(row.get("run_status", "")),
                "circuit_status_raw": "",
            }
        )
    return records


def _precision_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["case_name", "subproblem_size", "alpha", "epsilon_target"]
    if not set(group_cols).issubset(frame.columns):
        return frame
    work = frame.copy()
    meets = work.get("meets_epsilon_on_actual_singular_values")
    work["_meets_rank"] = (~meets.astype(bool)).astype(int) if meets is not None else 1
    work["_phase_rank"] = (
        work.get("phase_synthesis_status", "")
        .astype(str)
        .map(lambda value: 0 if "pass" in value.lower() else 1)
    )
    work["_error"] = pd.to_numeric(
        work.get("max_approximation_error_on_actual_singular_values"), errors="coerce"
    ).fillna(np.inf)
    work = work.sort_values([*group_cols, "_meets_rank", "_phase_rank", "_error"])
    return work.groupby(group_cols, as_index=False).first()


def _finalize_row(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("boundary_type") == "output_missing":
        boundary_type = "output_missing"
        pass_tolerance: Any = "unknown"
        likely_cause = record.get("likely_cause", "Source diagnostic output is absent on disk.")
    else:
        boundary_type, pass_tolerance, likely_cause = derive_boundary(record)
    safe, unsafe = INTERPRETATIONS[boundary_type]
    return {
        "case_or_block": record.get("case_or_block", ""),
        "matrix_shape": record.get("matrix_shape", ""),
        "alpha": _blank_if_none(record.get("alpha")),
        "degree": _blank_if_none(record.get("degree")),
        "target_tolerance": _blank_if_none(record.get("target_tolerance")),
        "tolerance_source": record.get("tolerance_source", ""),
        "actual_max_error": _blank_if_none(record.get("actual_max_error")),
        "pass_tolerance": pass_tolerance,
        "sigma_min": _blank_if_none(record.get("sigma_min")),
        "sigma_max": _blank_if_none(record.get("sigma_max")),
        "kappa": _blank_if_none(record.get("kappa")),
        "normalized_alpha": _blank_if_none(record.get("normalized_alpha")),
        "boundary_type": boundary_type,
        "likely_cause": likely_cause,
        "safe_interpretation": safe,
        "unsafe_interpretation": unsafe,
        "evaluation_stage": record.get("evaluation_stage", ""),
        "phase_status": record.get("phase_status", ""),
        "source_artifact": record.get("source_artifact", ""),
    }


def _missing_record(path: str, stage: str) -> dict[str, Any]:
    return {
        "source_artifact": path,
        "case_or_block": f"<missing:{Path(path).name}>",
        "matrix_shape": "",
        "alpha": None,
        "degree": None,
        "target_tolerance": None,
        "tolerance_source": "",
        "actual_max_error": None,
        "sigma_min": None,
        "sigma_max": None,
        "kappa": None,
        "normalized_alpha": None,
        "evaluation_stage": stage,
        "phase_status": "",
        "phase_in_scope": False,
        "run_status_raw": "source output missing",
        "circuit_status_raw": "",
        "boundary_type": "output_missing",
        "likely_cause": f"Expected diagnostic output {path} is absent on disk.",
    }


def _summary_markdown(frame: pd.DataFrame) -> str:
    counts = frame["boundary_type"].value_counts().to_dict()
    lines = [
        "# QSVT Boundary / Failure Diagnosis",
        "",
        "Where the QSVT-compatible polynomial/matrix/gate diagnostics **pass, miss tolerance, "
        "or stop**, classified into a fixed controlled vocabulary. Nothing is tuned to pass; "
        "missing source outputs are reported as `output_missing`. This is implementation-pathway "
        "boundary analysis under matched alpha - not a speedup claim and not a QSVT-over-Ridge "
        "claim.",
        "",
        "## Boundary-Type Counts",
        "",
        "| Boundary type | Rows |",
        "| --- | --- |",
    ]
    for boundary_type in sorted(BOUNDARY_TYPES):
        lines.append(f"| {boundary_type} | {int(counts.get(boundary_type, 0))} |")
    lines += [
        "",
        "## Rows by Evaluation Stage",
        "",
        "| Stage | Rows | Pass | Degree-limited | Tolerance-missed | Phase-unavailable | Skipped |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage, group in frame.groupby("evaluation_stage"):
        vc = group["boundary_type"].value_counts().to_dict()
        lines.append(
            f"| {stage} | {len(group)} | {vc.get('pass', 0)} | {vc.get('degree_limited', 0)} | "
            f"{vc.get('tolerance_missed', 0)} | {vc.get('phase_unavailable', 0)} | "
            f"{vc.get('skipped', 0)} |"
        )
    headline = frame[
        frame["evaluation_stage"].isin(["full_matrix_polynomial", "selected_block_polynomial"])
    ]
    lines += [
        "",
        "## Headline Larger-Matrix Rows",
        "",
        "| Case/block | Shape | Degree | Target tol | Max error | kappa | Boundary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in headline.iterrows():
        lines.append(
            f"| `{row['case_or_block']}` | {row['matrix_shape']} | {row['degree']} | "
            f"{_fmt(row['target_tolerance'])} | {_fmt(row['actual_max_error'])} | "
            f"{_fmt(row['kappa'])} | {row['boundary_type']} |"
        )
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `pass`: the polynomial target met the configured tolerance (gate-level rows mean "
        "the QSVT circuit was instantiated and simulated for a selected subproblem).",
        "- `degree_limited`: tolerance missed at an already-high degree "
        f"(>= {HIGH_DEGREE_THRESHOLD}); the regularized inverse-like target is "
        "degree/synthesis-limited for ill-conditioned blocks.",
        "- `tolerance_missed`: tolerance missed at a lower degree; a higher degree or a "
        "co-designed target may be required.",
        "- `phase_unavailable`: the polynomial met tolerance but end-to-end QSVT is bounded "
        "by phase-factor synthesis, validated only for selected configurations.",
        "- `skipped` / `output_missing`: configuration skipped before evaluation, or the "
        "source output is absent - reported, not hidden.",
        "",
        "Multi-case full-matrix rows requested no tolerance; they are compared against the "
        f"documented reference epsilon = {float(DEFAULT_EPSILON)} (disclosed in "
        "`tolerance_source`), not a value attributed to that run.",
        "",
    ]
    return "\n".join(lines)


def _self_check_safe_fields(frame: pd.DataFrame) -> None:
    safe_text = "\n".join(
        frame["safe_interpretation"].astype(str).tolist()
        + frame["likely_cause"].astype(str).tolist()
    )
    violations = find_forbidden(safe_text)
    if violations:
        raise RuntimeError(f"boundary safe interpretation contains forbidden phrases: {violations}")
    invalid = set(frame["boundary_type"]) - BOUNDARY_TYPES
    if invalid:
        raise RuntimeError(f"boundary_type outside controlled vocabulary: {invalid}")


def _condition_note(kappa: float | None, sigma_min: float | None, normalized_alpha: float | None):
    notes: list[str] = []
    if kappa is not None and kappa > 1.0e3:
        notes.append(
            f" Large condition number (kappa={kappa:.2e}) makes the regularized inverse-like "
            "target harder to approximate."
        )
    if sigma_min is not None and sigma_min <= 1.0e-8:
        notes.append(f" Smallest singular value is near zero (sigma_min={sigma_min:.2e}).")
    if normalized_alpha is not None and normalized_alpha < 1.0e-6:
        notes.append(
            f" Small normalized alpha (alpha/beta^2={normalized_alpha:.2e}) sharpens the target "
            "near the origin and demands higher degree."
        )
    return "".join(notes)


def _normalize_phase(status: Any, in_scope: bool) -> str:
    if not in_scope:
        return "not_in_scope"
    text = str(status or "").lower()
    if not text:
        return "unknown"
    if "pass" in text or "completed" in text or "success" in text:
        return "passed"
    if "fail" in text or "error" in text:
        return "failed"
    if "skip" in text or "budget" in text or "no_phase" in text or "none" in text:
        return "not_attempted"
    return "unknown"


def _latest_larger_qsvt_csv() -> str | None:
    root = Path("results/tqe_revision_evidence")
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob("*/larger_qsvt_matrix_validation.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not read_csv(candidate).empty:
            return str(candidate)
    return None


def _normalized_alpha(alpha: float | None, sigma_max: float | None) -> float | None:
    if alpha is None or sigma_max is None or sigma_max == 0.0:
        return None
    return float(alpha) / float(sigma_max) ** 2


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(result) else result


def _i(value: Any) -> int | None:
    result = _f(value)
    return None if result is None else round(result)


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the QSVT boundary diagnostics table")
    parser.add_argument("--output-dir", default=str(BOUNDARY_DIR))
    args = parser.parse_args(argv)
    run = run_boundary_diagnostics({"output_dir": args.output_dir})
    frame = run["frame"]
    counts = frame["boundary_type"].value_counts().to_dict()
    print(
        f"Boundary diagnostics complete ({len(frame)} rows, {counts}) -> "
        f"{run['artifacts']['qsvt_boundary_diagnostics_csv']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
