from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.external_backend_sanity import (
    run_external_backend_sanity_regression,
    sanity_passed_backends,
)
from robust_qsvt_se.qsvt.external_phase_candidates import (
    ExternalPhaseCandidate,
    build_external_phase_candidates,
    candidate_is_safe_for_backend,
)
from robust_qsvt_se.qsvt.phase_backend_adapters import available_backend_adapters
from robust_qsvt_se.utils.io import ensure_directory, write_json

TARGET_TOLERANCE = 1.0e-3
PHASE_VALIDATION_CAVEAT = (
    "External-backend scalar phase-response validation. Full-domain and "
    "actual-singular-value validation are reported separately. This is not hardware "
    "execution, quantum speedup, quantum advantage, or QSVT superiority over Ridge."
)


def run_external_backend_phase_validation(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sanity_run = run_external_backend_sanity_regression(
        {"output_dir": str(resolved["sanity_output_dir"])}
    )
    trusted_backends = sanity_passed_backends(sanity_run["summary"])
    candidate_run = build_external_phase_candidates(resolved["candidate_config"])
    candidates: list[ExternalPhaseCandidate] = candidate_run["candidates"]
    adapters = available_backend_adapters(enable_local_optimization=False)

    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    external_pass_found = False

    for adapter in adapters:
        if adapter.backend_name == "local_optimization_qsp" and external_pass_found:
            summary_rows.append(_not_needed_row(adapter.backend_name, candidates[0]))
            continue
        for candidate in candidates:
            safe, status, reason = candidate_is_safe_for_backend(
                candidate,
                backend_name=adapter.backend_name,
            )
            if adapter.backend_name not in trusted_backends:
                summary_rows.append(
                    _skip_row(
                        adapter.backend_name,
                        candidate,
                        "sanity regression failed or skipped",
                    )
                )
                continue
            if not safe:
                summary_rows.append(
                    _skip_row(adapter.backend_name, candidate, reason, status=status)
                )
                continue
            result = adapter.synthesize(candidate)
            if result.status != "passed_synthesis" or result.phases is None:
                summary_rows.append(_synthesis_failure_row(candidate, result))
                continue
            phases = np.asarray(result.phases, dtype=np.float64)
            try:
                validation = _validate_response(adapter, candidate, phases, result)
            except Exception as exc:
                summary_rows.append(
                    _response_failure_row(adapter.backend_name, candidate, str(exc))
                )
                continue
            summary_rows.append(validation["summary"])
            phase_rows.extend(validation["phase_rows"])
            response_rows.extend(validation["response_rows"])
            error_rows.extend(validation["error_rows"])
            if (
                adapter.backend_name not in {"local_optimization_qsp"}
                and validation["summary"]["passed_1e_minus_3_full_domain"]
            ):
                external_pass_found = True

    summary = pd.DataFrame(summary_rows)
    phases = pd.DataFrame(phase_rows)
    responses = pd.DataFrame(response_rows)
    errors = pd.DataFrame(error_rows)
    summary_csv = output_dir / "external_backend_phase_validation_summary.csv"
    summary_json = output_dir / "external_backend_phase_validation_summary.json"
    phases_csv = output_dir / "external_backend_phase_angles.csv"
    responses_csv = output_dir / "external_backend_phase_response_values.csv"
    errors_csv = output_dir / "external_backend_phase_error_grid.csv"
    report_md = output_dir / "external_backend_phase_report.md"
    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": summary_rows, "caveat": PHASE_VALIDATION_CAVEAT})
    phases.to_csv(phases_csv, index=False)
    responses.to_csv(responses_csv, index=False)
    errors.to_csv(errors_csv, index=False)
    report_md.write_text(_phase_report(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "external_backend_phase_validation_summary_csv": str(summary_csv),
            "external_backend_phase_validation_summary_json": str(summary_json),
            "external_backend_phase_angles_csv": str(phases_csv),
            "external_backend_phase_response_values_csv": str(responses_csv),
            "external_backend_phase_error_grid_csv": str(errors_csv),
            "external_backend_phase_report_md": str(report_md),
            "external_backend_sanity_summary_csv": str(
                Path(resolved["sanity_output_dir"]) / "external_backend_sanity_summary.csv"
            ),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "external_backend_phase_validation_summary_csv": summary_csv,
            "external_backend_phase_validation_summary_json": summary_json,
            "external_backend_phase_angles_csv": phases_csv,
            "external_backend_phase_response_values_csv": responses_csv,
            "external_backend_phase_error_grid_csv": errors_csv,
            "external_backend_phase_report_md": report_md,
            "manifest": manifest,
        },
    }


def _validate_response(
    adapter: Any,
    candidate: ExternalPhaseCandidate,
    phases: np.ndarray,
    result: Any,
) -> dict[str, Any]:
    full_response = adapter.evaluate_response(candidate.full_domain_grid, phases, candidate)
    actual_response = adapter.evaluate_response(candidate.actual_singular_values, phases, candidate)
    full_errors = np.abs(full_response - candidate.full_domain_target)
    actual_errors = np.abs(actual_response - candidate.actual_singular_targets)
    full_max = float(np.max(full_errors))
    actual_max = float(np.max(actual_errors)) if actual_errors.size else np.nan
    passed_full = bool(full_max <= TARGET_TOLERANCE)
    passed_actual = bool(actual_max <= TARGET_TOLERANCE) if actual_errors.size else False
    status = "passed" if passed_full else "failed_response_validation"
    failure_reason = "" if passed_full else "full-domain phase response exceeds 1e-3"
    summary = {
        "backend_name": adapter.backend_name,
        "backend_version": result.metadata.get("version", ""),
        "candidate_name": candidate.candidate_name,
        "alpha": float(candidate.alpha),
        "degree": int(candidate.degree),
        "input_basis": result.input_basis,
        "phase_count": int(phases.size),
        "phase_convention": result.convention,
        "response_convention": adapter.response_convention,
        "native_max_error": float(candidate.native_max_error_full_domain),
        "phase_response_max_error_full_domain": full_max,
        "phase_response_mean_error_full_domain": float(np.mean(full_errors)),
        "phase_response_rms_error_full_domain": float(np.sqrt(np.mean(full_errors**2))),
        "phase_response_max_error_actual_singular_values_if_available": actual_max,
        "grid_size": int(candidate.full_domain_grid.size),
        "actual_singular_value_count_if_available": int(candidate.actual_singular_values.size),
        "passed_1e_minus_3_full_domain": passed_full,
        "passed_1e_minus_3_actual_singular_values": passed_actual,
        "status": status,
        "failure_reason": failure_reason,
        "caveat": PHASE_VALIDATION_CAVEAT,
    }
    phase_rows = [
        {
            "backend_name": adapter.backend_name,
            "candidate_name": candidate.candidate_name,
            "phase_index": int(index),
            "phase_angle": float(phase),
        }
        for index, phase in enumerate(phases)
    ]
    response_rows = _domain_rows(adapter.backend_name, candidate, "full_domain", full_response)
    response_rows.extend(
        _domain_rows(adapter.backend_name, candidate, "actual_singular_values", actual_response)
    )
    error_rows = _error_rows(
        adapter.backend_name,
        candidate,
        "full_domain",
        full_response,
        full_errors,
    )
    error_rows.extend(
        _error_rows(
            adapter.backend_name,
            candidate,
            "actual_singular_values",
            actual_response,
            actual_errors,
        )
    )
    return {
        "summary": summary,
        "phase_rows": phase_rows,
        "response_rows": response_rows,
        "error_rows": error_rows,
    }


def _domain_rows(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    domain: str,
    response: np.ndarray,
) -> list[dict[str, Any]]:
    if domain == "full_domain":
        points = candidate.full_domain_grid
        targets = candidate.full_domain_target
    else:
        points = candidate.actual_singular_values
        targets = candidate.actual_singular_targets
    return [
        {
            "backend_name": backend_name,
            "candidate_name": candidate.candidate_name,
            "evaluation_domain": domain,
            "sigma_normalized": float(point),
            "target_value": float(target),
            "phase_response_value": float(value),
        }
        for point, target, value in zip(points, targets, response, strict=True)
    ]


def _error_rows(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    domain: str,
    response: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, Any]]:
    if domain == "full_domain":
        points = candidate.full_domain_grid
        targets = candidate.full_domain_target
    else:
        points = candidate.actual_singular_values
        targets = candidate.actual_singular_targets
    return [
        {
            "backend_name": backend_name,
            "candidate_name": candidate.candidate_name,
            "evaluation_domain": domain,
            "sigma_normalized": float(point),
            "target_value": float(target),
            "phase_response_value": float(value),
            "phase_response_abs_error": float(error),
        }
        for point, target, value, error in zip(points, targets, response, errors, strict=True)
    ]


def _skip_row(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    reason: str,
    *,
    status: str = "skipped_backend_unavailable",
) -> dict[str, Any]:
    return _summary_stub(backend_name, candidate, status=status, failure_reason=reason)


def _not_needed_row(backend_name: str, candidate: ExternalPhaseCandidate) -> dict[str, Any]:
    return _summary_stub(
        backend_name,
        candidate,
        status="skipped_not_needed_after_external_backend_pass",
        failure_reason="external stable backend already passed full-domain validation",
    )


def _synthesis_failure_row(candidate: ExternalPhaseCandidate, result: Any) -> dict[str, Any]:
    return _summary_stub(
        result.backend_name,
        candidate,
        status=result.status,
        failure_reason=result.error_message or "phase synthesis failed",
    )


def _response_failure_row(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    reason: str,
) -> dict[str, Any]:
    return _summary_stub(
        backend_name,
        candidate,
        status="failed_response_validation",
        failure_reason=reason,
    )


def _summary_stub(
    backend_name: str,
    candidate: ExternalPhaseCandidate,
    *,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "backend_name": backend_name,
        "backend_version": "",
        "candidate_name": candidate.candidate_name,
        "alpha": float(candidate.alpha),
        "degree": int(candidate.degree),
        "input_basis": "",
        "phase_count": 0,
        "phase_convention": "",
        "response_convention": "",
        "native_max_error": float(candidate.native_max_error_full_domain),
        "phase_response_max_error_full_domain": np.nan,
        "phase_response_mean_error_full_domain": np.nan,
        "phase_response_rms_error_full_domain": np.nan,
        "phase_response_max_error_actual_singular_values_if_available": np.nan,
        "grid_size": int(candidate.full_domain_grid.size),
        "actual_singular_value_count_if_available": int(candidate.actual_singular_values.size),
        "passed_1e_minus_3_full_domain": False,
        "passed_1e_minus_3_actual_singular_values": False,
        "status": status,
        "failure_reason": failure_reason,
        "caveat": PHASE_VALIDATION_CAVEAT,
    }


def _phase_report(summary: pd.DataFrame) -> str:
    passed = (
        summary[summary["passed_1e_minus_3_full_domain"] == True]  # noqa: E712
        if not summary.empty
        else pd.DataFrame()
    )
    verdict = "passed" if not passed.empty else "unresolved"
    lines = [
        "# External Backend Target Phase Validation",
        "",
        "## Verdict",
        "",
        f"Target-level full-domain phase validation status: `{verdict}`.",
        "",
    ]
    if not passed.empty:
        best = passed.sort_values("phase_response_max_error_full_domain").iloc[0]
        lines.append(
            "Best passing row: "
            f"`{best.backend_name}` / `{best.candidate_name}` with full-domain error "
            f"`{float(best.phase_response_max_error_full_domain):.6g}`."
        )
    else:
        lines.append("No backend/candidate row passed full-domain `1e-3` validation.")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| backend | candidate | degree | full error | actual-SV error | passed | status |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary.itertuples(index=False):
        full = (
            ""
            if not np.isfinite(row.phase_response_max_error_full_domain)
            else f"{row.phase_response_max_error_full_domain:.6g}"
        )
        actual = (
            ""
            if not np.isfinite(row.phase_response_max_error_actual_singular_values_if_available)
            else f"{row.phase_response_max_error_actual_singular_values_if_available:.6g}"
        )
        lines.append(
            "| "
            f"{row.backend_name} | {row.candidate_name} | {row.degree} | "
            f"{full} | {actual} | {row.passed_1e_minus_3_full_domain} | {row.status} |"
        )
    lines.extend(["", "## Claim Boundary", "", PHASE_VALIDATION_CAVEAT, ""])
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_external_backend_phase_validation",
        "sanity_output_dir": "outputs/qsvt_external_backend_sanity_regression",
        "candidate_config": {},
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run external backend target phase validation")
    parser.parse_args(argv)
    run = run_external_backend_phase_validation()
    print(f"External backend target phase validation complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
