from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json


def build_phase_and_multicase_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    convention = _read_csv(resolved["convention_summary_path"])
    sanity = _read_csv(resolved["sanity_results_path"])
    optional_phase = _read_csv(resolved["optional_phase_summary_path"])
    multicase = _read_csv(resolved["adaptive_multicase_summary_path"])

    rows = _summary_rows(convention, sanity, optional_phase, multicase)
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "phase_and_multicase_summary.csv"
    json_path = output_dir / "phase_and_multicase_summary.json"
    md_path = output_dir / "phase_and_multicase_summary.md"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    md_path.write_text(
        _summary_markdown(convention, sanity, optional_phase, multicase),
        encoding="utf-8",
    )
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "phase_and_multicase_summary_md": str(md_path),
            "phase_and_multicase_summary_csv": str(csv_path),
            "phase_and_multicase_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "phase_and_multicase_summary_md": md_path,
            "phase_and_multicase_summary_csv": csv_path,
            "phase_and_multicase_summary_json": json_path,
            "manifest": manifest_path,
        },
    }


def _summary_rows(
    convention: pd.DataFrame,
    sanity: pd.DataFrame,
    optional_phase: pd.DataFrame,
    multicase: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not convention.empty:
        ridge = convention[convention["target_type"] == "ridge_tikhonov_bounded_target"]
        if not ridge.empty:
            best = ridge.sort_values("max_pointwise_error").iloc[0]
            rows.append(
                {
                    "summary_area": "phase_response_convention",
                    "item": "best_ridge_convention",
                    "status": str(best["status"]),
                    "max_error": _json_scalar(best["max_pointwise_error"]),
                    "degree": _json_scalar(best["degree"]),
                    "query_count": 2 * int(best["degree"]) + 1,
                    "details": _convention_label(best),
                }
            )
    if not sanity.empty:
        rows.append(
            {
                "summary_area": "phase_response_convention",
                "item": "sanity_polynomial_max_best_error",
                "status": "passed" if (sanity["best_status"] == "passed").all() else "failed",
                "max_error": _json_scalar(sanity["best_max_pointwise_error"].max()),
                "degree": "",
                "query_count": "",
                "details": "known polynomial convention sanity tests",
            }
        )
    if not optional_phase.empty:
        for row in optional_phase.itertuples():
            rows.append(
                {
                    "summary_area": "optional_phase_synthesis",
                    "item": f"alpha_{row.alpha}",
                    "status": str(row.status),
                    "max_error": _json_scalar(row.max_pointwise_error),
                    "degree": _json_scalar(row.degree),
                    "query_count": _json_scalar(row.query_count_estimate),
                    "details": getattr(row, "convention", ""),
                }
            )
    if not multicase.empty:
        for row in multicase.itertuples():
            rows.append(
                {
                    "summary_area": "adaptive_multicase_degree_search",
                    "item": row.case_name,
                    "status": str(row.status),
                    "max_error": _json_scalar(row.achieved_max_error),
                    "degree": _json_scalar(row.selected_degree),
                    "query_count": _json_scalar(row.selected_query_count),
                    "details": row.failure_reason_if_any,
                }
            )
    return rows


def _summary_markdown(
    convention: pd.DataFrame,
    sanity: pd.DataFrame,
    optional_phase: pd.DataFrame,
    multicase: pd.DataFrame,
) -> str:
    sanity_text = _sanity_markdown(sanity)
    convention_text = _convention_markdown(convention)
    phase_text = _optional_phase_markdown(optional_phase)
    multicase_text = _multicase_markdown(multicase)
    passed_cases = (
        ", ".join(multicase[multicase["status"] == "passed"]["case_name"].astype(str))
        if not multicase.empty
        else "none"
    )
    failed_cases = (
        ", ".join(multicase[multicase["status"] != "passed"]["case_name"].astype(str))
        if not multicase.empty
        else "none"
    )
    return f"""# QSVT Phase and Adaptive Multicase Summary

## Executive Summary

Phase-response convention diagnostics identify the PennyLane `RX`/`PCPhase`
convention with `real(U[0,0])` as the sanity-polynomial-valid response
convention. For the bounded Ridge/Tikhonov target, phase-level validation is
reported separately from polynomial approximation success.

Adaptive multicase degree search quantifies the degree/query pressure needed to
reach a `1e-3` bounded polynomial approximation target on controlled
IEEE/PYPOWER weighted matrices. These results support feasibility discussion
only and do not imply quantum advantage.

## What Was Fixed Or Not Fixed

The previous failed phase-response validation was caused by using a scalar
response convention that did not match PennyLane's documented `QSVT` convention.
The corrected diagnostic uses PennyLane-style `RX(2 arccos x)`, `PCPhase`, and
`real(U[0,0])`. Sanity polynomials pass under that convention. The bounded
Ridge/Tikhonov target phase response is still limited by the highest stable
phase-synthesizable polynomial degree in monomial basis.

## Sanity Polynomial Validation Status

{sanity_text}

## Best Phase-Response Convention

{convention_text}

## Optional Phase-Synthesis Validation Status

{phase_text}

## Adaptive Multicase Degree-Search Results

{multicase_text}

Cases passing `1e-3`: {passed_cases}

Cases not passing `1e-3`: {failed_cases}

## Degree-Query-Resource Trade-Off

The query-count proxy is `2 * degree + 1`. Larger cases require higher degree
under the same odd minimax approximation method, and IEEE300 remains above the
strict `1e-3` tolerance within the configured resource-safe search cap.

## Safe Claims

- QSVT-compatible approximation diagnostics.
- Phase-response convention validation.
- Bounded polynomial approximation and adaptive degree search.
- Resource-aware feasibility evidence on controlled IEEE/PYPOWER matrices.

## Claims To Avoid

- Quantum speedup or quantum advantage.
- Full quantum hardware execution or hardware validation.
- Real PMU/SCADA field-data validation.
- QSVT numerical superiority over Ridge/Tikhonov under the same alpha.

## Remaining Limitations

- High-degree minimax polynomials can be stable in Chebyshev form but unstable
  when converted to monomial coefficients for PennyLane phase synthesis.
- Optional phase synthesis is not full hardware execution.
- Query-count estimates exclude oracle construction, state preparation,
  compilation, error correction, and readout.

## Recommended Manuscript Wording

We diagnose the QSP/QSVT phase-response convention and identify a PennyLane
`RX`/`PCPhase` convention that passes known-polynomial sanity checks. For the
bounded Ridge/Tikhonov target, polynomial approximation and phase-response
validation are reported separately: the bounded polynomial diagnostics can meet
strict tolerances at sufficiently high degree, while phase-level validation is
limited by stable monomial-basis phase synthesis. Adaptive multi-case searches
quantify the degree and query-count pressure for IEEE/PYPOWER benchmark
matrices without claiming speedup, hardware execution, or superiority over
Ridge/Tikhonov.
"""


def _sanity_markdown(sanity: pd.DataFrame) -> str:
    if sanity.empty:
        return "No sanity-polynomial rows were generated."
    return "\n".join(
        (
            f"- `{row.polynomial_name}`: `{row.best_status}`, max error "
            f"`{row.best_max_pointwise_error:.6g}`"
        )
        for row in sanity.itertuples()
    )


def _convention_markdown(convention: pd.DataFrame) -> str:
    if convention.empty:
        return "No phase-response convention rows were generated."
    ridge = convention[convention["target_type"] == "ridge_tikhonov_bounded_target"]
    if ridge.empty:
        return "No bounded Ridge/Tikhonov target convention rows were generated."
    passed = ridge[ridge["status"] == "passed"]
    if passed.empty:
        best = ridge.sort_values("max_pointwise_error").iloc[0]
        return (
            "No tested convention passed the bounded Ridge/Tikhonov target tolerance. "
            "Best row: "
            f"`{_convention_label(best)}`, max error "
            f"`{float(best['max_pointwise_error']):.6g}`."
        )
    best = passed.sort_values("max_pointwise_error").iloc[0]
    return (
        f"Best passing convention: `{_convention_label(best)}`, max error "
        f"`{float(best['max_pointwise_error']):.6g}`."
    )


def _optional_phase_markdown(optional_phase: pd.DataFrame) -> str:
    if optional_phase.empty:
        return "Optional phase-synthesis validation has not been generated."
    return "\n".join(
        (
            f"- alpha `{row.alpha}` degree `{row.degree}`: `{row.status}`, "
            f"max error `{row.max_pointwise_error:.6g}`"
        )
        for row in optional_phase.itertuples()
    )


def _multicase_markdown(multicase: pd.DataFrame) -> str:
    if multicase.empty:
        return "Adaptive multicase degree search has not been generated."
    return "\n".join(
        (
            f"- `{row.case_name}`: `{row.status}`, degree `{row.selected_degree}`, "
            f"query count `{row.selected_query_count}`, max error "
            f"`{row.achieved_max_error:.6g}`"
        )
        for row in multicase.itertuples()
    )


def _convention_label(row: Any) -> str:
    return (
        f"{row['phase_order']}/{row['phase_sign']}/{row['phase_offset_rule']}/"
        f"{row['signal_operator_convention']}/{row['response_component']}"
    )


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase_and_multicase_summary",
        "convention_summary_path": (
            "outputs/qsvt_phase_response_convention_diagnostics/convention_search_summary.csv"
        ),
        "sanity_results_path": (
            "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
        ),
        "optional_phase_summary_path": (
            "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv"
        ),
        "adaptive_multicase_summary_path": (
            "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv"
        ),
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT phase and multicase summary")
    parser.parse_args(argv)
    run = build_phase_and_multicase_summary()
    print(f"QSVT phase and multicase summary complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
