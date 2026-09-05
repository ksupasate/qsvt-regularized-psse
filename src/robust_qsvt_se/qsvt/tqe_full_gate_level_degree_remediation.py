from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    FULL_GATE_LEVEL_COVERAGE_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_full_gate_level_qsvt_coverage import (
    coverage_row_from_integrated_evaluation,
)
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    DEFAULT_BASIS_GATES,
    run_ieee_selected_block,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")

REMEDIATION_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "alpha",
    "epsilon_target",
    "original_degree",
    "candidate_degree",
    "phase_count",
    "phase_synthesis_status",
    "qsvt_circuit_status",
    "simulation_status",
    "transpilation_status",
    "num_qubits",
    "num_U_calls",
    "num_U_dagger_calls",
    "num_phase_rotations",
    "raw_depth",
    "transpiled_depth",
    "transpiled_cx_count",
    "transpiled_total_ops",
    "transform_block_fro_error",
    "transform_block_spectral_error",
    "circuit_vs_polynomial_fro_error",
    "circuit_vs_ridge_relative_update_error",
    "absolute_update_error",
    "residual_gap",
    "success_probability",
    "meets_transform_criterion",
    "meets_update_criterion",
    "meets_residual_criterion",
    "meets_stop_criteria",
    "case_final_status",
    "runtime_seconds",
    "failure_or_skip_reason",
]


def run_full_gate_level_degree_remediation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / FULL_GATE_LEVEL_COVERAGE_DIR)
    reports_dir = paths["reports"]

    forensic = pd.read_csv(resolved["forensic_rows_path"])
    targets = _target_rows(forensic)
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        rows.extend(_remediate_one_target(target, resolved))
    results = pd.DataFrame(rows, columns=REMEDIATION_COLUMNS)
    results = _assign_final_status(results)

    artifacts = {
        "remediation_csv": output_dir / "full_gate_level_qsvt_degree_remediation.csv",
        "remediation_report": reports_dir / "full_gate_level_qsvt_degree_remediation_report.md",
    }
    results.to_csv(artifacts["remediation_csv"], index=False)
    artifacts["remediation_report"].write_text(
        _report_markdown(config=resolved, results=results, artifacts=artifacts),
        encoding="utf-8",
    )

    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts={key: str(value) for key, value in artifacts.items()},
    )
    write_json(output_dir / "full_gate_level_qsvt_degree_remediation_metadata.json", metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: path for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results,
        "artifacts": artifacts,
    }


def _remediate_one_target(target: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_degree in config["degree_grid"]:
        started = time.perf_counter()
        row = _evaluate_candidate(target, config, int(candidate_degree))
        row["runtime_seconds"] = float(time.perf_counter() - started)
        rows.append(row)
        if bool(row["meets_stop_criteria"]):
            break
    return rows


def _evaluate_candidate(
    target: Any,
    config: dict[str, Any],
    candidate_degree: int,
) -> dict[str, Any]:
    try:
        evaluation = run_ieee_selected_block(
            {
                "seed": int(config["seed"]),
                "subproblem_spec": {
                    "case_name": str(target.case_name),
                    "subproblem_size": int(target.subproblem_size),
                    "selection_mode": str(target.selection_criterion),
                },
                "alpha": float(target.alpha),
                "epsilon_target": float(target.epsilon_target),
                "degree": int(candidate_degree),
                "angle_solver": str(config["angle_solver"]),
                "basis_gates": list(config["basis_gates"]),
                "transpile_qubit_limit": int(config["transpile_qubit_limit"]),
                "transpile_optimization_level": int(config["transpile_optimization_level"]),
                "block_results_path": str(config["block_results_path"]),
                "block_matrices_dir": str(config["block_matrices_dir"]),
                "end_to_end_results_path": str(config["end_to_end_results_path"]),
                "artifact_match_rtol": float(config["artifact_match_rtol"]),
                "artifact_match_atol": float(config["artifact_match_atol"]),
            }
        )
        coverage = coverage_row_from_integrated_evaluation(
            evaluation=evaluation,
            tier="degree_remediation",
            selection_criterion=str(target.selection_criterion),
            degree_selection=_DegreeStub(candidate_degree),
        )
        row = _row_from_coverage(target, coverage)
    except Exception as exc:  # pragma: no cover - defensive branch
        row = _empty_candidate_row(target, candidate_degree)
        row.update(
            {
                "phase_synthesis_status": "not_completed",
                "qsvt_circuit_status": "failed",
                "simulation_status": "failed",
                "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return _apply_stop_criteria(row, config)


def _row_from_coverage(target: Any, coverage: dict[str, Any]) -> dict[str, Any]:
    row = _empty_candidate_row(target, int(coverage["degree"]))
    row.update(
        {
            "phase_count": coverage["phase_count"],
            "phase_synthesis_status": coverage["phase_synthesis_status"],
            "qsvt_circuit_status": coverage["qsvt_circuit_status"],
            "simulation_status": coverage["simulation_status"],
            "transpilation_status": coverage["transpilation_status"],
            "num_qubits": coverage["num_qubits"],
            "num_U_calls": coverage["num_U_calls"],
            "num_U_dagger_calls": coverage["num_U_dagger_calls"],
            "num_phase_rotations": coverage["num_phase_rotations"],
            "raw_depth": coverage["raw_depth"],
            "transpiled_depth": coverage["transpiled_depth"],
            "transpiled_cx_count": coverage["transpiled_cx_count"],
            "transpiled_total_ops": coverage["transpiled_total_ops"],
            "transform_block_fro_error": coverage["transform_block_fro_error"],
            "transform_block_spectral_error": coverage["transform_block_spectral_error"],
            "circuit_vs_polynomial_fro_error": coverage["circuit_vs_polynomial_fro_error"],
            "circuit_vs_ridge_relative_update_error": coverage[
                "circuit_vs_ridge_relative_update_error"
            ],
            "absolute_update_error": coverage["absolute_update_error"],
            "residual_gap": coverage["residual_gap"],
            "success_probability": coverage["success_probability"],
            "failure_or_skip_reason": coverage["failure_or_skip_reason"],
        }
    )
    return row


def _apply_stop_criteria(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    transform = _finite_leq(row["circuit_vs_polynomial_fro_error"], config["transform_tolerance"])
    relative = _finite_leq(
        row["circuit_vs_ridge_relative_update_error"],
        config["relative_update_tolerance"],
    )
    absolute = _finite_leq(row["absolute_update_error"], config["absolute_update_tolerance"])
    residual = _finite_leq(row["residual_gap"], config["residual_gap_tolerance"])
    row.update(
        {
            "meets_transform_criterion": bool(transform),
            "meets_update_criterion": bool(relative or absolute),
            "meets_residual_criterion": bool(residual),
            "meets_stop_criteria": bool(transform and (relative or absolute) and residual),
        }
    )
    return row


def _assign_final_status(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    frame = results.copy()
    for key, group in frame.groupby(["case_name", "subproblem_size"], dropna=False):
        mask = (frame["case_name"] == key[0]) & (frame["subproblem_size"] == key[1])
        if group["meets_stop_criteria"].astype(bool).any():
            status = "remediated_by_higher_degree"
        elif (group["phase_synthesis_status"] == "failed").all():
            status = "failed_phase_synthesis"
        elif group["simulation_status"].astype(str).str.contains("skipped_by_budget").all():
            status = "skipped_by_budget"
        else:
            status = "still_failed_degree_boundary"
        frame.loc[mask, "case_final_status"] = status
    return frame


def _target_rows(forensic: pd.DataFrame) -> pd.DataFrame:
    targets = {
        ("ieee118", 4),
        ("ieee14", 8),
        ("ieee57", 8),
    }
    mask = [
        (str(row.case_name), int(row.subproblem_size)) in targets
        for row in forensic.itertuples(index=False)
    ]
    return forensic[mask].copy()


def _empty_candidate_row(target: Any, candidate_degree: int) -> dict[str, Any]:
    return {
        "case_name": str(target.case_name),
        "subproblem_size": int(target.subproblem_size),
        "selection_criterion": str(target.selection_criterion),
        "alpha": float(target.alpha),
        "epsilon_target": float(target.epsilon_target),
        "original_degree": int(target.degree),
        "candidate_degree": int(candidate_degree),
        "phase_count": 0,
        "phase_synthesis_status": "not_attempted",
        "qsvt_circuit_status": "not_attempted",
        "simulation_status": "not_attempted",
        "transpilation_status": "not_attempted",
        "num_qubits": np.nan,
        "num_U_calls": 0,
        "num_U_dagger_calls": 0,
        "num_phase_rotations": 0,
        "raw_depth": np.nan,
        "transpiled_depth": np.nan,
        "transpiled_cx_count": np.nan,
        "transpiled_total_ops": np.nan,
        "transform_block_fro_error": np.nan,
        "transform_block_spectral_error": np.nan,
        "circuit_vs_polynomial_fro_error": np.nan,
        "circuit_vs_ridge_relative_update_error": np.nan,
        "absolute_update_error": np.nan,
        "residual_gap": np.nan,
        "success_probability": np.nan,
        "meets_transform_criterion": False,
        "meets_update_criterion": False,
        "meets_residual_criterion": False,
        "meets_stop_criteria": False,
        "case_final_status": "not_assigned",
        "runtime_seconds": 0.0,
        "failure_or_skip_reason": "",
    }


def _report_markdown(
    *,
    config: dict[str, Any],
    results: pd.DataFrame,
    artifacts: dict[str, Path],
) -> str:
    lines = [
        "# Full Gate-Level QSVT Degree Remediation Report",
        "",
        "## Scope",
        "",
        "Flagged update-level mismatches were audited with higher-degree QSVT "
        "polynomials. Successful remediation indicates degree adequacy rather than "
        "circuit-convention failure; unresolved rows are reported as dense-circuit "
        "feasibility boundaries.",
        "",
        "No full coverage rerun was performed. This pass only evaluates IEEE118 4x4, "
        "IEEE14 8x8, and IEEE57 8x8 from the forensic true-mismatch set.",
        "",
        "## Configuration",
        "",
        f"- Command: `{current_command()}`",
        f"- Degree grid: `{config['degree_grid']}`",
        f"- Relative update tolerance: `{config['relative_update_tolerance']}`",
        f"- Absolute small-norm update tolerance: `{config['absolute_update_tolerance']}`",
        f"- Residual gap tolerance: `{config['residual_gap_tolerance']}`",
        f"- Transform tolerance: `{config['transform_tolerance']}`",
        "",
        "## Case Status",
        "",
    ]
    for (case_name, size), group in results.groupby(["case_name", "subproblem_size"]):
        status = str(group["case_final_status"].iloc[0])
        last = group.iloc[-1]
        lines.extend(
            [
                f"### {case_name} {int(size)}x{int(size)}",
                "",
                f"- Final status: `{status}`",
                f"- Candidate degrees attempted: `{list(group['candidate_degree'].astype(int))}`",
                f"- Last phase/simulation status: `{last.phase_synthesis_status}` / "
                f"`{last.simulation_status}`",
                "- Best relative update error: "
                f"`{_series_min(group['circuit_vs_ridge_relative_update_error'])}`",
                f"- Best residual gap: `{_series_min(group['residual_gap'])}`",
                f"- Best circuit-vs-polynomial Frobenius error: "
                f"`{_series_min(group['circuit_vs_polynomial_fro_error'])}`",
                f"- Reason from final attempted row: `{last.failure_or_skip_reason}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "This degree-remediation audit does not demonstrate full IEEE-scale QSVT "
            "execution, quantum speedup, hardware execution, scalable sparse-oracle "
            "construction, or QSVT numerical superiority over matched Ridge/Tikhonov.",
            "",
            "## Artifacts",
            "",
            f"- Remediation CSV: `{artifacts['remediation_csv']}`",
            f"- Remediation report: `{artifacts['remediation_report']}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )
    return "\n".join(lines)


def _series_min(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"{values.min():.3e}"


def _finite_leq(value: Any, threshold: float) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(np.isfinite(numeric) and numeric <= float(threshold))


class _DegreeStub:
    def __init__(self, degree: int) -> None:
        self.degree = int(degree)
        self.source = "degree_remediation_candidate"
        self.target_met = False
        self.reason = ""


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 123,
        "degree_grid": [11, 25, 35, 50, 75, 101, 151, 201],
        "angle_solver": "root-finding",
        "basis_gates": DEFAULT_BASIS_GATES,
        "transpile_qubit_limit": 3,
        "transpile_optimization_level": 1,
        "artifact_match_rtol": 1.0e-9,
        "artifact_match_atol": 1.0e-8,
        "transform_tolerance": 1.0e-10,
        "relative_update_tolerance": 1.0e-2,
        "absolute_update_tolerance": 1.0e-6,
        "residual_gap_tolerance": 1.0e-2,
        "forensic_rows_path": str(
            root / FULL_GATE_LEVEL_COVERAGE_DIR / "full_gate_level_qsvt_forensic_flagged_rows.csv"
        ),
        "block_results_path": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "block_matrices_dir": str(root / "explicit_block_encoding_demo" / "matrices"),
        "end_to_end_results_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_results.csv"
        ),
    }
    if config:
        resolved.update(config)
    resolved["degree_grid"] = [int(value) for value in resolved["degree_grid"]]
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run full gate-level QSVT degree remediation")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = run_full_gate_level_degree_remediation({"output_root": args.output_root})
    print(f"Wrote full gate-level QSVT degree remediation to {run['output_dir']}")


if __name__ == "__main__":
    main()
