"""Phase 5: minimal baseline coverage extension for LAV, normal-equation WLS, HHL-style proxy.

Runs a small controlled benchmark so the previously legacy/diagnostic-only baselines
(LAV, normal-equation WLS, HHL-style inverse proxy) have real multi-case results, or so
their limitations are recorded explicitly. The HHL-style proxy is an inverse-style proxy,
not a full HHL implementation, and is never claimed as such.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import (
    ALPHA_ESTIMATORS,
    DEFAULT_CASE_SOURCE,
    apply_bad_data,
    apply_missing,
    build_estimator,
    build_system,
    conditioning,
    high_condition_measurement_config,
    solve_detailed,
    subset_spec,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_baseline_coverage_extension.py"
BASELINE_CLAIM = (
    "controlled IEEE benchmark; classical baseline estimator; hhl_style_proxy is an "
    "HHL-style inverse proxy, not full HHL; not field data"
)

ALL_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "estimator",
    "alpha",
    "seed",
    "rmse",
    "weighted_residual_norm",
    "condition_number",
    "converged",
    "runtime_seconds",
    "result_status",
    "failure_reason",
    "source_script",
    "source_artifact",
    "claim_boundary",
    "notes",
]

DEFAULT_CASES = ("ieee14", "ieee57", "ieee118")
DEFAULT_STRESS_TYPES = ("bad_data_heavy", "missing_high_condition", "clean_reference")
DEFAULT_ESTIMATORS = (
    "lav",
    "normal_equation_wls",
    "hhl_style_proxy",
    "ridge_tikhonov",
    "huber_irls",
)
DEFAULT_SEEDS = (0, 1, 2)
FIXED_ALPHA = 1.0e-4
_BASELINES = ("lav", "normal_equation_wls", "hhl_style_proxy")


def build_baseline_coverage_extension(config: dict[str, Any]) -> dict[str, Any]:
    cases = list(config.get("cases", DEFAULT_CASES))
    workflow = str(config.get("workflow", "ac_linearized"))
    stress_types = list(config.get("stress_types", DEFAULT_STRESS_TYPES))
    estimators = list(config.get("estimators", DEFAULT_ESTIMATORS))
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    alpha = float(config.get("alpha", FIXED_ALPHA))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    output_dir = Path(config.get("output_dir", "outputs/baseline_coverage_extension"))
    input_root = Path(config.get("input_root", "outputs"))

    full_config = subset_spec("full_ac_measurement_set").measurement_config
    rows: list[dict[str, Any]] = []
    for case in cases:
        for seed in seeds:
            base = build_system(
                case=case, measurement_config=full_config, seed=seed, case_source=case_source
            )
            high_cond = build_system(
                case=case,
                measurement_config=high_condition_measurement_config(case, case_source=case_source),
                seed=seed,
                case_source=case_source,
            )
            for stress in stress_types:
                system = _realize(stress, base=base, high_condition=high_cond, seed=seed)
                if system is None:
                    continue
                cond = conditioning(system)
                for estimator_name in estimators:
                    detail = solve_detailed(build_estimator(estimator_name, alpha), system)
                    rows.append(
                        _row(case, workflow, stress, estimator_name, alpha, seed, detail, cond)
                    )

    return _write_outputs(
        output_dir=output_dir,
        input_root=input_root,
        rows=rows,
        cases=cases,
        input_config={
            "cases": cases,
            "workflow": workflow,
            "stress_types": stress_types,
            "estimators": estimators,
            "seeds": seeds,
            "alpha": alpha,
            "case_source": case_source,
            "output_dir": str(output_dir),
        },
    )


def _realize(stress: str, *, base: Any, high_condition: Any, seed: int) -> Any:
    if stress == "clean_reference":
        return base
    if stress == "bad_data_heavy":
        return apply_bad_data(base, ratio=0.15, magnitude=10.0, target="random", seed=seed)
    if stress == "missing_high_condition":
        try:
            return apply_missing(high_condition, missing_ratio=0.2, seed=seed)
        except Exception:
            return None
    return None


def _row(
    case: str,
    workflow: str,
    stress: str,
    estimator: str,
    alpha: float,
    seed: int,
    detail: dict[str, Any],
    cond: dict[str, float],
) -> dict[str, Any]:
    return {
        "case": case,
        "workflow": workflow,
        "stress_type": stress,
        "estimator": estimator,
        "alpha": alpha if estimator in ALPHA_ESTIMATORS else "",
        "seed": seed,
        "rmse": detail["rmse"],
        "weighted_residual_norm": detail["weighted_residual_norm"],
        "condition_number": cond["condition_number"],
        "converged": detail["converged"],
        "runtime_seconds": detail["runtime_seconds"],
        "result_status": "failed_with_error" if detail["failed"] else "computed",
        "failure_reason": detail["failure_reason"],
        "source_script": SOURCE_SCRIPT,
        "source_artifact": f"computed:{workflow}:{case}:{stress}:seed{seed}",
        "claim_boundary": BASELINE_CLAIM,
        "notes": "",
    }


def _median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def _summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (case, stress, estimator), group in frame.groupby(
        ["case", "stress_type", "estimator"], sort=False
    ):
        failures = int((group["result_status"] == "failed_with_error").sum())
        rows.append(
            {
                "case": case,
                "stress_type": stress,
                "estimator": estimator,
                "median_rmse": _median(group["rmse"]),
                "median_weighted_residual_norm": _median(group["weighted_residual_norm"]),
                "median_condition_number": _median(group["condition_number"]),
                "failure_rate": failures / len(group) if len(group) else float("nan"),
                "n_trials": len(group),
            }
        )
    return pd.DataFrame(rows)


def _coverage_status(frame: pd.DataFrame, cases: list[str]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    if frame.empty:
        return status
    for estimator in _BASELINES:
        sub = frame[frame["estimator"] == estimator]
        if sub.empty:
            status[estimator] = {"coverage": "no_results", "cases": [], "failure_rate": 1.0}
            continue
        computed = sub[sub["result_status"] == "computed"]
        covered = sorted(set(computed["case"]))
        failure_rate = float((sub["result_status"] == "failed_with_error").mean())
        if set(covered) >= set(cases) and failure_rate < 0.5:
            coverage = (
                "broadly_benchmarked" if estimator == "lav" else "benchmarked_with_limitations"
            )
        elif covered:
            coverage = "diagnostic_only_with_limitations"
        else:
            coverage = "fails_under_stress_diagnostic_only"
        status[estimator] = {
            "coverage": coverage,
            "cases": covered,
            "failure_rate": failure_rate,
        }
    return status


def _robust_beats_ridge_bad_data(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    bad = frame[frame["stress_type"] == "bad_data_heavy"]
    if bad.empty:
        return False
    ridge = _median(bad.loc[bad["estimator"] == "ridge_tikhonov", "rmse"])
    robust = min(_median(bad.loc[bad["estimator"] == est, "rmse"]) for est in ("lav", "huber_irls"))
    return bool(np.isfinite(robust) and np.isfinite(ridge) and robust < ridge)


def _interpretation_markdown(coverage: dict[str, dict[str, Any]], robust_wins: bool) -> str:
    lav = coverage.get("lav", {})
    wls = coverage.get("normal_equation_wls", {})
    hhl = coverage.get("hhl_style_proxy", {})
    lines = [
        "# Baseline Coverage Extension (Phase 5)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Controlled IEEE benchmark, AC-linearized weighted update. The weighted Jacobian "
        "condition number is",
        "",
        r"\[",
        r"\kappa(\tilde H)",
        r"=",
        r"\frac{\sigma_{\max}(\tilde H)}",
        r"{\sigma_{\min}(\tilde H)}.",
        r"\]",
        "",
        "## Interpretation",
        "",
        f"1. **Is LAV broadly benchmarked after this phase?** LAV coverage is "
        f"`{lav.get('coverage', 'no_results')}` across cases {lav.get('cases', [])} "
        "(previously only a legacy IEEE14 result).",
        f"2. **Are normal-equation WLS and HHL-style proxy still diagnostic only?** "
        f"normal_equation_wls is `{wls.get('coverage', 'no_results')}` "
        f"(failure rate {wls.get('failure_rate', float('nan')):.2f}); "
        f"hhl_style_proxy is `{hhl.get('coverage', 'no_results')}` "
        f"(failure rate {hhl.get('failure_rate', float('nan')):.2f}). They remain "
        "conditioning/ablation probes, not headline estimators.",
        "3. **Under bad-data-heavy stress, do robust estimators outperform Ridge?** "
        + (
            "Yes — LAV or Huber attains a lower median RMSE than Ridge under bad-data-heavy "
            "stress (`baseline_coverage_summary.csv`)."
            if robust_wins
            else "Not in this run; see `baseline_coverage_summary.csv`."
        ),
        "4. **Which baselines remain limited and why?** The HHL-style proxy is an inverse-style "
        "proxy (precision/instability model), not a full HHL solver; normal-equation WLS squares "
        "the condition number and is sensitive to ill-conditioning. Neither is claimed beyond a "
        "classical baseline.",
        "",
        "LAV is not claimed as all-case beyond the cases listed above; HHL-style proxy is not "
        "claimed as full HHL.",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    input_root: Path,
    rows: list[dict[str, Any]],
    cases: list[str],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    frame = pd.DataFrame(rows)
    summary = _summary_frame(frame)
    coverage = _coverage_status(frame, cases)
    robust_wins = _robust_beats_ridge_bad_data(frame)

    results_path = output_dir / "baseline_coverage_results.csv"
    summary_path = output_dir / "baseline_coverage_summary.csv"
    interpretation_path = output_dir / "baseline_coverage_interpretation.md"
    missing_path = output_dir / "missing_baseline_coverage_outputs.csv"

    rows_to_table(rows, results_path, ALL_COLUMNS)
    summary.to_csv(summary_path, index=False)
    interpretation_path.write_text(
        _interpretation_markdown(coverage, robust_wins), encoding="utf-8"
    )
    rows_to_table(
        _missing_rows(coverage),
        missing_path,
        ["missing_output", "reason", "result_status"],
    )

    artifacts = {
        "baseline_coverage_results": str(results_path),
        "baseline_coverage_summary": str(summary_path),
        "baseline_coverage_interpretation": str(interpretation_path),
        "missing_baseline_coverage_outputs": str(missing_path),
    }
    paper_dir = _index_into_package(input_root, artifacts)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "paper_dir": paper_dir,
        "rows": rows,
        "summary": summary,
        "coverage": coverage,
        "robust_beats_ridge_bad_data": robust_wins,
        "artifacts": artifacts,
        "frame": frame,
    }


def _missing_rows(coverage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for estimator, info in coverage.items():
        if info["coverage"] not in ("broadly_benchmarked", "benchmarked_with_limitations"):
            missing.append(
                {
                    "missing_output": f"{estimator} robust all-case coverage",
                    "reason": (
                        f"coverage={info['coverage']}; failure_rate={info['failure_rate']:.2f}"
                    ),
                    "result_status": "diagnostic_only",
                }
            )
    missing.append(
        {
            "missing_output": "full HHL solver (vs HHL-style proxy)",
            "reason": "not_implemented_proxy_only",
            "result_status": "not_applicable",
        }
    )
    return missing


def _index_into_package(input_root: Path, artifacts: dict[str, str]) -> Path | None:
    package = input_root / "final_manuscript_package" / "phase2_baseline_coverage_extension"
    try:
        ensure_directory(package)
    except Exception:
        return None
    for key in (
        "baseline_coverage_summary",
        "baseline_coverage_interpretation",
        "missing_baseline_coverage_outputs",
    ):
        target = package / Path(artifacts[key]).name
        try:
            target.write_text(Path(artifacts[key]).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
    return package
