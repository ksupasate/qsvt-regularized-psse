"""Phase 2 hardening: nonlinear AC per-iteration logging refresh.

Runs a small, focused nonlinear AC job with per-iteration RMSE and weighted-Jacobian
condition-number logging enabled, producing an ``iteration_trace.csv`` that records the
state RMSE and the iteration-Jacobian condition number per step. This is a fast refresh
on a small case (real computation, nothing fabricated); older logs that predate the
logging are left untouched and remain honestly marked as not recording these metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import (
    build_ac_nonlinear_problem,
    run_iterative_estimators,
)
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import (
    DEFAULT_CASE_SOURCE,
    HUBER_DELTA,
    HUBER_MAX_ITER,
    HUBER_TOL,
    PAPER_TO_INTERNAL,
    TSVD_TAU,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_nonlinear_logging_refresh.py"
_LOGGER = logging.getLogger("nonlinear_logging_refresh")
_LOGGER.addHandler(logging.NullHandler())

DEFAULT_CASES = ("ieee14",)
DEFAULT_ESTIMATORS = ("ridge_tikhonov", "truncated_svd", "qsvt_target_classical")
DEFAULT_SEEDS = (10,)
DEFAULT_ALPHA = 1.0e-4
DEFAULT_STRESS_TYPES = ("clean_or_noise",)
# Stress profiles applied to the generated measurements (controlled IEEE benchmark only).
_STRESS_PROFILES = {
    "clean_or_noise": {"noise_std": 0.01, "missing_ratio": 0.0, "bad_data_ratio": 0.0},
    "missing_20_percent": {"noise_std": 0.01, "missing_ratio": 0.2, "bad_data_ratio": 0.0},
    "bad_data_5_percent": {"noise_std": 0.01, "missing_ratio": 0.0, "bad_data_ratio": 0.05},
}
_MAX_ITERATIONS = 8
# Total wall-clock budget. Remaining (case, seed, stress) combinations are recorded as
# ``runtime_limited`` once the budget is exhausted rather than skipped silently.
_MAX_TOTAL_SECONDS = 900.0

# Columns recorded by the logging-enabled trace. The first block matches the legacy
# iteration_trace schema; the second block is the newly logged per-iteration metrics.
TRACE_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "estimator",
    "alpha",
    "seed",
    "iteration",
    "weighted_residual_before",
    "weighted_residual_after",
    "update_norm",
    "state_rmse",
    "angle_rmse",
    "voltage_rmse",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "numerical_rank",
    "effective_rank",
    "converged_so_far",
    "failed",
    "converged",
    "trial_id",
    "sweep_name",
    "sweep_parameter",
    "sweep_value",
    "source_script",
]
RUNTIME_LIMITED_COLUMNS = [
    "case",
    "seed",
    "stress_type",
    "reason",
    "elapsed_seconds",
    "result_status",
]


def build_nonlinear_logging_refresh(config: dict[str, Any]) -> dict[str, Any]:
    cases = list(config.get("cases", DEFAULT_CASES))
    estimators = list(config.get("estimators", DEFAULT_ESTIMATORS))
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    alpha = float(config.get("alpha", DEFAULT_ALPHA))
    stress_types = list(config.get("stress_types", DEFAULT_STRESS_TYPES))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    max_total_seconds = float(config.get("max_total_seconds", _MAX_TOTAL_SECONDS))
    output_dir = Path(config.get("output_dir", "outputs/nonlinear_logging_refresh"))
    ensure_directory(output_dir)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    runtime_limited: list[dict[str, Any]] = []
    start = time.time()
    for case in cases:
        for seed in seeds:
            for stress in stress_types:
                elapsed_total = time.time() - start
                if elapsed_total > max_total_seconds:
                    runtime_limited.append(
                        _runtime_limited_row(
                            case, seed, stress, "wall-clock budget exhausted", elapsed_total
                        )
                    )
                    continue
                frame, elapsed, error = _refresh_trace(
                    case, case_source, seed, estimators, alpha, stress
                )
                if frame is None:
                    failures.append(
                        {
                            "case": case,
                            "seed": seed,
                            "stress_type": stress,
                            "reason": error or "problem_build_or_run_failed",
                        }
                    )
                    continue
                frames.append(frame)
                if elapsed > max_total_seconds:
                    # The run completed but exceeded the budget; flag it for transparency.
                    runtime_limited.append(
                        _runtime_limited_row(
                            case, seed, stress, "single-run exceeded budget", elapsed
                        )
                    )

    trace = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACE_COLUMNS)
    return _write_outputs(
        output_dir=output_dir,
        trace=trace,
        failures=failures,
        runtime_limited=runtime_limited,
        requested=[
            (case, seed, stress) for case in cases for seed in seeds for stress in stress_types
        ],
        input_config={
            "cases": cases,
            "estimators": estimators,
            "seeds": seeds,
            "alpha": alpha,
            "stress_types": stress_types,
            "case_source": case_source,
            "max_iterations": _MAX_ITERATIONS,
            "max_total_seconds": max_total_seconds,
            "log_iteration_metrics": True,
            "log_condition_number": True,
            "output_dir": str(output_dir),
        },
    )


def _runtime_limited_row(
    case: str, seed: int, stress: str, reason: str, elapsed: float
) -> dict[str, Any]:
    return {
        "case": case,
        "seed": seed,
        "stress_type": stress,
        "reason": reason,
        "elapsed_seconds": round(float(elapsed), 2),
        "result_status": "runtime_limited",
    }


def _refresh_trace(
    case: str, case_source: str, seed: int, estimators: list[str], alpha: float, stress: str
) -> tuple[pd.DataFrame | None, float, str]:
    cfg = _config(case, case_source, seed, estimators, alpha, stress)
    start = time.time()
    try:
        problem = build_ac_nonlinear_problem(cfg)
        _, trace = run_iterative_estimators(config=cfg, problem=problem, logger=_LOGGER)
    except Exception as exc:
        return None, time.time() - start, f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start
    if trace.empty:
        return None, elapsed, "empty_trace"
    trace = trace.copy()
    trace["case"] = case
    trace["workflow"] = "nonlinear_ac_iterative"
    trace["stress_type"] = stress
    trace["seed"] = seed
    trace["sweep_name"] = stress
    trace["sweep_parameter"] = "iteration"
    trace["sweep_value"] = ""
    trace["trial_id"] = f"{case}_seed{seed}_{stress}"
    trace["source_script"] = SOURCE_SCRIPT
    alpha_by_internal = {PAPER_TO_INTERNAL[name]: alpha for name in estimators}
    trace["alpha"] = trace["estimator"].map(lambda name: alpha_by_internal.get(str(name), ""))
    return trace, elapsed, ""


def _config(
    case: str, case_source: str, seed: int, estimators: list[str], alpha: float, stress: str
) -> dict[str, Any]:
    return {
        "run_name": f"nonlinear_logging_refresh_{case}_{seed}_{stress}",
        "seed": int(seed),
        "system": {
            "case_name": case,
            "case_source": case_source,
            "mode": "nonlinear_ac_state_estimation",
            "measurement": {
                "include_voltage_magnitudes": True,
                "include_p_injections": True,
                "include_q_injections": True,
                "include_p_branch_flows": True,
                "include_q_branch_flows": True,
                "voltage_std": 0.01,
                "injection_p_std": 0.03,
                "injection_q_std": 0.03,
                "flow_p_std": 0.02,
                "flow_q_std": 0.02,
            },
            "linearization": {
                "angle_perturbation_std": 0.005,
                "voltage_perturbation_std": 0.005,
                "min_voltage_magnitude": 0.5,
            },
            "iteration": {
                "max_iterations": _MAX_ITERATIONS,
                "update_tolerance": 1.0e-7,
                "residual_tolerance": 1.0e-7,
                "damping": 1.0,
                "max_update_norm": 1000.0,
                "residual_growth_limit": 10000.0,
                "log_iteration_metrics": True,
                "log_condition_number": True,
            },
        },
        "scenario": _scenario(stress),
        "estimators": [_estimator_config(name, alpha) for name in estimators],
        "output": {
            "root": "outputs",
            "run_id": f"nonlinear_logging_refresh_{case}_{seed}_{stress}",
        },
    }


def _scenario(stress: str) -> dict[str, Any]:
    profile = _STRESS_PROFILES.get(stress, _STRESS_PROFILES["clean_or_noise"])
    return {
        "name": stress,
        "noise_std": profile["noise_std"],
        "missing_ratio": profile["missing_ratio"],
        "bad_data": {
            "enabled": profile["bad_data_ratio"] > 0.0,
            "ratio": profile["bad_data_ratio"],
            "magnitude": 10.0,
            "target": "random",
        },
    }


def _estimator_config(paper_name: str, alpha: float) -> dict[str, Any]:
    internal = PAPER_TO_INTERNAL[paper_name]
    if internal == "truncated_svd":
        return {"name": "truncated_svd", "tau": TSVD_TAU}
    if internal == "huber_irls":
        return {
            "name": "huber_irls",
            "delta": HUBER_DELTA,
            "max_iterations": HUBER_MAX_ITER,
            "tolerance": HUBER_TOL,
        }
    if internal == "ridge":
        return {"name": "ridge", "alpha": alpha}
    if internal == "qsvt_regularized":
        return {"name": "qsvt_regularized", "alpha": alpha}
    if internal == "pseudoinverse":
        return {"name": "pseudoinverse"}
    raise ValueError(f"unsupported logging-refresh estimator: {paper_name}")


def _summary_markdown(
    trace: pd.DataFrame,
    failures: list[dict[str, Any]],
    runtime_limited: list[dict[str, Any]],
) -> str:
    cases = sorted(trace["case"].astype(str).unique()) if not trace.empty else []
    stresses = (
        sorted(trace["stress_type"].astype(str).unique())
        if not trace.empty and "stress_type" in trace.columns
        else []
    )
    rmse_logged = "state_rmse" in trace.columns and trace["state_rmse"].notna().any()
    kappa_logged = "condition_number" in trace.columns and trace["condition_number"].notna().any()
    limited = sorted({f"{r['case']}/{r['stress_type']}" for r in runtime_limited})
    lines = [
        "# Nonlinear AC Logging Refresh",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "A small, focused nonlinear AC run with per-iteration logging enabled. Each iteration "
        "rebuilds the residual and Jacobian,",
        "",
        r"\[",
        r"r_k = z - h(x_k),",
        r"\qquad",
        r"H_k = \left.\frac{\partial h}{\partial x}\right|_{x=x_k},",
        r"\]",
        "",
        "and now also records the state RMSE and the weighted-Jacobian condition number "
        r"\(\kappa(\tilde H_k)\) per step.",
        "",
        "## What was logged",
        f"- Cases refreshed: {', '.join(cases) or 'none'}.",
        f"- Stress types: {', '.join(stresses) or 'none'}.",
        f"- Trace rows: {len(trace)}.",
        f"- Per-iteration state RMSE recorded: {'yes' if rmse_logged else 'no'}.",
        f"- Per-iteration condition number recorded: {'yes' if kappa_logged else 'no'}.",
        f"- Runs that failed to build/run: {len(failures)}.",
        f"- Runtime-limited (case/stress): {', '.join(limited) or 'none'}.",
        "",
        "## Scope",
        "- This refresh is real computation on small controlled IEEE cases; it is nonlinear AC "
        "consistency evidence, not a nonlinear QSVT solver. QSVT-target classical equals Ridge for "
        "matched alpha.",
        "- Older logs that predate this logging are not modified or backfilled; their "
        "per-iteration RMSE / condition number remain marked `not_recorded_by_current_solver`.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    trace: pd.DataFrame,
    failures: list[dict[str, Any]],
    runtime_limited: list[dict[str, Any]],
    requested: list[tuple[str, int, str]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ordered = pd.DataFrame(columns=TRACE_COLUMNS)
    if not trace.empty:
        for column in TRACE_COLUMNS:
            ordered[column] = trace[column] if column in trace.columns else ""
    trace_path = output_dir / "iteration_trace.csv"
    ordered.to_csv(trace_path, index=False)
    summary_path = output_dir / "nonlinear_logging_refresh_summary.md"
    summary_path.write_text(_summary_markdown(trace, failures, runtime_limited), encoding="utf-8")
    runtime_path = rows_to_table(
        runtime_limited, output_dir / "runtime_limited_cases.csv", RUNTIME_LIMITED_COLUMNS
    )

    artifacts = {
        "iteration_trace": str(trace_path),
        "nonlinear_logging_refresh_summary": str(summary_path),
        "runtime_limited_cases": str(runtime_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    rmse_logged = bool(
        not trace.empty and "state_rmse" in trace.columns and trace["state_rmse"].notna().any()
    )
    kappa_logged = bool(
        not trace.empty
        and "condition_number" in trace.columns
        and trace["condition_number"].notna().any()
    )
    computed_cases = set(trace["case"].astype(str).unique()) if not trace.empty else set()
    limited_cases = {r["case"] for r in runtime_limited}
    return {
        "output_dir": output_dir,
        "trace": ordered,
        "n_rows": len(ordered),
        "rmse_logged": rmse_logged,
        "kappa_logged": kappa_logged,
        "failures": failures,
        "runtime_limited": runtime_limited,
        "computed_cases": sorted(computed_cases),
        "limited_cases": sorted(limited_cases),
        "requested": requested,
        "artifacts": artifacts,
    }
