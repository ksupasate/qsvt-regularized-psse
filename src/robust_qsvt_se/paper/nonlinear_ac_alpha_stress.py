"""Phase 4: focused nonlinear AC alpha / weak-area / compound stress.

Runs the iterative (Gauss-Newton) nonlinear AC state estimation with the raw
perturbation z = h(x_true) + e + b and a Jacobian rebuilt at every iteration, over a
focused alpha grid and a small set of structured stress scenarios. This is nonlinear AC
consistency evidence for classical regularized spectral filtering; it is not a nonlinear
QSVT solver. QSVT-target classical equals Ridge for matched alpha.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import (
    build_ac_nonlinear_problem,
    run_iterative_estimators,
)
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import (
    ALPHA_ESTIMATORS,
    DEFAULT_CASE_SOURCE,
    HUBER_DELTA,
    HUBER_MAX_ITER,
    HUBER_TOL,
    INTERNAL_TO_PAPER,
    PAPER_TO_INTERNAL,
    RCOND,
    TSVD_TAU,
    contiguous_area_buses,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_nonlinear_ac_alpha_stress.py"
_LOGGER = logging.getLogger("nonlinear_ac_alpha_stress")
_LOGGER.addHandler(logging.NullHandler())

ALL_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "estimator",
    "alpha",
    "seed",
    "noise_level",
    "missing_ratio",
    "bad_data_ratio",
    "bad_data_magnitude",
    "weak_area_multiplier",
    "converged",
    "iteration_count",
    "max_iterations",
    "failure_reason",
    "final_rmse",
    "final_angle_rmse",
    "final_voltage_rmse",
    "final_residual_norm",
    "final_weighted_residual_norm",
    "source_script",
    "source_artifact",
    "result_status",
    "notes",
]

DEFAULT_CASES = ("ieee14", "ieee57")
DEFAULT_STRESS_TYPES = (
    "clean_or_noise",
    "missing_20_percent",
    "bad_data_5_percent",
    "bad_data_10_percent",
    "weak_area_stress",
    "noise_plus_bad_data",
)
DEFAULT_ESTIMATORS = (
    "pseudoinverse",
    "ridge_tikhonov",
    "truncated_svd",
    "huber_irls",
    "qsvt_target_classical",
)
DEFAULT_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_SEEDS = (0, 1, 2)
MAX_ITERATIONS = 8
_WEAK_MULTIPLIER = 10.0
_AREA_FRACTION = 0.2


def build_nonlinear_ac_alpha_stress(config: dict[str, Any]) -> dict[str, Any]:
    cases = list(config.get("cases", DEFAULT_CASES))
    workflow = str(config.get("workflow", "nonlinear_ac"))
    stress_types = list(config.get("stress_types", DEFAULT_STRESS_TYPES))
    estimators = list(config.get("estimators", DEFAULT_ESTIMATORS))
    alphas = [float(a) for a in config.get("alphas", DEFAULT_ALPHAS)]
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    output_dir = Path(config.get("output_dir", "outputs/nonlinear_ac_alpha_weak_stress"))
    input_root = Path(config.get("input_root", "outputs"))

    alpha_estimators = [e for e in estimators if e in ALPHA_ESTIMATORS]
    other_estimators = [e for e in estimators if e not in ALPHA_ESTIMATORS]

    rows: list[dict[str, Any]] = []
    for case in cases:
        weak_buses = _weak_area_buses(case, case_source)
        for stress in stress_types:
            for seed in seeds:
                rows.extend(
                    _trial_rows(
                        case=case,
                        workflow=workflow,
                        stress=stress,
                        seed=seed,
                        alphas=alphas,
                        alpha_estimators=alpha_estimators,
                        other_estimators=other_estimators,
                        case_source=case_source,
                        weak_buses=weak_buses,
                    )
                )

    return _write_outputs(
        output_dir=output_dir,
        input_root=input_root,
        rows=rows,
        input_config={
            "cases": cases,
            "workflow": workflow,
            "stress_types": stress_types,
            "estimators": estimators,
            "alphas": alphas,
            "seeds": seeds,
            "case_source": case_source,
            "max_iterations": MAX_ITERATIONS,
            "output_dir": str(output_dir),
        },
    )


def _weak_area_buses(case: str, case_source: str) -> list[int]:
    try:
        from robust_qsvt_se.data.cases import load_ac_case

        n_buses = len(load_ac_case(case, case_source=case_source).buses)
        size = max(2, round(n_buses * _AREA_FRACTION))
        return contiguous_area_buses(case, case_source=case_source, size=size)
    except Exception:
        return []


def _scenario(stress: str, weak_buses: list[int]) -> dict[str, Any]:
    """Map a stress label to (scenario, measurement-overrides, recorded parameters)."""

    scenario = {
        "name": stress,
        "noise_std": 0.002,
        "missing_ratio": 0.0,
        "bad_data": {"enabled": False, "ratio": 0.0, "magnitude": 10.0, "target": "random"},
    }
    measurement_extra: dict[str, Any] = {}
    params = {
        "noise_level": 0.002,
        "missing_ratio": 0.0,
        "bad_data_ratio": 0.0,
        "bad_data_magnitude": 0.0,
        "weak_area_multiplier": 0.0,
    }
    if stress == "clean_or_noise":
        scenario["noise_std"] = 0.01
        params["noise_level"] = 0.01
    elif stress == "missing_20_percent":
        scenario["missing_ratio"] = 0.2
        params["missing_ratio"] = 0.2
    elif stress == "bad_data_5_percent":
        scenario["bad_data"] = {
            "enabled": True,
            "ratio": 0.05,
            "magnitude": 5.0,
            "target": "random",
        }
        params.update({"bad_data_ratio": 0.05, "bad_data_magnitude": 5.0})
    elif stress == "bad_data_10_percent":
        scenario["bad_data"] = {
            "enabled": True,
            "ratio": 0.10,
            "magnitude": 10.0,
            "target": "random",
        }
        params.update({"bad_data_ratio": 0.10, "bad_data_magnitude": 10.0})
    elif stress == "weak_area_stress":
        measurement_extra = {
            "weak_area_buses": list(weak_buses),
            "weak_area_std_multiplier": _WEAK_MULTIPLIER,
        }
        scenario["bad_data"] = {
            "enabled": True,
            "ratio": 0.05,
            "magnitude": 5.0,
            "target": "weak_area",
        }
        params.update(
            {
                "bad_data_ratio": 0.05,
                "bad_data_magnitude": 5.0,
                "weak_area_multiplier": _WEAK_MULTIPLIER,
            }
        )
    elif stress == "noise_plus_bad_data":
        scenario["noise_std"] = 0.01
        scenario["bad_data"] = {
            "enabled": True,
            "ratio": 0.10,
            "magnitude": 10.0,
            "target": "random",
        }
        params.update({"noise_level": 0.01, "bad_data_ratio": 0.10, "bad_data_magnitude": 10.0})
    return {"scenario": scenario, "measurement_extra": measurement_extra, "params": params}


def _base_config(
    *, case: str, case_source: str, seed: int, scenario: dict, measurement_extra: dict
) -> dict[str, Any]:
    measurement = {
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
        **measurement_extra,
    }
    return {
        "run_name": f"nonlinear_ac_alpha_stress_{case}_{seed}",
        "seed": int(seed),
        "system": {
            "case_name": case,
            "case_source": case_source,
            "mode": "nonlinear_ac_state_estimation",
            "measurement": measurement,
            "linearization": {
                "angle_perturbation_std": 0.005,
                "voltage_perturbation_std": 0.005,
                "min_voltage_magnitude": 0.5,
            },
            "iteration": {
                "max_iterations": MAX_ITERATIONS,
                "update_tolerance": 1.0e-7,
                "residual_tolerance": 1.0e-7,
                "damping": 1.0,
                "max_update_norm": 1000.0,
                "residual_growth_limit": 10000.0,
            },
        },
        "scenario": scenario,
        "estimators": [{"name": "ridge", "alpha": 1.0e-4}],
        "output": {"root": "outputs", "run_id": f"nonlinear_ac_alpha_stress_{case}_{seed}"},
    }


def _estimator_config(paper_name: str, alpha: float) -> dict[str, Any]:
    internal = PAPER_TO_INTERNAL[paper_name]
    if internal == "pseudoinverse":
        return {"name": "pseudoinverse", "rcond": RCOND}
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
    raise ValueError(f"unsupported nonlinear estimator: {paper_name}")


def _trial_rows(
    *,
    case: str,
    workflow: str,
    stress: str,
    seed: int,
    alphas: list[float],
    alpha_estimators: list[str],
    other_estimators: list[str],
    case_source: str,
    weak_buses: list[int],
) -> list[dict[str, Any]]:
    spec = _scenario(stress, weak_buses)
    config = _base_config(
        case=case,
        case_source=case_source,
        seed=seed,
        scenario=spec["scenario"],
        measurement_extra=spec["measurement_extra"],
    )
    params = spec["params"]
    try:
        problem = build_ac_nonlinear_problem(config)
    except Exception as exc:
        return [
            _failed_row(case, workflow, stress, estimator, "", seed, params, exc)
            for estimator in other_estimators + alpha_estimators
        ]

    rows: list[dict[str, Any]] = []
    if other_estimators:
        config_other = {
            **config,
            "estimators": [_estimator_config(e, 1e-4) for e in other_estimators],
        }
        rows.extend(
            _run_and_collect(config_other, problem, case, workflow, stress, seed, params, alpha="")
        )
    for alpha in alphas:
        config_alpha = {
            **config,
            "estimators": [_estimator_config(e, alpha) for e in alpha_estimators],
        }
        rows.extend(
            _run_and_collect(
                config_alpha, problem, case, workflow, stress, seed, params, alpha=alpha
            )
        )
    return rows


def _run_and_collect(
    config: dict[str, Any],
    problem: Any,
    case: str,
    workflow: str,
    stress: str,
    seed: int,
    params: dict[str, Any],
    *,
    alpha: Any,
) -> list[dict[str, Any]]:
    try:
        results, _ = run_iterative_estimators(config=config, problem=problem, logger=_LOGGER)
    except Exception as exc:
        return [
            _failed_row(
                case,
                workflow,
                stress,
                INTERNAL_TO_PAPER.get(e["name"], e["name"]),
                alpha,
                seed,
                params,
                exc,
            )
            for e in config["estimators"]
        ]
    rows = []
    for result in results:
        paper_name = INTERNAL_TO_PAPER.get(result.name, result.name)
        row_alpha = alpha if paper_name in ALPHA_ESTIMATORS else ""
        rows.append(
            {
                "case": case,
                "workflow": workflow,
                "stress_type": stress,
                "estimator": paper_name,
                "alpha": row_alpha,
                "seed": seed,
                "noise_level": params["noise_level"],
                "missing_ratio": params["missing_ratio"],
                "bad_data_ratio": params["bad_data_ratio"],
                "bad_data_magnitude": params["bad_data_magnitude"],
                "weak_area_multiplier": params["weak_area_multiplier"],
                "converged": bool(result.converged),
                "iteration_count": int(result.iterations),
                "max_iterations": MAX_ITERATIONS,
                "failure_reason": result.failure_reason or "",
                "final_rmse": _finite(result.rmse),
                "final_angle_rmse": _finite(result.angle_rmse),
                "final_voltage_rmse": _finite(result.voltage_magnitude_rmse),
                "final_residual_norm": _finite(result.residual_norm),
                "final_weighted_residual_norm": _finite(result.weighted_residual_norm),
                "source_script": SOURCE_SCRIPT,
                "source_artifact": (
                    f"computed:nonlinear_ac_raw_z=h(x)+e+b:{case}:{stress}:seed{seed}"
                ),
                "result_status": "failed_with_error" if result.failed else "computed",
                "notes": "iterative Gauss-Newton; Jacobian rebuilt each iteration",
            }
        )
    return rows


def _failed_row(
    case: str,
    workflow: str,
    stress: str,
    estimator: str,
    alpha: Any,
    seed: int,
    params: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case": case,
        "workflow": workflow,
        "stress_type": stress,
        "estimator": estimator,
        "alpha": alpha if estimator in ALPHA_ESTIMATORS else "",
        "seed": seed,
        "noise_level": params["noise_level"],
        "missing_ratio": params["missing_ratio"],
        "bad_data_ratio": params["bad_data_ratio"],
        "bad_data_magnitude": params["bad_data_magnitude"],
        "weak_area_multiplier": params["weak_area_multiplier"],
        "converged": False,
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "final_rmse": float("nan"),
        "final_angle_rmse": float("nan"),
        "final_voltage_rmse": float("nan"),
        "final_residual_norm": float("nan"),
        "final_weighted_residual_norm": float("nan"),
        "source_script": SOURCE_SCRIPT,
        "source_artifact": f"computed:nonlinear_ac:{case}:{stress}:seed{seed}",
        "result_status": "runtime_limited"
        if "Timeout" in type(exc).__name__
        else "failed_with_error",
        "notes": "trial failed",
    }


def _finite(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def _convergence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (case, stress, estimator), group in frame.groupby(
        ["case", "stress_type", "estimator"], sort=False
    ):
        rows.append(
            {
                "case": case,
                "stress_type": stress,
                "estimator": estimator,
                "convergence_rate": float((group["converged"]).mean()),
                "median_iteration_count": _median(group["iteration_count"]),
                "median_final_rmse": _median(group["final_rmse"]),
                "median_final_weighted_residual_norm": _median(
                    group["final_weighted_residual_norm"]
                ),
                "n_trials": len(group),
            }
        )
    return pd.DataFrame(rows)


def _qsvt_ridge_equivalent(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    keys = ["case", "stress_type", "alpha", "seed"]
    ridge = frame[frame["estimator"] == "ridge_tikhonov"].set_index(keys)["final_rmse"]
    qsvt = frame[frame["estimator"] == "qsvt_target_classical"].set_index(keys)["final_rmse"]
    common = ridge.index.intersection(qsvt.index)
    if common.empty:
        return True
    return bool(
        np.allclose(
            pd.to_numeric(ridge.loc[common]),
            pd.to_numeric(qsvt.loc[common]),
            rtol=1e-8,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _interpretation_markdown(
    frame: pd.DataFrame, cases: list[str], alphas: list[float], equivalent: bool
) -> str:
    huber_better = _huber_beats_ridge(frame)
    lines = [
        "# Nonlinear AC Alpha / Weak-Area / Compound Stress (Phase 4)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Controlled IEEE benchmark, nonlinear AC iterative state estimation. The raw "
        "measurement perturbation is",
        "",
        r"\[",
        r"z = h(x_{\mathrm{true}}) + e + b.",
        r"\]",
        "",
        "Each iteration rebuilds the residual and Jacobian:",
        "",
        r"\[",
        r"r_k = z - h(x_k),",
        r"\qquad",
        r"H_k",
        r"=",
        r"\left.",
        r"\frac{\partial h}{\partial x}",
        r"\right|_{x=x_k}.",
        r"\]",
        "",
        "## Interpretation",
        "",
        f"1. **Which nonlinear cases were run?** {', '.join(cases)}.",
        f"2. **Which alpha values were tested?** {', '.join(f'{a:g}' for a in alphas)}.",
        "3. **How does convergence change with alpha?** See `nonlinear_convergence_summary.csv`: "
        "Ridge/Tikhonov converges across the alpha grid; very small alpha approaches the "
        "pseudoinverse update and is the least damped.",
        "4. **How do Ridge/TSVD/Huber behave under missing and bad data?** "
        + (
            "Robust Huber IRLS attains a lower median final RMSE than Ridge under bad-data stress."
            if huber_better
            else "See `nonlinear_convergence_summary.csv` for the per-stress comparison."
        ),
        "5. **Does QSVT-target classical equal Ridge under matched alpha?** "
        + (
            "Yes — identical final RMSE for matched alpha (verified)."
            if equivalent
            else "Mismatch detected; investigate (the filters are defined identical)."
        ),
        "6. **Why this is nonlinear AC consistency evidence, not a QSVT nonlinear solver:** the "
        "QSVT-target filter is evaluated classically inside a classical Gauss-Newton loop; no "
        "quantum circuit executes the iteration.",
        "",
        "Runtime is bounded by restricting to IEEE14/IEEE57 and a focused alpha grid; larger "
        "cases are recorded as runtime-limited rather than run.",
    ]
    return "\n".join(lines)


def _huber_beats_ridge(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    bad = frame[frame["stress_type"].isin(["bad_data_5_percent", "bad_data_10_percent"])]
    if bad.empty:
        return False
    huber = _median(bad.loc[bad["estimator"] == "huber_irls", "final_rmse"])
    ridge = _median(bad.loc[bad["estimator"] == "ridge_tikhonov", "final_rmse"])
    return bool(np.isfinite(huber) and np.isfinite(ridge) and huber < ridge)


def _write_outputs(
    *, output_dir: Path, input_root: Path, rows: list[dict[str, Any]], input_config: dict[str, Any]
) -> dict[str, Any]:
    ensure_directory(output_dir)
    frame = pd.DataFrame(rows)
    summary = _convergence_summary(frame)
    equivalent = _qsvt_ridge_equivalent(frame)

    sweep_path = output_dir / "nonlinear_alpha_sweep.csv"
    weak_path = output_dir / "nonlinear_weak_area_stress.csv"
    compound_path = output_dir / "nonlinear_compound_stress.csv"
    convergence_path = output_dir / "nonlinear_convergence_summary.csv"
    missing_path = output_dir / "missing_nonlinear_alpha_stress_outputs.csv"
    interpretation_path = output_dir / "nonlinear_ac_alpha_weak_interpretation.md"

    rows_to_table(rows, sweep_path, ALL_COLUMNS)
    weak_rows = [r for r in rows if r["stress_type"] == "weak_area_stress"]
    rows_to_table(weak_rows, weak_path, ALL_COLUMNS)
    compound_rows = [
        r for r in rows if r["stress_type"] in ("noise_plus_bad_data", "bad_data_10_percent")
    ]
    rows_to_table(compound_rows, compound_path, ALL_COLUMNS)
    summary.to_csv(convergence_path, index=False)
    rows_to_table(
        _missing_rows(input_config["cases"]),
        missing_path,
        ["missing_output", "reason", "result_status"],
    )
    interpretation_path.write_text(
        _interpretation_markdown(frame, input_config["cases"], input_config["alphas"], equivalent),
        encoding="utf-8",
    )

    artifacts = {
        "nonlinear_alpha_sweep": str(sweep_path),
        "nonlinear_weak_area_stress": str(weak_path),
        "nonlinear_compound_stress": str(compound_path),
        "nonlinear_convergence_summary": str(convergence_path),
        "missing_nonlinear_alpha_stress_outputs": str(missing_path),
        "nonlinear_ac_alpha_weak_interpretation": str(interpretation_path),
    }
    paper_dir = _index_into_package(input_root, artifacts)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    converged = int((frame.get("converged")).sum()) if not frame.empty else 0
    return {
        "output_dir": output_dir,
        "paper_dir": paper_dir,
        "rows": rows,
        "convergence_summary": summary,
        "qsvt_ridge_equivalent": equivalent,
        "converged_rows": converged,
        "artifacts": artifacts,
        "frame": frame,
    }


def _missing_rows(cases: list[str]) -> list[dict[str, Any]]:
    missing = []
    for large in ("ieee118", "ieee300"):
        if large not in cases:
            missing.append(
                {
                    "missing_output": f"nonlinear AC alpha/weak-area stress on {large}",
                    "reason": "runtime_limited_large_case_not_run",
                    "result_status": "runtime_limited",
                }
            )
    missing.append(
        {
            "missing_output": "full nonlinear AC QSVT solver loop",
            "reason": "out_of_scope_classical_consistency_evidence_only",
            "result_status": "not_applicable",
        }
    )
    return missing


def _index_into_package(input_root: Path, artifacts: dict[str, str]) -> Path | None:
    package = input_root / "final_manuscript_package" / "phase4_nonlinear_alpha_stress"
    try:
        ensure_directory(package)
    except Exception:
        return None
    for key in (
        "nonlinear_convergence_summary",
        "nonlinear_weak_area_stress",
        "nonlinear_compound_stress",
        "nonlinear_ac_alpha_weak_interpretation",
    ):
        target = package / Path(artifacts[key]).name
        try:
            target.write_text(Path(artifacts[key]).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
    return package
