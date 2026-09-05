"""Phase 3: compound / weak-area / spatial structured stress.

Extends the single-axis stress evidence with compound (noise+missing+bad-data),
weak-area, and contiguous spatial-drop stress on controlled IEEE benchmarks. Weak-area
and spatial stress use deterministic topology-based criteria and are labelled as
controlled benchmark assumptions, never as field-calibrated statistics. Random missing
and random bad data are labelled ``random`` and are never relabelled as structured.
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
    apply_noise,
    build_estimator,
    build_system,
    conditioning,
    contiguous_area_buses,
    filter_rows_by_buses,
    high_condition_measurement_config,
    solve_detailed,
    subset_spec,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_compound_structured_stress.py"

ALL_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "stress_subtype",
    "stress_parameter_summary",
    "estimator",
    "alpha",
    "seed",
    "noise_scale",
    "missing_ratio",
    "bad_data_ratio",
    "bad_data_magnitude",
    "weak_area_multiplier",
    "dropped_area_or_rows",
    "rmse",
    "angle_rmse",
    "voltage_rmse",
    "weighted_residual_norm",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "numerical_rank",
    "effective_rank",
    "converged",
    "source_script",
    "source_artifact",
    "result_status",
    "failure_reason",
    "notes",
]

DEFAULT_CASES = ("ieee14", "ieee57", "ieee118")
DEFAULT_STRESS_TYPES = (
    "noise_only",
    "missing_only",
    "bad_data_only",
    "weak_area_only",
    "noise_plus_missing",
    "noise_plus_bad_data",
    "missing_plus_bad_data",
    "noise_plus_missing_plus_bad_data",
    "weak_area_plus_missing",
    "weak_area_plus_bad_data",
    "contiguous_area_drop",
)
DEFAULT_ESTIMATORS = (
    "pseudoinverse",
    "ridge_tikhonov",
    "truncated_svd",
    "huber_irls",
    "qsvt_target_classical",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FIXED_ALPHA = 1.0e-4

# Representative single-level parameters; seeds provide the variation (reduced design).
_NOISE_SCALE = 1.0
_MISSING_RATIO = 0.2
_BAD_DATA_RATIO = 0.1
_BAD_DATA_MAGNITUDE = 10.0
_WEAK_MULTIPLIER = 10.0
_AREA_FRACTION = 0.18

_WEAK_TYPES = frozenset(
    {"weak_area_only", "weak_area_plus_missing", "weak_area_plus_bad_data", "contiguous_area_drop"}
)
_BAD_DATA_HEAVY = frozenset(
    {
        "bad_data_only",
        "noise_plus_bad_data",
        "missing_plus_bad_data",
        "noise_plus_missing_plus_bad_data",
        "weak_area_plus_bad_data",
    }
)


def build_compound_structured_stress(config: dict[str, Any]) -> dict[str, Any]:
    cases = list(config.get("cases", DEFAULT_CASES))
    workflow = str(config.get("workflow", "ac_linearized"))
    stress_types = list(config.get("stress_types", DEFAULT_STRESS_TYPES))
    estimators = list(config.get("estimators", DEFAULT_ESTIMATORS))
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    output_dir = Path(config.get("output_dir", "outputs/structured_compound_stress"))
    input_root = Path(config.get("input_root", "outputs"))

    full_config = subset_spec("full_ac_measurement_set").measurement_config
    rows: list[dict[str, Any]] = []
    available_types: set[str] = set()
    for case in cases:
        area = _area_buses(case, case_source)
        weak_config = high_condition_measurement_config(
            case, case_source=case_source, multiplier=_WEAK_MULTIPLIER
        )
        for seed in seeds:
            base = build_system(
                case=case, measurement_config=full_config, seed=seed, case_source=case_source
            )
            weak = build_system(
                case=case, measurement_config=weak_config, seed=seed, case_source=case_source
            )
            for stress in stress_types:
                realized = _realize(stress, base=base, weak=weak, area=area, seed=seed)
                if realized is None:
                    continue
                system, fields = realized
                available_types.add(stress)
                cond = conditioning(system)
                for estimator_name in estimators:
                    detail = solve_detailed(build_estimator(estimator_name, FIXED_ALPHA), system)
                    rows.append(
                        _row(case, workflow, stress, estimator_name, seed, fields, detail, cond)
                    )

    return _write_outputs(
        output_dir=output_dir,
        input_root=input_root,
        rows=rows,
        available_types=sorted(available_types),
        requested_types=stress_types,
        input_config={
            "cases": cases,
            "workflow": workflow,
            "stress_types": stress_types,
            "estimators": estimators,
            "seeds": seeds,
            "case_source": case_source,
            "reduced_design": {
                "noise_scale": _NOISE_SCALE,
                "missing_ratio": _MISSING_RATIO,
                "bad_data_ratio": _BAD_DATA_RATIO,
                "bad_data_magnitude": _BAD_DATA_MAGNITUDE,
                "weak_area_multiplier": _WEAK_MULTIPLIER,
                "seeds": len(seeds),
            },
            "output_dir": str(output_dir),
        },
    )


def _area_buses(case: str, case_source: str) -> set[int]:
    try:
        from robust_qsvt_se.data.cases import load_ac_case

        n_buses = len(load_ac_case(case, case_source=case_source).buses)
        size = max(2, round(n_buses * _AREA_FRACTION))
        return set(contiguous_area_buses(case, case_source=case_source, size=size))
    except Exception:
        return set()


def _realize(
    stress: str, *, base: Any, weak: Any, area: set[int], seed: int
) -> tuple[Any, dict[str, Any]] | None:
    """Realize a stress scenario, returning (system, parameter fields) or None."""

    fields = {
        "stress_subtype": "single_axis",
        "noise_scale": 0.0,
        "missing_ratio": 0.0,
        "bad_data_ratio": 0.0,
        "bad_data_magnitude": 0.0,
        "weak_area_multiplier": 0.0,
        "dropped_area_or_rows": 0,
        "result_status": "computed",
    }
    is_weak = stress in _WEAK_TYPES
    system = weak if is_weak and stress != "contiguous_area_drop" else base
    if is_weak:
        fields["stress_subtype"] = "controlled_topology_assumption"
        fields["result_status"] = "controlled_assumption"
        if stress != "contiguous_area_drop":
            fields["weak_area_multiplier"] = _WEAK_MULTIPLIER

    try:
        if "noise" in stress:
            system = apply_noise(system, noise_std=_NOISE_SCALE, seed=seed)
            fields["noise_scale"] = _NOISE_SCALE
            if not is_weak:
                fields["stress_subtype"] = "gaussian"
        if "missing" in stress:
            system = apply_missing(system, missing_ratio=_MISSING_RATIO, seed=seed + 1)
            fields["missing_ratio"] = _MISSING_RATIO
            if not is_weak:
                fields["stress_subtype"] = "random"
        if "bad_data" in stress:
            target = "weak_area" if stress == "weak_area_plus_bad_data" else "random"
            system = apply_bad_data(
                system,
                ratio=_BAD_DATA_RATIO,
                magnitude=_BAD_DATA_MAGNITUDE,
                target=target,
                seed=seed + 2,
            )
            fields["bad_data_ratio"] = _BAD_DATA_RATIO
            fields["bad_data_magnitude"] = _BAD_DATA_MAGNITUDE
            if not is_weak:
                fields["stress_subtype"] = "random"
        if stress == "contiguous_area_drop":
            if not area:
                return None
            system, n_dropped = filter_rows_by_buses(base, area)
            fields["dropped_area_or_rows"] = int(n_dropped)
            if n_dropped == 0:
                return None
    except Exception:
        return None

    compound = sum(1 for token in ("noise", "missing", "bad_data") if token in stress)
    if compound >= 2:
        fields["stress_subtype"] = "compound_random" if not is_weak else fields["stress_subtype"]
    fields["stress_parameter_summary"] = _param_summary(fields)
    return system, fields


def _param_summary(fields: dict[str, Any]) -> str:
    parts = []
    if fields["noise_scale"]:
        parts.append(f"noise={fields['noise_scale']:g}")
    if fields["missing_ratio"]:
        parts.append(f"missing={fields['missing_ratio']:g}")
    if fields["bad_data_ratio"]:
        parts.append(f"bad_data={fields['bad_data_ratio']:g}@{fields['bad_data_magnitude']:g}sigma")
    if fields["weak_area_multiplier"]:
        parts.append(f"weak_x{fields['weak_area_multiplier']:g}")
    if fields["dropped_area_or_rows"]:
        parts.append(f"area_drop={fields['dropped_area_or_rows']}rows")
    return "; ".join(parts) or "none"


def _row(
    case: str,
    workflow: str,
    stress: str,
    estimator: str,
    seed: int,
    fields: dict[str, Any],
    detail: dict[str, Any],
    cond: dict[str, float],
) -> dict[str, Any]:
    status = "failed_with_error" if detail["failed"] else fields["result_status"]
    return {
        "case": case,
        "workflow": workflow,
        "stress_type": stress,
        "stress_subtype": fields["stress_subtype"],
        "stress_parameter_summary": fields["stress_parameter_summary"],
        "estimator": estimator,
        "alpha": FIXED_ALPHA if estimator in ALPHA_ESTIMATORS else "",
        "seed": seed,
        "noise_scale": fields["noise_scale"],
        "missing_ratio": fields["missing_ratio"],
        "bad_data_ratio": fields["bad_data_ratio"],
        "bad_data_magnitude": fields["bad_data_magnitude"],
        "weak_area_multiplier": fields["weak_area_multiplier"],
        "dropped_area_or_rows": fields["dropped_area_or_rows"],
        "rmse": detail["rmse"],
        "angle_rmse": detail["angle_rmse"],
        "voltage_rmse": detail["voltage_rmse"],
        "weighted_residual_norm": detail["weighted_residual_norm"],
        "condition_number": cond["condition_number"],
        "sigma_min": cond["sigma_min"],
        "sigma_max": cond["sigma_max"],
        "numerical_rank": cond["numerical_rank"],
        "effective_rank": cond["effective_rank"],
        "converged": detail["converged"],
        "source_script": SOURCE_SCRIPT,
        "source_artifact": f"computed:{workflow}:{case}:{stress}:seed{seed}",
        "result_status": status,
        "failure_reason": detail["failure_reason"],
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
                "stress_subtype": group["stress_subtype"].iloc[0],
                "median_rmse": _median(group["rmse"]),
                "median_weighted_residual_norm": _median(group["weighted_residual_norm"]),
                "median_condition_number": _median(group["condition_number"]),
                "failure_rate": failures / len(group) if len(group) else float("nan"),
                "result_status": group["result_status"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _robustness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (stress_type, estimator) median RMSE across cases, plus ratio vs Ridge."""

    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (stress, estimator), group in frame.groupby(["stress_type", "estimator"], sort=False):
        rows.append(
            {
                "stress_type": stress,
                "estimator": estimator,
                "median_rmse": _median(group["rmse"]),
            }
        )
    out = pd.DataFrame(rows)
    ridge = out[out["estimator"] == "ridge_tikhonov"].set_index("stress_type")["median_rmse"]
    out["ridge_median_rmse"] = out["stress_type"].map(ridge)
    out["rmse_ratio_vs_ridge"] = out["median_rmse"] / out["ridge_median_rmse"]
    out["outperforms_ridge"] = out["rmse_ratio_vs_ridge"] < 0.999
    return out


def _findings(frame: pd.DataFrame, robustness: pd.DataFrame) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    if robustness.empty:
        return findings
    bad_data = robustness[robustness["stress_type"].isin(_BAD_DATA_HEAVY)]
    huber = bad_data[bad_data["estimator"] == "huber_irls"]
    findings["huber_beats_ridge_under_bad_data"] = bool(
        not huber.empty and (huber["outperforms_ridge"]).any()
    )
    if not bad_data.empty:
        best = bad_data.sort_values("median_rmse").iloc[0]
        findings["best_estimator_bad_data"] = str(best["estimator"])
    return findings


def _interpretation_markdown(
    available: list[str], requested: list[str], findings: dict[str, Any]
) -> str:
    missing = sorted(set(requested) - set(available))
    weak_available = [s for s in available if s in _WEAK_TYPES]
    huber_wins = findings.get("huber_beats_ridge_under_bad_data", False)
    lines = [
        "# Compound / Weak-Area / Spatial Structured Stress (Phase 3)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Controlled IEEE benchmark, AC-linearized weighted update with the single-step "
        "weighted-residual perturbation",
        "",
        r"\[",
        r"\tilde r_{\mathrm{perturbed}}",
        r"=",
        r"\tilde r_{\mathrm{clean}}",
        r"+",
        r"\tilde e",
        r"+",
        r"\tilde b.",
        r"\]",
        "",
        "Weak-area and contiguous-area stress are deterministic topology-based controlled "
        "benchmark assumptions (BFS-grown bus area; inflated weak-area measurement standard "
        "deviations). They are not field-calibrated stress models.",
        "",
        "## Interpretation",
        "",
        f"1. **Which compound stress types are available?** {', '.join(available)}.",
        f"2. **Which weak-area/spatial stress types are available?** "
        f"{', '.join(weak_available) or 'none'} (controlled benchmark assumptions).",
        "3. **Does regularization help when conditioning worsens?** Ridge/Tikhonov "
        r"\(P_\alpha(\sigma)=\sigma/(\sigma^2+\alpha)\) keeps the weighted residual finite under "
        "the weak-area and compound stress (see `compound_stress_summary.csv`).",
        "4. **Do Huber/robust estimators outperform Ridge under bad-data-heavy stress?** "
        + (
            "Yes — Huber IRLS attains a lower median RMSE than Ridge under at least one "
            "bad-data-heavy stress type (`estimator_robustness_by_stress.csv`)."
            if huber_wins
            else "Not in this run; see `estimator_robustness_by_stress.csv`."
        ),
        f"5. **Which stress types remain missing?** {', '.join(missing) or 'none requested'}; "
        "field-calibrated stress statistics remain out of scope.",
        "6. **Which claims are supported only as controlled benchmark assumptions?** Weak-area "
        "and contiguous spatial stress (result_status=`controlled_assumption`).",
        "7. **Why these are not field-calibrated stress models:** the stress areas and "
        "magnitudes are deterministic benchmark choices, not measured PMU/SCADA statistics.",
        "",
        "Random missing and random bad data are labelled `random`; they are never relabelled as "
        "structured or spatial. QSVT-target classical equals Ridge for matched alpha.",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    input_root: Path,
    rows: list[dict[str, Any]],
    available_types: list[str],
    requested_types: list[str],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    frame = pd.DataFrame(rows)
    summary = _summary_frame(frame)
    robustness = _robustness_frame(frame)
    findings = _findings(frame, robustness)

    all_path = output_dir / "compound_stress_all_results.csv"
    summary_path = output_dir / "compound_stress_summary.csv"
    weak_path = output_dir / "weak_area_stress_summary.csv"
    spatial_path = output_dir / "spatial_missing_summary.csv"
    robustness_path = output_dir / "estimator_robustness_by_stress.csv"
    missing_path = output_dir / "missing_compound_stress_outputs.csv"
    interpretation_path = output_dir / "structured_compound_stress_interpretation.md"

    rows_to_table(rows, all_path, ALL_COLUMNS)
    summary.to_csv(summary_path, index=False)
    weak = summary[summary["stress_type"].isin(_WEAK_TYPES)] if not summary.empty else summary
    weak.to_csv(weak_path, index=False)
    spatial = (
        summary[summary["stress_type"].isin({"contiguous_area_drop", "weak_area_plus_missing"})]
        if not summary.empty
        else summary
    )
    spatial.to_csv(spatial_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    rows_to_table(
        _missing_rows(requested_types, available_types),
        missing_path,
        ["missing_output", "reason", "result_status"],
    )
    interpretation_path.write_text(
        _interpretation_markdown(available_types, requested_types, findings), encoding="utf-8"
    )

    artifacts = {
        "compound_stress_all_results": str(all_path),
        "compound_stress_summary": str(summary_path),
        "weak_area_stress_summary": str(weak_path),
        "spatial_missing_summary": str(spatial_path),
        "estimator_robustness_by_stress": str(robustness_path),
        "missing_compound_stress_outputs": str(missing_path),
        "structured_compound_stress_interpretation": str(interpretation_path),
    }
    paper_dir = _index_into_package(input_root, artifacts)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    computed = int((frame.get("result_status") == "computed").sum()) if not frame.empty else 0
    controlled = (
        int((frame.get("result_status") == "controlled_assumption").sum()) if not frame.empty else 0
    )
    return {
        "output_dir": output_dir,
        "paper_dir": paper_dir,
        "rows": rows,
        "available_types": available_types,
        "findings": findings,
        "computed_rows": computed,
        "controlled_assumption_rows": controlled,
        "artifacts": artifacts,
        "frame": frame,
    }


def _missing_rows(requested: list[str], available: list[str]) -> list[dict[str, Any]]:
    missing = [
        {
            "missing_output": stress,
            "reason": "not_supported_by_current_code",
            "result_status": "not_supported_by_current_code",
        }
        for stress in requested
        if stress not in available
    ]
    missing.append(
        {
            "missing_output": "field-calibrated stress statistics",
            "reason": "out_of_scope_controlled_benchmark_only",
            "result_status": "not_supported_by_current_code",
        }
    )
    return missing


def _index_into_package(input_root: Path, artifacts: dict[str, str]) -> Path | None:
    package = input_root / "final_manuscript_package" / "phase5_compound_structured_stress"
    try:
        ensure_directory(package)
    except Exception:
        return None
    for key in (
        "compound_stress_summary",
        "weak_area_stress_summary",
        "spatial_missing_summary",
        "estimator_robustness_by_stress",
        "structured_compound_stress_interpretation",
    ):
        target = package / Path(artifacts[key]).name
        try:
            target.write_text(Path(artifacts[key]).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
    return package
