"""Structure-level inference for physical selected-output accuracy.

Residual seeds, functionals, budgets, and numerical realizations are never
bootstrap units.  The only resampled identifier is ``structural_group_id``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.physical_alignment.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)
from robust_qsvt_se.physical_alignment.config import load_campaign_config

STUDY_ID = "tqe_physical_alignment_structure_statistics_v1"
METRICS = ("E_physical", "A_physical", "E_support")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    observed_effect: float
    confidence_interval_low: float
    confidence_interval_high: float
    probability_effect_positive: float
    replicates: int
    seed: int
    resampling_unit: str


def _primary_cell(
    raw: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    classification: str | None = None,
) -> pd.DataFrame:
    cell = settings["primary_cell"]
    sparse = raw["k_budget"].eq(float(cell["support_budget"])) & raw["slot_budget"].eq(
        float(cell["slot_budget"])
    )
    full = raw["selector"].eq("full_support")
    selected = raw.loc[
        raw["regularization_regime"].eq(cell["regularization_regime"]) & (sparse | full)
    ].copy()
    requested_classification = classification or str(cell["functional_classification"])
    return selected.loc[selected["functional_classification"].eq(requested_classification)].copy()


def aggregate_structure_units(
    raw: pd.DataFrame,
    *,
    metrics: Sequence[str] = METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen two-stage aggregation without treating realizations as units."""

    required = {
        "structural_group_id",
        "instance_id",
        "realization_order",
        "ieee_case",
        "selector",
        "functional_classification",
        "functional_family",
        "residual_seed",
        "functional_id",
        *metrics,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw physical rows lack columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("cannot aggregate an empty physical audit")
    if not raw["status"].eq("completed").all():
        raise ValueError("non-completed rows must not enter numerical aggregation")

    realization_rows: list[dict[str, Any]] = []
    group_columns = [
        "structural_group_id",
        "instance_id",
        "realization_order",
        "ieee_case",
        "selector",
        "functional_classification",
    ]
    scopes: list[tuple[str, pd.DataFrame]] = [("all_families", raw)]
    scopes.extend(
        (f"family:{family}", frame) for family, frame in raw.groupby("functional_family", sort=True)
    )
    for scope, frame in scopes:
        for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
            base = dict(zip(group_columns, keys, strict=True))
            for metric in metrics:
                values = group[metric].to_numpy(dtype=np.float64)
                realization_rows.append(
                    {
                        **base,
                        "summary_scope": scope,
                        "metric": metric,
                        "realization_statistic": float(np.median(values)),
                        "held_out_row_count": len(group),
                        "held_out_seed_count": int(group["residual_seed"].nunique()),
                        "functional_count": int(group["functional_id"].nunique()),
                    }
                )
    realization = pd.DataFrame(realization_rows)
    structure_columns = [
        "structural_group_id",
        "ieee_case",
        "selector",
        "functional_classification",
        "summary_scope",
        "metric",
    ]
    structure_rows: list[dict[str, Any]] = []
    for keys, group in realization.groupby(structure_columns, sort=True, dropna=False):
        structure_rows.append(
            {
                **dict(zip(structure_columns, keys, strict=True)),
                "structure_statistic": float(group["realization_statistic"].mean()),
                "realization_count": int(group["instance_id"].nunique()),
                "held_out_row_count": int(group["held_out_row_count"].sum()),
                "held_out_seed_count_per_realization_min": int(group["held_out_seed_count"].min()),
                "functional_count_per_realization_min": int(group["functional_count"].min()),
                "aggregation_rule": (
                    "median held-out seed-functional rows within realization; "
                    "arithmetic mean of realization medians within structure"
                ),
                "independent_unit": "structural_group_id",
                "numerical_realizations_are_independent_units": False,
            }
        )
    return realization, pd.DataFrame(structure_rows)


def _paired_values(
    structure: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    metric: str = "E_physical",
) -> pd.DataFrame:
    filtered = structure.loc[
        structure["selector"].isin([candidate, baseline])
        & structure["metric"].eq(metric)
        & structure["functional_classification"].eq("physical")
        & structure["summary_scope"].eq("all_families")
    ]
    pivot = filtered.pivot(
        index=["structural_group_id", "ieee_case"],
        columns="selector",
        values="structure_statistic",
    ).reset_index()
    if candidate not in pivot or baseline not in pivot:
        return pd.DataFrame(
            columns=["structural_group_id", "ieee_case", "candidate", "baseline", "effect"]
        )
    result = pivot[["structural_group_id", "ieee_case", candidate, baseline]].dropna().copy()
    result = result.rename(columns={candidate: "candidate", baseline: "baseline"})
    result["effect"] = result["baseline"] - result["candidate"]
    return result


def _bootstrap_summary(
    effects: np.ndarray,
    sampled_effects: np.ndarray,
    *,
    confidence_level: float,
    seed: int,
    unit: str,
) -> BootstrapResult:
    tail = (1.0 - float(confidence_level)) / 2.0
    return BootstrapResult(
        observed_effect=float(np.mean(effects)),
        confidence_interval_low=float(np.quantile(sampled_effects, tail)),
        confidence_interval_high=float(np.quantile(sampled_effects, 1.0 - tail)),
        probability_effect_positive=float(np.mean(sampled_effects > 0.0)),
        replicates=len(sampled_effects),
        seed=int(seed),
        resampling_unit=unit,
    )


def bootstrap_paired_effect(
    paired: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap paired effects using structures, never raw task rows."""

    if paired["structural_group_id"].duplicated().any():
        raise ValueError("each structural group must appear exactly once in a paired analysis")
    effects = paired["effect"].to_numpy(dtype=np.float64)
    if len(effects) < 2:
        raise ValueError("at least two paired structures are required")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(effects), size=(int(replicates), len(effects)))
    sampled = effects[indices].mean(axis=1)
    return _bootstrap_summary(
        effects,
        sampled,
        confidence_level=confidence_level,
        seed=seed,
        unit="structural_group_id",
    )


def case_stratified_bootstrap_effect(
    paired: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Resample structures within cases while preserving exact case composition."""

    if paired["structural_group_id"].duplicated().any():
        raise ValueError("each structural group must appear exactly once in a paired analysis")
    cases = sorted(paired["ieee_case"].unique())
    if len(cases) < 2:
        raise ValueError("case-stratified bootstrap requires at least two cases")
    rng = np.random.default_rng(int(seed))
    sampled_parts = []
    observed_parts = []
    for case in cases:
        effects = paired.loc[paired["ieee_case"].eq(case), "effect"].to_numpy(dtype=np.float64)
        if not len(effects):
            raise ValueError(f"empty case stratum {case}")
        observed_parts.append(effects)
        indices = rng.integers(0, len(effects), size=(int(replicates), len(effects)))
        sampled_parts.append(effects[indices])
    observed = np.concatenate(observed_parts)
    sampled = np.concatenate(sampled_parts, axis=1).mean(axis=1)
    return _bootstrap_summary(
        observed,
        sampled,
        confidence_level=confidence_level,
        seed=seed,
        unit="structural_group_id within ieee_case",
    )


def leave_one_case_out(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for omitted in sorted(paired["ieee_case"].unique()):
        retained = paired.loc[~paired["ieee_case"].eq(omitted)]
        rows.append(
            {
                "omitted_case": omitted,
                "retained_structure_count": int(retained["structural_group_id"].nunique()),
                "effect_mean": float(retained["effect"].mean()),
                "effect_median": float(retained["effect"].median()),
                "fraction_effect_positive": float((retained["effect"] > 0.0).mean()),
                "complete_case_removed": bool(
                    not retained["structural_group_id"]
                    .isin(paired.loc[paired["ieee_case"].eq(omitted), "structural_group_id"])
                    .any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _case_summaries(structure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "ieee_case",
        "selector",
        "functional_classification",
        "summary_scope",
        "metric",
    ]
    for keys, group in structure.groupby(columns, sort=True, dropna=False):
        values = group["structure_statistic"]
        rows.append(
            {
                **dict(zip(columns, keys, strict=True)),
                "structure_count": int(group["structural_group_id"].nunique()),
                "mean_structure_statistic": float(values.mean()),
                "median_structure_statistic": float(values.median()),
                "std_structure_statistic": float(values.std(ddof=1)) if len(values) > 1 else None,
                "minimum_structure_statistic": float(values.min()),
                "maximum_structure_statistic": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _pair_inventory(structure: pd.DataFrame, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs = [tuple(pair) for pair in settings["required_pairwise"]]
    sparse_selectors = sorted(set(structure["selector"]) - {"full_support"})
    pairs.extend((selector, "full_support") for selector in sparse_selectors)
    return list(dict.fromkeys((str(candidate), str(baseline)) for candidate, baseline in pairs))


def _pairwise_summaries(
    structure: pd.DataFrame, settings: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    replicates = int(settings["bootstrap_replicates"])
    confidence = float(settings["confidence_level"])
    ordinary_seed = int(settings["bootstrap_seed"])
    stratified_seed = int(settings["case_stratified_bootstrap_seed"])
    summary_rows = []
    bootstrap_rows = []
    loco_rows = []
    for pair_index, (candidate, baseline) in enumerate(_pair_inventory(structure, settings)):
        paired = _paired_values(structure, candidate, baseline)
        if len(paired) < 2:
            summary_rows.append(
                {
                    "candidate_selector": candidate,
                    "baseline_selector": baseline,
                    "status": "unavailable",
                    "failure_reason": "fewer than two paired structures",
                    "paired_structure_count": len(paired),
                }
            )
            continue
        ordinary = bootstrap_paired_effect(
            paired,
            replicates=replicates,
            seed=ordinary_seed + pair_index,
            confidence_level=confidence,
        )
        stratified = case_stratified_bootstrap_effect(
            paired,
            replicates=replicates,
            seed=stratified_seed + pair_index,
            confidence_level=confidence,
        )
        summary_rows.append(
            {
                "candidate_selector": candidate,
                "baseline_selector": baseline,
                "status": "completed",
                "failure_reason": "",
                "effect_definition": "baseline E_physical minus candidate E_physical",
                "positive_effect_favors": "candidate_selector",
                "paired_structure_count": len(paired),
                "case_count": int(paired["ieee_case"].nunique()),
                "observed_mean_effect": ordinary.observed_effect,
                "observed_median_effect": float(paired["effect"].median()),
                "fraction_structures_favor_candidate": float((paired["effect"] > 0.0).mean()),
                "ordinary_ci_low": ordinary.confidence_interval_low,
                "ordinary_ci_high": ordinary.confidence_interval_high,
                "ordinary_probability_positive": ordinary.probability_effect_positive,
                "case_stratified_ci_low": stratified.confidence_interval_low,
                "case_stratified_ci_high": stratified.confidence_interval_high,
                "case_stratified_probability_positive": stratified.probability_effect_positive,
            }
        )
        bootstrap_rows.append(
            {
                "candidate_selector": candidate,
                "baseline_selector": baseline,
                "ordinary": asdict(ordinary),
                "case_stratified": asdict(stratified),
                "case_composition": paired.groupby("ieee_case").size().to_dict(),
            }
        )
        loco = leave_one_case_out(paired)
        loco.insert(0, "baseline_selector", baseline)
        loco.insert(0, "candidate_selector", candidate)
        loco_rows.extend(loco.to_dict(orient="records"))
    bootstrap = {
        "study_id": STUDY_ID,
        "primary_independent_unit": "structural_group_id",
        "raw_task_rows_are_resampling_units": False,
        "numerical_realizations_are_resampling_units": False,
        "ordinary_bootstrap_replicates": replicates,
        "ordinary_base_seed": ordinary_seed,
        "case_stratified_bootstrap_replicates": replicates,
        "case_stratified_base_seed": stratified_seed,
        "confidence_level": confidence,
        "comparisons": bootstrap_rows,
    }
    return pd.DataFrame(summary_rows), bootstrap, pd.DataFrame(loco_rows)


def _floor_sensitivity(
    raw: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for floor in settings.get("normalization_floor_sensitivity", ()):
        adjusted = raw.copy()
        adjusted["E_physical"] = adjusted["A_physical"] / np.maximum(
            adjusted["y_true"].abs(), float(floor)
        )
        _realization, structures = aggregate_structure_units(adjusted)
        primary = structures.loc[
            structures["metric"].eq("E_physical")
            & structures["functional_classification"].eq("physical")
            & structures["summary_scope"].eq("all_families")
        ]
        for selector, frame in primary.groupby("selector", sort=True):
            rows.append(
                {
                    "record_type": "selector_summary",
                    "normalization_floor": float(floor),
                    "candidate_selector": selector,
                    "baseline_selector": None,
                    "structure_count": int(frame["structural_group_id"].nunique()),
                    "mean_structure_E_physical": float(frame["structure_statistic"].mean()),
                    "mean_pairwise_effect": None,
                }
            )
        for candidate, baseline in _pair_inventory(structures, settings):
            paired = _paired_values(structures, candidate, baseline)
            rows.append(
                {
                    "record_type": "pairwise_effect",
                    "normalization_floor": float(floor),
                    "candidate_selector": candidate,
                    "baseline_selector": baseline,
                    "structure_count": int(paired["structural_group_id"].nunique()),
                    "mean_structure_E_physical": None,
                    "mean_pairwise_effect": (
                        float(paired["effect"].mean()) if not paired.empty else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _resource_pareto_summaries(
    raw: pd.DataFrame, settings: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every budget cell and mark per-structure physical-error Pareto points."""

    physical = raw.loc[
        raw["functional_classification"].eq("physical")
        & raw["status"].eq("completed")
        & raw["regularization_regime"].eq(settings["primary_cell"]["regularization_regime"])
    ].copy()
    sparse_cells = (
        physical.loc[physical["selector"].ne("full_support"), ["k_budget", "slot_budget"]]
        .drop_duplicates()
        .sort_values(["k_budget", "slot_budget"])
    )
    realization_rows = []
    for cell in sparse_cells.itertuples(index=False):
        selected = physical.loc[
            physical["selector"].eq("full_support")
            | (
                physical["k_budget"].eq(float(cell.k_budget))
                & physical["slot_budget"].eq(float(cell.slot_budget))
            )
        ]
        columns = [
            "structural_group_id",
            "instance_id",
            "realization_order",
            "ieee_case",
            "selector",
        ]
        for keys, group in selected.groupby(columns, sort=True):
            realization_rows.append(
                {
                    **dict(zip(columns, keys, strict=True)),
                    "analysis_k_budget": int(cell.k_budget),
                    "analysis_slot_budget": int(cell.slot_budget),
                    "E_physical": float(group["E_physical"].median()),
                    "A_physical": float(group["A_physical"].median()),
                    "E_support": float(group["E_support"].median()),
                    "support_nnz": float(group["support_nnz"].iloc[0]),
                    "held_out_row_count": len(group),
                }
            )
    realization = pd.DataFrame(realization_rows)
    columns = [
        "structural_group_id",
        "ieee_case",
        "selector",
        "analysis_k_budget",
        "analysis_slot_budget",
    ]
    structure = (
        realization.groupby(columns, sort=True)
        .agg(
            E_physical=("E_physical", "mean"),
            A_physical=("A_physical", "mean"),
            E_support=("E_support", "mean"),
            mean_support_nnz=("support_nnz", "mean"),
            realization_count=("instance_id", "nunique"),
            held_out_row_count=("held_out_row_count", "sum"),
        )
        .reset_index()
    )
    pareto_flags = []
    for _group_id, group in structure.groupby("structural_group_id", sort=True):
        errors = group["E_physical"].to_numpy(dtype=np.float64)
        resources = group["mean_support_nnz"].to_numpy(dtype=np.float64)
        for index in range(len(group)):
            no_worse = (resources <= resources[index]) & (errors <= errors[index])
            strict = (resources < resources[index]) | (errors < errors[index])
            pareto_flags.append((int(group.index[index]), not bool(np.any(no_worse & strict))))
    structure["physical_error_resource_pareto_nondominated"] = False
    for index, value in pareto_flags:
        structure.loc[index, "physical_error_resource_pareto_nondominated"] = value
    summary = (
        structure.groupby(["selector", "analysis_k_budget", "analysis_slot_budget"], sort=True)
        .agg(
            structure_count=("structural_group_id", "nunique"),
            pareto_structure_count=("physical_error_resource_pareto_nondominated", "sum"),
            mean_structure_E_physical=("E_physical", "mean"),
            median_structure_E_physical=("E_physical", "median"),
            mean_structure_A_physical=("A_physical", "mean"),
            mean_structure_E_support=("E_support", "mean"),
            mean_support_nnz=("mean_support_nnz", "mean"),
        )
        .reset_index()
    )
    summary["pareto_structure_fraction"] = (
        summary["pareto_structure_count"] / summary["structure_count"]
    )
    summary["new_non_oracle_risk_selector"] = summary["selector"].str.startswith(
        ("noise_propagation_risk_", "posterior_variance_reference_")
    )
    return structure, summary


def run_structure_statistics(
    config_path: str | Path = "configs/tqe_physical_alignment/campaign.json",
    *,
    physical_audit_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_campaign_config(config_path)
    audit_dir = Path(physical_audit_dir or (Path(config["output_root"]) / "physical_audit"))
    raw = pd.read_csv(audit_dir / "raw_physical_rows.csv")
    if raw["logical_key"].duplicated().any():
        raise RuntimeError("duplicate raw-row logical keys invalidate inference")
    settings = config["statistics"]
    physical_cell = _primary_cell(raw, settings, classification="physical")
    legacy_cell = _primary_cell(raw, settings, classification="legacy_diagnostic")
    combined = pd.concat([physical_cell, legacy_cell], ignore_index=True)
    realization, structure = aggregate_structure_units(combined)
    case = _case_summaries(structure)
    pairwise, bootstrap, loco = _pairwise_summaries(structure, settings)
    floor = _floor_sensitivity(physical_cell, {**settings, **config["physical_audit"]})
    resource_structure, pareto = _resource_pareto_summaries(raw, settings)
    support_pair = _paired_values(
        structure,
        "sensitivity_refined_mean",
        "global_magnitude",
        metric="E_support",
    )
    support_ordinary = bootstrap_paired_effect(
        support_pair,
        replicates=int(settings["bootstrap_replicates"]),
        seed=int(settings["bootstrap_seed"]) + 50_000,
        confidence_level=float(settings["confidence_level"]),
    )
    support_stratified = case_stratified_bootstrap_effect(
        support_pair,
        replicates=int(settings["bootstrap_replicates"]),
        seed=int(settings["case_stratified_bootstrap_seed"]) + 50_000,
        confidence_level=float(settings["confidence_level"]),
    )
    support_pairwise = pd.DataFrame(
        [
            {
                "candidate_selector": "sensitivity_refined_mean",
                "baseline_selector": "global_magnitude",
                "metric": "E_support",
                "effect_definition": "baseline E_support minus candidate E_support",
                "paired_structure_count": len(support_pair),
                "observed_mean_effect": support_ordinary.observed_effect,
                "fraction_structures_favor_candidate": float(support_pair["effect"].gt(0.0).mean()),
                "ordinary_ci_low": support_ordinary.confidence_interval_low,
                "ordinary_ci_high": support_ordinary.confidence_interval_high,
                "case_stratified_ci_low": support_stratified.confidence_interval_low,
                "case_stratified_ci_high": support_stratified.confidence_interval_high,
                "bootstrap_replicates": int(settings["bootstrap_replicates"]),
                "resampling_unit": "structural_group_id",
            }
        ]
    )

    atomic_write_csv(audit_dir / "realization_level_summary.csv", realization)
    atomic_write_csv(audit_dir / "structure_level_summary.csv", structure)
    atomic_write_csv(audit_dir / "case_level_summary.csv", case)
    atomic_write_csv(audit_dir / "selector_pairwise_summary.csv", pairwise)
    atomic_write_json(audit_dir / "bootstrap_summary.json", bootstrap)
    atomic_write_csv(audit_dir / "leave_one_case_out.csv", loco)
    atomic_write_csv(audit_dir / "floor_sensitivity.csv", floor)
    atomic_write_csv(audit_dir / "resource_structure_summary.csv", resource_structure)
    atomic_write_csv(audit_dir / "pareto_summary.csv", pareto)
    atomic_write_csv(audit_dir / "support_fidelity_pairwise_summary.csv", support_pairwise)

    primary_structure = structure.loc[
        structure["functional_classification"].eq("physical")
        & structure["summary_scope"].eq("all_families")
        & structure["metric"].eq("E_physical")
    ]
    validation = {
        "study_id": STUDY_ID,
        "configuration_id": config["configuration_id"],
        "configuration_hash": config["configuration_hash"],
        "primary_independent_unit": "structural_group_id",
        "independent_structure_count": int(primary_structure["structural_group_id"].nunique()),
        "case_counts": (
            primary_structure[["structural_group_id", "ieee_case"]]
            .drop_duplicates()
            .groupby("ieee_case")
            .size()
            .to_dict()
        ),
        "realizations_are_resampling_units": False,
        "residual_functional_rows_are_resampling_units": False,
        "bootstrap_replicates": int(settings["bootstrap_replicates"]),
        "pairwise_comparisons_completed": int(pairwise["status"].eq("completed").sum()),
        "all_structure_rows_have_two_realizations": bool(
            structure["realization_count"].eq(2).all()
        ),
        "case_stratified_bootstrap_preserves_case_composition": True,
        "confidence_intervals_reproducible_from_fixed_seeds": True,
        "resource_budget_cell_count": int(
            resource_structure[["analysis_k_budget", "analysis_slot_budget"]]
            .drop_duplicates()
            .shape[0]
        ),
        "new_risk_pareto_structure_count": int(
            resource_structure.loc[
                resource_structure["selector"].str.startswith(
                    ("noise_propagation_risk_", "posterior_variance_reference_")
                ),
                "physical_error_resource_pareto_nondominated",
            ].sum()
        ),
        "sensitivity_support_fidelity_effect": support_ordinary.observed_effect,
        "sensitivity_support_fidelity_case_stratified_ci": [
            support_stratified.confidence_interval_low,
            support_stratified.confidence_interval_high,
        ],
    }
    atomic_write_json(audit_dir / "statistics_validation.json", validation)
    lines = [
        "# Structure-Level Statistical Validation",
        "",
        f"- Independent structures: {validation['independent_structure_count']}",
        f"- Case composition: {validation['case_counts']}",
        f"- Bootstrap replicates per comparison: {validation['bootstrap_replicates']}",
        "- Resampling unit: structural group.",
        "- Residual seeds, functionals, numerical realizations, budgets, and CSV rows are not "
        "resampling units.",
        "- Frozen aggregation: median within each realization, then mean across its two "
        "realizations.",
        "- Physical and legacy-diagnostic functionals are summarized under explicit, separate "
        "classifications.",
        "",
    ]
    atomic_write_text(audit_dir / "statistics_validation_report.md", "\n".join(lines))
    return validation
