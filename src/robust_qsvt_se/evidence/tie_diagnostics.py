"""Deterministic explanations for frozen structural primary-test ties."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .canonical_registry import (
    RESULT_FIELDS,
    _canonical_record,
    _read_csv,
    _read_primary_heldout,
    atomic_write_csv,
    atomic_write_json,
    load_json,
    stable_json_fingerprint,
)

TieDiagnosticCategory = Literal[
    "full_support_saturation",
    "identical_selected_support",
    "identical_effective_matrix",
    "near_zero_reference_output",
    "functional_insensitivity",
    "numerically_negligible_difference",
    "genuine_selector_tie",
    "mixed_or_ambiguous_tie",
    "not_a_tie",
]


@dataclass(frozen=True, slots=True)
class TieDiagnosticInputs:
    original_outcome: str
    full_support_saturation: bool
    support_fingerprints_equal: bool
    effective_matrices_equal: bool
    near_zero_reference: bool
    functional_insensitivity: bool
    numerically_negligible_difference: bool
    frozen_tie_rule_holds: bool


def classify_primary_tie(inputs: TieDiagnosticInputs) -> TieDiagnosticCategory:
    """Apply the frozen explanatory precedence without changing the outcome."""

    if inputs.original_outcome != "tie":
        return "not_a_tie"
    if inputs.full_support_saturation:
        return "full_support_saturation"
    if inputs.support_fingerprints_equal:
        return "identical_selected_support"
    if inputs.effective_matrices_equal:
        return "identical_effective_matrix"
    if inputs.near_zero_reference:
        return "near_zero_reference_output"
    if inputs.functional_insensitivity:
        return "functional_insensitivity"
    if inputs.numerically_negligible_difference:
        return "numerically_negligible_difference"
    if inputs.frozen_tie_rule_holds:
        return "genuine_selector_tie"
    return "mixed_or_ambiguous_tie"


def _support_payload_matrix(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, set[int]]:
    payload = load_json(path)
    matrix = np.zeros(shape, dtype=float)
    cells: set[int] = set()
    for item in payload["selected_entries"]:
        row = int(item["row"])
        column = int(item["column"])
        matrix[row, column] = float(item["value"])
        cells.add(row * shape[1] + column)
    return matrix, cells


def _frozen_tie(
    baseline_error: float,
    candidate_error: float,
    relative_tolerance: float,
    epsilon: float,
) -> bool:
    scale = max(abs(baseline_error), abs(candidate_error), epsilon)
    return abs(candidate_error - baseline_error) <= max(epsilon, relative_tolerance * scale)


def build_tie_diagnostics(root: Path, output_dir: Path) -> pd.DataFrame:
    """Classify every realization while preserving the 6/5/1 group outcome."""

    family = root / "outputs/output_aware_structural_generalization"
    study = load_json(family / "study_configuration.json")
    closure_config = load_json(root / "configs/final_contribution_evidence.json")
    rules = closure_config["tie_diagnostics"]
    comparison = study["primary_comparison"]
    baseline = comparison["baseline_selector"]
    candidate = comparison["candidate_selector"]
    k_budget = int(comparison["k_budget"])
    slot_budget = int(comparison["slot_budget"])
    near_zero_threshold = float(study["near_zero_output_threshold"])
    matrix_atol = float(rules["matrix_absolute_tolerance"])
    matrix_rtol = float(rules["matrix_relative_tolerance"])
    functional_atol = float(rules["functional_absolute_tolerance"])
    floating_multiplier = float(rules["floating_scale_multiplier"])

    instances = _read_csv(family / "instance_registry.csv")
    supports = _read_csv(family / "support_registry.csv")
    primary_supports = supports[
        supports["selector"].isin([baseline, candidate])
        & (supports["k_budget"] == k_budget)
        & (supports["slot_budget"] == slot_budget)
        & (supports["status"] == "completed")
    ]
    pairs = _read_csv(family / "structural_primary_matched_pairs.csv")
    heldout = _read_primary_heldout(
        family / "heldout_results.csv", baseline, candidate, k_budget, slot_budget
    )
    rows: list[dict[str, object]] = []
    for _, instance in instances.sort_values("instance_id").iterrows():
        instance_id = str(instance["instance_id"])
        group_id = str(instance["structural_group_id"])
        group_pair = pairs[pairs["structural_group_id"] == group_id].iloc[0]
        support_pair = primary_supports[primary_supports["instance_id"] == instance_id]
        magnitude = support_pair[support_pair["selector"] == baseline].iloc[0]
        sensitivity = support_pair[support_pair["selector"] == candidate].iloc[0]
        shape_values = json.loads(str(instance["matrix_shape"]))
        shape = (int(shape_values[0]), int(shape_values[1]))
        magnitude_path = family / str(magnitude["support_file"])
        sensitivity_path = family / str(sensitivity["support_file"])
        magnitude_matrix, magnitude_cells = _support_payload_matrix(magnitude_path, shape)
        sensitivity_matrix, sensitivity_cells = _support_payload_matrix(sensitivity_path, shape)
        intersection = len(magnitude_cells & sensitivity_cells)
        union = len(magnitude_cells | sensitivity_cells)
        jaccard = intersection / union if union else 1.0
        difference = magnitude_matrix - sensitivity_matrix
        difference_fro = float(np.linalg.norm(difference, ord="fro"))
        difference_spectral = float(np.linalg.norm(difference, ord=2))
        matrix_scale = max(
            float(np.linalg.norm(magnitude_matrix, ord="fro")),
            float(np.linalg.norm(sensitivity_matrix, ord="fro")),
            np.finfo(float).tiny,
        )
        matrices_equal = difference_fro <= matrix_atol + matrix_rtol * matrix_scale

        task_rows = heldout[heldout["instance_id"] == instance_id]
        magnitude_tasks = task_rows[task_rows["selector"] == baseline]
        sensitivity_tasks = task_rows[task_rows["selector"] == candidate]
        merged = magnitude_tasks.merge(
            sensitivity_tasks,
            on="task_id",
            suffixes=("_magnitude", "_sensitivity"),
            validate="one_to_one",
        )
        reference_magnitude = float(merged["full_ridge_output_magnitude"].abs().median())
        magnitude_output = float(merged["sparse_ridge_output_magnitude"].median())
        sensitivity_output = float(merged["sparse_ridge_output_sensitivity"].median())
        selector_difference = float(
            (merged["sparse_ridge_output_magnitude"] - merged["sparse_ridge_output_sensitivity"])
            .abs()
            .median()
        )
        normalized_difference = selector_difference / max(reference_magnitude, near_zero_threshold)
        magnitude_absolute_error = float(merged["absolute_error_magnitude"].median())
        sensitivity_absolute_error = float(merged["absolute_error_sensitivity"].median())
        magnitude_normalized_error = float(merged["normalized_error_magnitude"].median())
        sensitivity_normalized_error = float(merged["normalized_error_sensitivity"].median())
        candidate_nonzeros = int(instance["candidate_nonzeros"])
        magnitude_nonzeros = int(magnitude["actual_nonzeros"])
        sensitivity_nonzeros = int(sensitivity["actual_nonzeros"])
        full_support = (
            magnitude_nonzeros == candidate_nonzeros and sensitivity_nonzeros == candidate_nonzeros
        )
        support_equal = str(magnitude["support_fingerprint"]) == str(
            sensitivity["support_fingerprint"]
        )
        near_zero = reference_magnitude < near_zero_threshold
        insensitive = (
            not support_equal
            and magnitude_absolute_error <= functional_atol
            and sensitivity_absolute_error <= functional_atol
        )
        floating_scale = (
            floating_multiplier
            * np.finfo(float).eps
            * max(
                reference_magnitude,
                abs(magnitude_output),
                abs(sensitivity_output),
                np.finfo(float).tiny,
            )
        )
        negligible = selector_difference <= floating_scale
        frozen_tie = _frozen_tie(
            float(group_pair["baseline_group_normalized_error"]),
            float(group_pair["candidate_group_normalized_error"]),
            float(comparison["tie_relative_tolerance"]),
            float(comparison["tie_epsilon"]),
        )
        category = classify_primary_tie(
            TieDiagnosticInputs(
                original_outcome=str(group_pair["outcome"]),
                full_support_saturation=full_support,
                support_fingerprints_equal=support_equal,
                effective_matrices_equal=matrices_equal,
                near_zero_reference=near_zero,
                functional_insensitivity=insensitive,
                numerically_negligible_difference=negligible,
                frozen_tie_rule_holds=frozen_tie,
            )
        )
        rows.append(
            {
                "structural_group_id": group_id,
                "instance_id": instance_id,
                "realization_order": int(instance["realization_order"]),
                "ieee_case": instance["ieee_case"],
                "original_outcome": group_pair["outcome"],
                "diagnostic_class": category,
                "full_candidate_nonzeros": candidate_nonzeros,
                "primary_k_budget": k_budget,
                "primary_slot_budget": slot_budget,
                "magnitude_support_nonzeros": magnitude_nonzeros,
                "sensitivity_support_nonzeros": sensitivity_nonzeros,
                "magnitude_support_fingerprint": magnitude["support_fingerprint"],
                "sensitivity_support_fingerprint": sensitivity["support_fingerprint"],
                "support_fingerprints_equal": support_equal,
                "support_jaccard": jaccard,
                "matrix_difference_frobenius": difference_fro,
                "matrix_difference_spectral": difference_spectral,
                "full_output_magnitude": reference_magnitude,
                "magnitude_output": magnitude_output,
                "sensitivity_output": sensitivity_output,
                "selector_output_difference": selector_difference,
                "normalized_difference": normalized_difference,
                "magnitude_absolute_error_median": magnitude_absolute_error,
                "sensitivity_absolute_error_median": sensitivity_absolute_error,
                "magnitude_normalized_error_median": magnitude_normalized_error,
                "sensitivity_normalized_error_median": sensitivity_normalized_error,
                "near_zero_threshold": near_zero_threshold,
                "near_zero_reference_flag": near_zero,
                "full_support_saturation_flag": full_support,
                "identical_effective_matrix_flag": matrices_equal,
                "functional_insensitivity_flag": insensitive,
                "numerically_negligible_difference_flag": negligible,
                "frozen_tie_rule_holds": frozen_tie,
                "informative_pruning_flag": not full_support,
                "classification_precedence_version": "tie_diagnostic_precedence_v1",
            }
        )
    frame = pd.DataFrame(rows)
    atomic_write_csv(
        output_dir / "primary_tie_diagnostics.csv",
        frame.to_dict(orient="records"),
        frame.columns.tolist(),
    )
    group_rows = (
        frame.sort_values(["structural_group_id", "realization_order"])
        .groupby("structural_group_id", sort=True)
        .first()
        .reset_index()
    )
    category_counts = {
        category: int((group_rows["diagnostic_class"] == category).sum())
        for category in (
            "full_support_saturation",
            "identical_selected_support",
            "identical_effective_matrix",
            "near_zero_reference_output",
            "functional_insensitivity",
            "numerically_negligible_difference",
            "genuine_selector_tie",
            "mixed_or_ambiguous_tie",
            "not_a_tie",
        )
    }
    original_counts = {
        outcome: int(pairs["outcome"].astype(str).eq(outcome).sum())
        for outcome in ("win", "tie", "loss")
    }
    summary = {
        "schema_version": 1,
        "classification_precedence": rules["classification_precedence"],
        "matrix_absolute_tolerance": matrix_atol,
        "matrix_relative_tolerance": matrix_rtol,
        "functional_absolute_tolerance": functional_atol,
        "floating_scale_multiplier": floating_multiplier,
        "near_zero_threshold": near_zero_threshold,
        "realization_rows": len(frame),
        "structural_groups": len(group_rows),
        "original_primary_win_tie_loss": original_counts,
        "diagnostic_category_counts_by_group": category_counts,
        "diagnostic_category_counts_by_realization": {
            category: int((frame["diagnostic_class"] == category).sum())
            for category in category_counts
        },
        "saturated_ties": category_counts["full_support_saturation"],
        "identical_support_ties": category_counts["identical_selected_support"],
        "near_zero_ties": category_counts["near_zero_reference_output"],
        "genuine_selector_ties": category_counts["genuine_selector_tie"],
        "informative_pruning_groups": int(group_rows["informative_pruning_flag"].sum()),
        "noninformative_full_support_groups": int(group_rows["full_support_saturation_flag"].sum()),
        "primary_result_unchanged": original_counts == {"win": 6, "tie": 5, "loss": 1},
        "status": "pass"
        if original_counts == {"win": 6, "tie": 5, "loss": 1}
        and category_counts["mixed_or_ambiguous_tie"] == 0
        else "blocking_failure",
    }
    atomic_write_json(output_dir / "primary_tie_diagnostic_summary.json", summary)
    _append_tie_results(output_dir, frame, instances)
    return frame


def _append_tie_results(
    output_dir: Path, diagnostics: pd.DataFrame, instances: pd.DataFrame
) -> None:
    registry_path = output_dir / "canonical_result_registry.csv"
    registry = _read_csv(registry_path, dtype=str)
    existing = set(registry["result_id"])
    new_rows: list[dict[str, object]] = []
    for group_id, group in diagnostics.groupby("structural_group_id", sort=True):
        result_id = f"res:tie:{group_id}:diagnostic_class"
        if result_id in existing:
            continue
        instance_fps = instances.loc[
            instances["structural_group_id"] == group_id, "matrix_fingerprint"
        ].astype(str)
        classes = sorted(group["diagnostic_class"].astype(str).unique())
        value = classes[0] if len(classes) == 1 else "mixed_or_ambiguous_tie"
        first = group.iloc[0]
        new_rows.append(
            _canonical_record(
                result_id=result_id,
                claim_family="structural_generalization",
                experiment_family="tie_diagnostics",
                configuration_id="cfg:output_aware_structural_generalization:study",
                source_artifact=Path(
                    "outputs/final_contribution_evidence/primary_tie_diagnostics.csv"
                ),
                source_row_locator=f"rows[structural_group_id={group_id}]",
                evidence_tier="diagnostic_only",
                matrix_fingerprint=stable_json_fingerprint(sorted(instance_fps)),
                ieee_case=str(first["ieee_case"]),
                structural_group_id=str(group_id),
                value=value,
                unit="tie_diagnostic_category",
                limitation_code="small_scale_8x8;case_dependence;functional_dependence",
                notes="Descriptive secondary diagnosis; frozen primary outcome unchanged.",
            ).csv_row()
        )
    if new_rows:
        combined = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.sort_values("result_id").reset_index(drop=True)
        atomic_write_csv(registry_path, combined.to_dict(orient="records"), RESULT_FIELDS)


def _task_outcome(
    baseline_error: float,
    candidate_error: float,
    relative_tolerance: float,
    epsilon: float,
) -> str:
    if _frozen_tie(baseline_error, candidate_error, relative_tolerance, epsilon):
        return "tie"
    return "win" if candidate_error < baseline_error else "loss"


def build_near_zero_audit(root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit primary held-out rows without excluding near-zero references."""

    family = root / "outputs/output_aware_structural_generalization"
    study = load_json(family / "study_configuration.json")
    comparison = study["primary_comparison"]
    baseline = comparison["baseline_selector"]
    candidate = comparison["candidate_selector"]
    near_zero_threshold = float(study["near_zero_output_threshold"])
    heldout = _read_primary_heldout(
        family / "heldout_results.csv",
        baseline,
        candidate,
        int(comparison["k_budget"]),
        int(comparison["slot_budget"]),
    )
    baseline_rows = heldout[heldout["selector"] == baseline]
    candidate_rows = heldout[heldout["selector"] == candidate]
    paired = baseline_rows.merge(
        candidate_rows,
        on="task_id",
        suffixes=("_magnitude", "_sensitivity"),
        validate="one_to_one",
    )
    outcomes = {
        str(row["task_id"]): _task_outcome(
            float(row["normalized_error_magnitude"]),
            float(row["normalized_error_sensitivity"]),
            float(comparison["tie_relative_tolerance"]),
            float(comparison["tie_epsilon"]),
        )
        for _, row in paired.iterrows()
    }
    audit = heldout[
        [
            "task_id",
            "instance_id",
            "ieee_case",
            "structural_group_id",
            "realization_order",
            "residual_seed",
            "functional_id",
            "selector",
            "full_ridge_output",
            "sparse_ridge_output",
            "absolute_error",
            "normalized_error",
            "failure_above_frozen_threshold",
        ]
    ].copy()
    audit = audit.rename(columns={"full_ridge_output": "reference_output"})
    audit["near_zero_threshold"] = near_zero_threshold
    audit["near_zero_flag"] = audit["reference_output"].abs() < near_zero_threshold
    audit["paired_sensitivity_vs_magnitude_outcome"] = audit["task_id"].map(outcomes)
    audit["included_in_original_primary_evidence"] = True
    audit = audit.sort_values(["task_id", "selector"]).reset_index(drop=True)
    atomic_write_csv(
        output_dir / "near_zero_output_audit.csv",
        audit.to_dict(orient="records"),
        audit.columns.tolist(),
    )

    summary_rows: list[dict[str, object]] = []
    scope_masks = {
        "all_outputs": pd.Series(True, index=audit.index),
        "near_zero_outputs": audit["near_zero_flag"],
        "non_near_zero_outputs": ~audit["near_zero_flag"],
    }
    unique_tasks_total = audit["task_id"].nunique()
    for scope, scope_mask in scope_masks.items():
        scoped = audit.loc[scope_mask]
        for selector, group in scoped.groupby("selector", sort=True):
            summary_rows.append(
                {
                    "scope": scope,
                    "summary_type": "selector_error",
                    "selector": selector,
                    "record_count": len(group),
                    "unique_task_count": group["task_id"].nunique(),
                    "unique_task_fraction": (
                        group["task_id"].nunique() / unique_tasks_total
                        if unique_tasks_total
                        else math.nan
                    ),
                    "median_absolute_error": float(group["absolute_error"].median()),
                    "p90_absolute_error": float(group["absolute_error"].quantile(0.9)),
                    "median_normalized_error": float(group["normalized_error"].median()),
                    "p90_normalized_error": float(group["normalized_error"].quantile(0.9)),
                    "failure_fraction": float(
                        group["failure_above_frozen_threshold"].astype(bool).mean()
                    ),
                    "sensitivity_wins": "",
                    "ties": "",
                    "sensitivity_losses": "",
                    "near_zero_threshold": near_zero_threshold,
                    "original_rows_removed": 0,
                }
            )
        scoped_tasks = scoped.drop_duplicates("task_id")
        counts = scoped_tasks["paired_sensitivity_vs_magnitude_outcome"].value_counts()
        summary_rows.append(
            {
                "scope": scope,
                "summary_type": "paired_outcome",
                "selector": f"{candidate}_vs_{baseline}",
                "record_count": len(scoped_tasks),
                "unique_task_count": scoped_tasks["task_id"].nunique(),
                "unique_task_fraction": (
                    scoped_tasks["task_id"].nunique() / unique_tasks_total
                    if unique_tasks_total
                    else math.nan
                ),
                "median_absolute_error": "",
                "p90_absolute_error": "",
                "median_normalized_error": "",
                "p90_normalized_error": "",
                "failure_fraction": "",
                "sensitivity_wins": int(counts.get("win", 0)),
                "ties": int(counts.get("tie", 0)),
                "sensitivity_losses": int(counts.get("loss", 0)),
                "near_zero_threshold": near_zero_threshold,
                "original_rows_removed": 0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(
        output_dir / "near_zero_output_summary.csv",
        summary.to_dict(orient="records"),
        summary.columns.tolist(),
    )
    _append_near_zero_results(output_dir, audit, summary)
    return audit, summary


def _append_near_zero_results(output_dir: Path, audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    registry_path = output_dir / "canonical_result_registry.csv"
    registry = _read_csv(registry_path, dtype=str)
    existing = set(registry["result_id"])
    rows: list[dict[str, object]] = []
    unique = audit.drop_duplicates("task_id")
    values: list[tuple[str, float, str]] = [
        (
            "res:near_zero:task_count",
            float(unique["near_zero_flag"].sum()),
            "near_zero_tasks",
        ),
        (
            "res:near_zero:non_near_zero_task_count",
            float((~unique["near_zero_flag"]).sum()),
            "non_near_zero_tasks",
        ),
    ]
    paired = summary[
        (summary["scope"] == "near_zero_outputs") & (summary["summary_type"] == "paired_outcome")
    ].iloc[0]
    for name, column in (
        ("wins", "sensitivity_wins"),
        ("ties", "ties"),
        ("losses", "sensitivity_losses"),
    ):
        values.append((f"res:near_zero:{name}", float(paired[column]), "tasks"))
    for result_id, value, unit in values:
        if result_id in existing:
            continue
        rows.append(
            _canonical_record(
                result_id=result_id,
                claim_family="functional_dependence",
                experiment_family="near_zero_output_audit",
                configuration_id="cfg:output_aware_structural_generalization:study",
                source_artifact=Path(
                    "outputs/final_contribution_evidence/near_zero_output_summary.csv"
                ),
                source_row_locator=f"derived_summary[{result_id}]",
                evidence_tier="diagnostic_only",
                matrix_fingerprint=stable_json_fingerprint(
                    sorted(audit["instance_id"].astype(str).unique())
                ),
                value=value,
                unit=unit,
                limitation_code="small_scale_8x8;functional_dependence;controlled_generated_residuals",
                notes="Near-zero outputs remain in the full primary evidence.",
            ).csv_row()
        )
    if rows:
        combined = pd.concat([registry, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.sort_values("result_id").reset_index(drop=True)
        atomic_write_csv(registry_path, combined.to_dict(orient="records"), RESULT_FIELDS)
