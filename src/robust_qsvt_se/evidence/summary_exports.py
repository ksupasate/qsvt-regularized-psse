"""Artifact-driven tables and figure-data exports from canonical registries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .canonical_registry import _read_csv, atomic_write_csv

PROVENANCE_COLUMNS = ["source_result_ids", "source_artifacts", "configuration_ids"]


def _registry_lookup(output_dir: Path) -> pd.DataFrame:
    frame = _read_csv(output_dir / "canonical_result_registry.csv", dtype=str).set_index(
        "result_id", drop=False
    )
    frame.index.name = None
    return frame


def _provenance(registry: pd.DataFrame, result_ids: Iterable[str]) -> dict[str, str]:
    ids = list(dict.fromkeys(str(item) for item in result_ids))
    missing = [item for item in ids if item not in registry.index]
    if missing:
        raise KeyError(f"summary references missing canonical results: {missing[:5]}")
    rows = registry.loc[ids]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return {
        "source_result_ids": ";".join(ids),
        "source_artifacts": ";".join(dict.fromkeys(rows["source_artifact"].astype(str).tolist())),
        "configuration_ids": ";".join(dict.fromkeys(rows["configuration_id"].astype(str).tolist())),
    }


def _write(path: Path, frame: pd.DataFrame) -> None:
    for column in PROVENANCE_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"table {path.name} lacks provenance column {column}")
    atomic_write_csv(path, frame.to_dict(orient="records"), frame.columns.tolist())


def _primary_structural_table(root: Path, output_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    family = root / "outputs/output_aware_structural_generalization"
    pairs = _read_csv(family / "structural_primary_matched_pairs.csv")
    diagnostics = _read_csv(output_dir / "primary_tie_diagnostics.csv")
    group_diag = (
        diagnostics.sort_values(["structural_group_id", "realization_order"])
        .groupby("structural_group_id", sort=True)
        .first()
        .reset_index()
    )
    frame = pairs.merge(
        group_diag[
            [
                "structural_group_id",
                "diagnostic_class",
                "support_fingerprints_equal",
                "full_support_saturation_flag",
                "near_zero_reference_flag",
            ]
        ],
        on="structural_group_id",
        how="left",
        validate="one_to_one",
    )
    provenance = []
    for group_id in frame["structural_group_id"]:
        provenance.append(
            _provenance(
                registry,
                [
                    f"res:structural:primary:{group_id}:paired_difference",
                    f"res:tie:{group_id}:diagnostic_class",
                ],
            )
        )
    return pd.concat([frame, pd.DataFrame(provenance)], axis=1)


def _case_generalization_table(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    pairs = _read_csv(
        root / "outputs/output_aware_structural_generalization/structural_primary_matched_pairs.csv"
    )
    rows: list[dict[str, Any]] = []
    for case, group in pairs.groupby("ieee_case", sort=True):
        ids = [
            f"res:structural:primary:{group_id}:paired_difference"
            for group_id in group["structural_group_id"]
        ]
        rows.append(
            {
                "ieee_case": case,
                "structural_groups": len(group),
                "wins": int((group["outcome"] == "win").sum()),
                "ties": int((group["outcome"] == "tie").sum()),
                "losses": int((group["outcome"] == "loss").sum()),
                "baseline_median_normalized_error": float(
                    group["baseline_group_normalized_error"].median()
                ),
                "candidate_median_normalized_error": float(
                    group["candidate_group_normalized_error"].median()
                ),
                "median_paired_difference": float(
                    group["paired_difference_candidate_minus_baseline"].median()
                ),
                **_provenance(registry, ids),
            }
        )
    return pd.DataFrame(rows)


def _functional_table(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    pairs = _read_csv(
        root
        / "outputs/output_aware_structural_generalization/structural_primary_functional_pairs.csv"
    )
    rows: list[dict[str, Any]] = []
    for functional_id, group in pairs.groupby("functional_id", sort=True):
        ids = [
            f"res:structural:functional:{row.structural_group_id}:{functional_id}:paired_difference"
            for row in group.itertuples()
        ]
        rows.append(
            {
                "functional_id": functional_id,
                "wins": int((group["outcome"] == "win").sum()),
                "ties": int((group["outcome"] == "tie").sum()),
                "losses": int((group["outcome"] == "loss").sum()),
                "baseline_median_normalized_error": float(
                    group["baseline_group_normalized_error"].median()
                ),
                "candidate_median_normalized_error": float(
                    group["candidate_group_normalized_error"].median()
                ),
                "median_paired_difference": float(
                    group["paired_difference_candidate_minus_baseline"].median()
                ),
                **_provenance(registry, ids),
            }
        )
    return pd.DataFrame(rows)


def _error_hierarchy_table(registry: pd.DataFrame) -> pd.DataFrame:
    dominant = registry[registry["result_id"].str.startswith("res:error_decomposition:dominant:")]
    rows = []
    for _, record in dominant.sort_values("functional_id").iterrows():
        rows.append(
            {
                "functional_id": record["functional_id"],
                "dominant_error_source": record["value"],
                **_provenance(registry, [record["result_id"]]),
            }
        )
    return pd.DataFrame(rows)


def _integrated_table(registry: pd.DataFrame) -> pd.DataFrame:
    ids = [
        "res:integrated:matrix_shape",
        "res:integrated:matrix_nonzeros",
        "res:integrated:polynomial_degree",
        "res:integrated:phase_count",
        "res:integrated:postselection_probability",
        "res:integrated:sparse_dense_action_error",
        "res:integrated:qsvt_exact_svt_action_error",
        "res:integrated:selected_output_qsvt_error",
        "res:integrated:finite_shot_coordinate_1e6",
        "res:integrated:resource:transpiled_gate_count",
        "res:integrated:resource:transpiled_depth",
        "res:integrated:resource:toffoli_count",
        "res:integrated:resource:controlled_rotation_count",
    ]
    row = {record_id.split(":")[-1]: registry.loc[record_id, "value"] for record_id in ids}
    row.update(_provenance(registry, ids))
    return pd.DataFrame([row])


def _common_design_table(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    path = root / "outputs/output_aware_structural_generalization/qsvt_validation_results.csv"
    frame = _read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        result_id = (
            f"res:output_aware_structural_generalization:qsvt:{item['instance_id']}:"
            f"{item['support_id']}:{item['functional_id']}:error"
        )
        rows.append(
            {
                "structural_group_id": item["structural_group_id"],
                "instance_id": item["instance_id"],
                "ieee_case": item["ieee_case"],
                "selector": item["selector"],
                "functional_id": item["functional_id"],
                "full_ridge_output": item["full_matrix_ridge_output"],
                "sparse_ridge_output": item["sparse_matrix_ridge_output"],
                "exact_polynomial_output": item["exact_polynomial_svt_output"],
                "sparse_qsvt_output": item["sparse_qsvt_statevector_output"],
                "support_error": item["support_selection_error"],
                "qsvt_error": item["qsvt_error_on_sparse_matrix"],
                "postselection_probability": item["postselection_probability"],
                "degree": item["degree"],
                "phase_count": item["phase_count"],
                "status": item["status"],
                **_provenance(registry, [result_id]),
            }
        )
    return pd.DataFrame(rows)


def _resource_table(registry: pd.DataFrame) -> pd.DataFrame:
    subset = registry[registry["claim_family"] == "resource_accounting"].copy()
    rows = []
    for _, item in subset.sort_values("result_id").iterrows():
        rows.append(
            {
                "experiment_family": item["experiment_family"],
                "metric": item["result_id"].split(":")[-2]
                if item["result_id"].endswith((":minimum", ":maximum"))
                else item["result_id"].split(":")[-1],
                "bound": item["result_id"].split(":")[-1]
                if item["result_id"].endswith((":minimum", ":maximum"))
                else "value",
                "value": item["value"],
                "unit": item["unit"],
                "evidence_tier": item["evidence_tier"],
                **_provenance(registry, [item["result_id"]]),
            }
        )
    return pd.DataFrame(rows)


def _conditioning_table(output_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(output_dir / "regularized_conditioning_summary.csv")
    rows = []
    for _, item in frame.iterrows():
        matching = registry[
            (registry["experiment_family"] == "regularized_conditioning_audit")
            & (registry["ieee_case"] == str(item["ieee_case"]))
        ]
        ids = matching["result_id"].tolist()
        rows.append({**item.to_dict(), **_provenance(registry, ids)})
    return pd.DataFrame(rows)


def _tie_table(output_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    diagnostics = _read_csv(output_dir / "primary_tie_diagnostics.csv")
    groups = (
        diagnostics.sort_values(["structural_group_id", "realization_order"])
        .groupby("structural_group_id", sort=True)
        .first()
        .reset_index()
    )
    rows = []
    for _, item in groups.iterrows():
        result_id = f"res:tie:{item['structural_group_id']}:diagnostic_class"
        rows.append(
            {
                "structural_group_id": item["structural_group_id"],
                "ieee_case": item["ieee_case"],
                "original_outcome": item["original_outcome"],
                "diagnostic_class": item["diagnostic_class"],
                "support_fingerprints_equal": item["support_fingerprints_equal"],
                "full_support_saturation": item["full_support_saturation_flag"],
                "near_zero_reference": item["near_zero_reference_flag"],
                **_provenance(registry, [result_id]),
            }
        )
    return pd.DataFrame(rows)


def _limitation_table(output_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    limitations = _read_csv(output_dir / "canonical_limitation_registry.csv", dtype=str)
    rows = []
    for _, item in limitations.iterrows():
        ids = [value for value in str(item["affected_result_ids"]).split(";") if value]
        if ids:
            provenance = _provenance(registry, ids)
        else:
            provenance = {
                "source_result_ids": "",
                "source_artifacts": item["evidence_source"],
                "configuration_ids": "",
            }
        rows.append({**item.to_dict(), **provenance})
    return pd.DataFrame(rows)


def build_summary_tables(root: Path, output_dir: Path) -> list[Path]:
    registry = _registry_lookup(output_dir)
    tables = {
        "primary_structural_generalization.csv": _primary_structural_table(
            root, output_dir, registry
        ),
        "case_generalization_summary.csv": _case_generalization_table(root, registry),
        "functional_tradeoff_summary.csv": _functional_table(root, registry),
        "error_hierarchy_summary.csv": _error_hierarchy_table(registry),
        "integrated_sparse_chain_summary.csv": _integrated_table(registry),
        "common_design_qsvt_summary.csv": _common_design_table(root, registry),
        "resource_summary.csv": _resource_table(registry),
        "conditioning_summary.csv": _conditioning_table(output_dir, registry),
        "tie_diagnostic_summary.csv": _tie_table(output_dir, registry),
        "limitation_summary.csv": _limitation_table(output_dir, registry),
    }
    paths = []
    for name, frame in tables.items():
        path = output_dir / "tables" / name
        _write(path, frame)
        paths.append(path)
    return paths


def _with_group_provenance(
    frame: pd.DataFrame, registry: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    provenance = [
        _provenance(
            registry,
            [f"{prefix}{group_id}:paired_difference"],
        )
        for group_id in frame["structural_group_id"]
    ]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(provenance)], axis=1)


def _pareto_representatives(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    family = root / "outputs/output_aware_structural_generalization"
    specifications = (
        ("nonzeros", "pareto_frontier_error_nnz.csv", "actual_nonzeros"),
        ("slots", "pareto_frontier_error_slots.csv", "slot_count"),
        ("gates", "pareto_frontier_error_gates.csv", "signal_unitary_gate_count"),
        ("depth", "pareto_frontier_error_depth.csv", "signal_unitary_depth"),
    )
    rows: list[dict[str, Any]] = []
    source_ids = [
        result_id
        for result_id in registry.index
        if str(result_id).startswith("res:headline:structural:resource:")
    ]
    provenance = _provenance(registry, source_ids)
    for frontier, filename, cost_column in specifications:
        frame = _read_csv(family / filename)
        for case, group in frame.groupby("ieee_case", sort=True):
            ordered = group.sort_values(
                [cost_column, "median_normalized_error", "support_id"]
            ).reset_index(drop=True)
            count = min(5, len(ordered))
            indices = sorted(set(np.linspace(0, len(ordered) - 1, count, dtype=int).tolist()))
            for index in indices:
                item = ordered.iloc[index]
                rows.append(
                    {
                        "frontier": frontier,
                        "ieee_case": case,
                        "representative_index": index,
                        "frontier_size": len(ordered),
                        "selection_rule": "evenly_spaced_indices_after_cost_error_support_sort_v1",
                        "support_id": item["support_id"],
                        "selector": item["selector"],
                        "cost": item[cost_column],
                        "median_normalized_error": item["median_normalized_error"],
                        **provenance,
                    }
                )
    return pd.DataFrame(rows)


def build_figure_data(root: Path, output_dir: Path) -> list[Path]:
    registry = _registry_lookup(output_dir)
    family = root / "outputs/output_aware_structural_generalization"
    paired = _read_csv(family / "structural_primary_matched_pairs.csv")
    paired = _with_group_provenance(paired, registry, "res:structural:primary:")
    case = _case_generalization_table(root, registry)[
        [
            "ieee_case",
            "wins",
            "ties",
            "losses",
            *PROVENANCE_COLUMNS,
        ]
    ]
    functional = _functional_table(root, registry)[
        [
            "functional_id",
            "wins",
            "ties",
            "losses",
            *PROVENANCE_COLUMNS,
        ]
    ]
    hierarchy = _error_hierarchy_table(registry)
    qsvt = _read_csv(family / "qsvt_validation_results.csv")
    support_qsvt_rows = []
    for _, item in qsvt.iterrows():
        result_id = (
            f"res:output_aware_structural_generalization:qsvt:{item['instance_id']}:"
            f"{item['support_id']}:{item['functional_id']}:error"
        )
        support_qsvt_rows.append(
            {
                "structural_group_id": item["structural_group_id"],
                "instance_id": item["instance_id"],
                "ieee_case": item["ieee_case"],
                "selector": item["selector"],
                "functional_id": item["functional_id"],
                "support_error": item["support_selection_error"],
                "qsvt_error": item["qsvt_error_on_sparse_matrix"],
                "total_error": item["total_full_to_qsvt_error"],
                **_provenance(registry, [result_id]),
            }
        )
    support_qsvt = pd.DataFrame(support_qsvt_rows)
    conditioning = _read_csv(output_dir / "regularized_conditioning_audit.csv")
    conditioning_rows = []
    for _, item in conditioning.iterrows():
        matching = registry[
            registry["result_id"].str.contains(
                str(item["matrix_id"]).replace(":", ":matrix:"), regex=False
            )
        ]
        ids = matching["result_id"].tolist()
        conditioning_rows.append(
            {
                "matrix_id": item["matrix_id"],
                "experiment_family": item["experiment_family"],
                "ieee_case": item["ieee_case"],
                "structural_group_id": item["structural_group_id"],
                "rank_deficient": item["rank_deficient"],
                "raw_condition_number": item["raw_condition_number"],
                "regularized_condition_number": item["regularized_condition_number"],
                "max_ridge_filter_response_global": item["max_ridge_filter_response_global"],
                "max_ridge_filter_response_actual": item["max_ridge_filter_response_actual"],
                **_provenance(registry, ids),
            }
        )
    raw_regularized = pd.DataFrame(conditioning_rows)
    figures = {
        "structural_group_paired_errors.csv": paired,
        "case_win_tie_loss.csv": case,
        "functional_win_tie_loss.csv": functional,
        "error_source_hierarchy.csv": hierarchy,
        "support_error_vs_qsvt_error.csv": support_qsvt,
        "resource_error_frontier_representatives.csv": _pareto_representatives(root, registry),
        "raw_vs_regularized_conditioning.csv": raw_regularized,
    }
    paths = []
    for name, frame in figures.items():
        path = output_dir / "figure_data" / name
        _write(path, frame)
        paths.append(path)
    return paths
