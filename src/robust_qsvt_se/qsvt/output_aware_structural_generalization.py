"""Structurally diverse multi-block generalization benchmark for output-aware selection.

The benchmark freezes the previously validated Ridge sensitivity selectors, one-swap
refinement, conservative perturbation certificate, sparse wrapper, and common-design QSVT
implementation, then evaluates them on structurally distinct PSSE-derived blocks whose
measurement rows, state columns, type compositions, and sparsity patterns all change.
Structural blocks are selected deterministically before any selector runs; numerical
realizations within a block are never treated as independent structural systems.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.tqe_revision_support_common import git_commit_hash, now_iso
from robust_qsvt_se.qsvt import output_aware_generalization as _generalization
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.output_aware_generalization import (
    _atomic_write_text,
    _git_status,
    _json_ready,
    _protected_file_snapshot,
    _selected_measurement_records,
    _sha256_file,
    _state_records,
    classify_matched_errors,
    configuration_fingerprint,
    grouped_pareto_frontier,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    SupportSelectionResult,
    atomic_write_csv,
    atomic_write_json,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.qsvt.research_matrix import extract_weighted_jacobian_matrix
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint

DEFAULT_OUTPUT_DIR = Path("outputs/output_aware_structural_generalization")
DEFAULT_CONFIG_PATH = Path("configs/output_aware_structural_generalization.json")
PRIOR_BENCHMARK_DIR = Path("outputs/output_aware_generalization")
STUDY_ID = "output_aware_structural_generalization_v1"
PREVIOUS_BLOCK_LABEL = "previous_evaluation_block"

CANDIDATE_POLICIES = (
    "topology_local",
    "measurement_balanced",
    "angle_dominant",
    "voltage_dominant",
    "mixed_state",
    "injection_heavy",
    "branch_flow_heavy",
    "voltage_measurement_heavy",
    "seeded_random",
    "lexicographic_metadata",
)

STAGES = (
    "audit",
    "freeze",
    "candidates",
    "descriptors",
    "structural-selection",
    "realizations",
    "functionals",
    "residuals",
    "supports",
    "heldout",
    "primary-test",
    "secondary-test",
    "structural-analysis",
    "stability",
    "certificates",
    "resources",
    "pareto",
    "qsvt",
    "finite-shot",
    "summary",
    "verify",
)

FAILURE_CATEGORIES = (
    "candidate_generation_failure",
    "insufficient_structural_diversity",
    "duplicate_previous_block",
    "duplicate_candidate_block",
    "metadata_mapping_failure",
    "matrix_realization_failure",
    "residual_generation_failure",
    "support_budget_infeasible",
    "milp_failure",
    "slot_assignment_failure",
    "certificate_violation",
    "wrapper_reconstruction_failure",
    "common_normalization_failure",
    "polynomial_fit_failure",
    "qsvt_statevector_failure",
    "finite_shot_cost_ceiling_exceeded",
    "resource_compilation_limit",
    "other_verified_failure",
)


class CandidateGenerationError(RuntimeError):
    """Raised when a predeclared candidate policy cannot produce a block."""


def load_structural_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"structural configuration must use study_id={STUDY_ID}")
    if not payload.get("declared_before_benchmark_evaluation", False):
        raise ValueError("benchmark configuration must be declared before evaluation")
    if tuple(payload.get("required_cases", ())) != ("ieee14", "ieee30", "ieee57"):
        raise ValueError("required cases must be IEEE-14, IEEE-30, and IEEE-57")
    if payload["support_budgets"] != [8, 12, 16, 24, 32]:
        raise ValueError("support budgets differ from the frozen development study")
    if payload["slot_budgets"] != [2, 3, 4, 6, 8]:
        raise ValueError("slot budgets differ from the frozen development study")
    if tuple(payload["candidate_pool"]["policies"]) != CANDIDATE_POLICIES:
        raise ValueError("candidate policies differ from the predeclared policy list")
    weights = payload["structural_selection"]["distance_weights"]
    if abs(sum(float(weights[key]) for key in weights) - 1.0) > 1.0e-12:
        raise ValueError("structural distance weights must sum to one")
    if not payload["structural_selection"]["threshold_fixed_before_selector_evaluation"]:
        raise ValueError("diversity threshold must be fixed before selector evaluation")
    return payload


class StructuralCheckpoint:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "checkpoint.json"

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"study_id": STUDY_ID, "stages": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"study_id": STUDY_ID, "stages": {}}
        if payload.get("study_id") != STUDY_ID:
            return {"study_id": STUDY_ID, "stages": {}}
        payload.setdefault("stages", {})
        return payload

    def is_complete(self, stage: str) -> bool:
        record = self.read()["stages"].get(stage, {})
        if record.get("status") != "completed":
            return False
        return all((self.output_dir / path).is_file() for path in record.get("outputs", []))

    def clear(self, stage: str) -> None:
        payload = self.read()
        payload["stages"].pop(stage, None)
        atomic_write_json(self.path, payload)

    def mark_complete(
        self,
        stage: str,
        *,
        outputs: Sequence[str],
        result: Mapping[str, Any],
        elapsed_seconds: float,
    ) -> None:
        payload = self.read()
        payload["stages"][stage] = {
            "status": "completed",
            "completed_at": now_iso(),
            "outputs": list(outputs),
            "result": _json_ready(dict(result)),
            "elapsed_seconds": float(elapsed_seconds),
        }
        payload["last_completed_stage"] = stage
        payload["last_update"] = now_iso()
        atomic_write_json(self.path, payload)


@dataclass(slots=True)
class StructuralContext:
    root: Path
    output_dir: Path
    config_path: Path
    config: dict[str, Any]
    checkpoint: StructuralCheckpoint
    resume: bool
    force: bool
    max_workers: int
    seed: int

    def part_path(self, stage: str, item_id: str, suffix: str = ".json") -> Path:
        return self.output_dir / "checkpoint_parts" / stage / f"{item_id}{suffix}"


def make_context(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    resume: bool = False,
    force: bool = False,
    max_workers: int = 1,
    seed: int | None = None,
) -> StructuralContext:
    root = Path.cwd().resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    config = load_structural_configuration(config_path)
    frozen_seed = int(config["random_objective"]["base_seed"])
    selected_seed = frozen_seed if seed is None else int(seed)
    if selected_seed != frozen_seed:
        raise ValueError(
            f"--seed is frozen at {frozen_seed}; received {selected_seed}. "
            "Create a new predeclared study configuration for another seed."
        )
    return StructuralContext(
        root=root,
        output_dir=destination,
        config_path=Path(config_path),
        config=config,
        checkpoint=StructuralCheckpoint(destination),
        resume=bool(resume),
        force=bool(force),
        max_workers=max(1, int(max_workers)),
        seed=selected_seed,
    )


def _load_prior_frozen_configuration(root: Path) -> dict[str, Any]:
    path = root / PRIOR_BENCHMARK_DIR / "frozen_selector_configuration.json"
    if not path.is_file():
        raise FileNotFoundError(f"previous frozen configuration is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prior_study_configuration(root: Path) -> dict[str, Any]:
    path = root / PRIOR_BENCHMARK_DIR / "study_configuration.json"
    if not path.is_file():
        raise FileNotFoundError(f"previous study configuration is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_frozen_method_configuration(
    config: Mapping[str, Any],
    prior_frozen: Mapping[str, Any],
    prior_study: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy and cross-check every frozen method field from the previous benchmark."""

    if prior_frozen["configuration_fingerprint"] != configuration_fingerprint(prior_frozen):
        raise ValueError("previous frozen configuration fingerprint is corrupt")
    primary = prior_frozen["primary_comparison_budget"]
    secondary = prior_frozen["secondary_comparison_budget"]
    frozen = {
        "previous_configuration_fingerprint": prior_frozen["configuration_fingerprint"],
        "previous_study_id": prior_study["study_id"],
        "previous_study_git_commit": prior_study["git_commit"],
        "sensitivity_formula_version": prior_frozen["sensitivity_score_definition"],
        "entry_score_formula": prior_frozen["sensitivity_score_definition"],
        "score_normalization": prior_frozen["task_score_normalization"],
        "normalization_epsilon": prior_frozen["normalization_epsilon"],
        "normalized_error_floor": prior_frozen["normalized_error_floor"],
        "normalized_error_failure_threshold": prior_frozen["normalized_error_failure_threshold"],
        "training_residual_count": prior_study["residuals"]["training_count_per_instance"],
        "heldout_residual_count": prior_study["residuals"]["held_out_count_per_instance"],
        "support_budgets": prior_frozen["support_budgets"],
        "slot_budgets": prior_frozen["slot_budgets"],
        "coverage_policy": prior_frozen["support_coverage_policy"],
        "milp_options": prior_frozen["milp_solver_options"],
        "refinement_max_iterations": prior_frozen["refinement_max_iterations"],
        "refinement_tolerance": prior_frozen["refinement_improvement_tolerance"],
        "refinement_tie_breaking": prior_frozen["refinement_tie_breaking"],
        "refinement_candidate_order": prior_frozen["refinement_candidate_order"],
        "random_support_count": prior_frozen["random_support_count"],
        "random_support_seed_policy": prior_frozen["random_support_seeds"],
        "certificate_formula_version": prior_frozen["certificate_formula_version"],
        "certificate_validation": prior_frozen["certificate_validation"],
        "regularization_policy": prior_study["regularization"],
        "mean_objective": prior_frozen["mean_objective"],
        "worst_case_objective": prior_frozen["worst_case_objective"],
        "primary_selector": primary["candidate_selector"],
        "primary_baseline": primary["baseline_selector"],
        "primary_k": primary["k_budget"],
        "primary_slot_budget": primary["slot_budget"],
        "secondary_selector": secondary["candidate_selector"],
        "secondary_baseline": secondary["baseline_selector"],
        "secondary_k": secondary["k_budget"],
        "secondary_slot_budget": secondary["slot_budget"],
        "win_tie_loss_tolerance": primary["tie_relative_tolerance"],
        "win_tie_loss_epsilon": primary["tie_epsilon"],
        "bootstrap_seed": primary["bootstrap_seed"],
        "bootstrap_samples": primary["bootstrap_samples"],
        "qsvt_candidate_degrees": prior_frozen["qsvt_candidate_degrees"],
        "qsvt_uniform_tolerance": prior_frozen["qsvt_uniform_error_tolerance"],
        "structural_metric_declaration": {
            "primary_metric": "structural_group_median_heldout_normalized_error",
            "realization_combination": "mean_of_realization_medians",
            "resampling_unit": "structural_group",
            "numerical_realizations_are_not_independent_structures": True,
        },
        "immutable_after_benchmark_evaluation_begins": True,
    }
    cross_checks = {
        "normalization_epsilon": config["score_normalization_epsilon"],
        "normalized_error_floor": config["normalized_error_floor"],
        "normalized_error_failure_threshold": config["normalized_error_failure_threshold"],
        "support_budgets": config["support_budgets"],
        "slot_budgets": config["slot_budgets"],
        "coverage_policy": config["support_coverage_policy"],
        "milp_options": config["milp_solver_options"],
        "refinement_max_iterations": config["refinement"]["max_iterations"],
        "refinement_tolerance": config["refinement"]["strict_improvement_tolerance"],
        "refinement_tie_breaking": config["refinement"]["tie_breaking"],
        "refinement_candidate_order": config["refinement"]["candidate_order"],
        "random_support_count": config["random_objective"]["support_count_per_budget"],
        "random_support_seed_policy": config["random_objective"],
        "certificate_formula_version": config["certificate"]["formula_version"],
        "certificate_validation": config["certificate"],
        "regularization_policy": config["regularization"],
        "mean_objective": config["mean_objective"],
        "worst_case_objective": config["worst_case_objective"],
        "sensitivity_formula_version": config["sensitivity_score_definition"],
        "score_normalization": config["task_score_normalization"],
        "training_residual_count": config["residuals"]["training_count_per_instance"],
        "heldout_residual_count": config["residuals"]["held_out_count_per_instance"],
        "primary_selector": config["primary_comparison"]["candidate_selector"],
        "primary_baseline": config["primary_comparison"]["baseline_selector"],
        "primary_k": config["primary_comparison"]["k_budget"],
        "primary_slot_budget": config["primary_comparison"]["slot_budget"],
        "secondary_selector": config["secondary_comparison"]["candidate_selector"],
        "secondary_baseline": config["secondary_comparison"]["baseline_selector"],
        "secondary_k": config["secondary_comparison"]["k_budget"],
        "secondary_slot_budget": config["secondary_comparison"]["slot_budget"],
        "win_tie_loss_tolerance": config["primary_comparison"]["tie_relative_tolerance"],
        "win_tie_loss_epsilon": config["primary_comparison"]["tie_epsilon"],
        "bootstrap_seed": config["primary_comparison"]["bootstrap_seed"],
        "bootstrap_samples": config["primary_comparison"]["bootstrap_samples"],
        "qsvt_candidate_degrees": config["qsvt"]["candidate_degrees"],
        "qsvt_uniform_tolerance": config["qsvt"]["uniform_approximation_tolerance"],
    }
    for key, config_value in cross_checks.items():
        if frozen[key] != config_value:
            raise ValueError(f"frozen method mismatch for {key}")
    if float(frozen["regularization_policy"]["lambda_ref"]) != float(
        config["regularization"]["lambda_ref"]
    ):
        raise ValueError("frozen method mismatch for lambda_ref")
    frozen["configuration_fingerprint"] = configuration_fingerprint(frozen)
    return frozen


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def deterministic_structural_group_id(case_name: str, order: int) -> str:
    case = str(case_name).lower().replace("-", "").replace("_", "")
    if case not in {"ieee14", "ieee30", "ieee57"}:
        raise ValueError(f"unsupported benchmark case {case_name}")
    if int(order) <= 0:
        raise ValueError("structural-group order must be positive")
    return f"{case}_structural_group_{int(order):02d}"


def deterministic_realization_id(
    structural_group_id: str, realization_order: int, matrix_seed: int
) -> str:
    if int(realization_order) <= 0:
        raise ValueError("realization order must be positive")
    return (
        f"{structural_group_id}_realization_{int(realization_order):02d}_"
        f"seed_{int(matrix_seed)}_8x8"
    )


def _rank_indices(
    candidates: Sequence[int] | np.ndarray,
    *descending_keys: Sequence[float] | np.ndarray,
) -> list[int]:
    indices = [int(value) for value in candidates]
    if not indices:
        return []
    arrays = [np.asarray(values, dtype=np.float64) for values in descending_keys]
    return sorted(
        indices,
        key=lambda index: (*(-float(values[index]) for values in arrays), index),
    )


def _circular_window(values: Sequence[int], *, count: int, offset: int) -> list[int]:
    items = [int(value) for value in values]
    if len(items) < int(count):
        raise CandidateGenerationError(
            f"candidate window needs {count} entries but only {len(items)} are available"
        )
    start = int(offset) % len(items)
    return [items[(start + step) % len(items)] for step in range(int(count))]


def _top_rows_for_columns(
    matrix: np.ndarray,
    columns: Sequence[int],
    *,
    count: int = 8,
    allowed_rows: Sequence[int] | None = None,
) -> list[int]:
    values = np.asarray(matrix, dtype=np.float64)
    candidates = (
        np.arange(values.shape[0], dtype=np.int64)
        if allowed_rows is None
        else np.asarray(allowed_rows, dtype=np.int64)
    )
    restricted = values[:, np.asarray(columns, dtype=np.int64)]
    degrees = np.count_nonzero(restricted, axis=1).astype(np.float64)
    norms = np.linalg.norm(restricted, axis=1)
    global_norms = np.linalg.norm(values, axis=1)
    ranked = _rank_indices(candidates, degrees, norms, global_norms)
    active = [index for index in ranked if degrees[index] > 0]
    if len(active) < int(count):
        raise CandidateGenerationError("fewer than eight active rows for selected columns")
    return active[: int(count)]


def _top_columns_for_rows(
    matrix: np.ndarray,
    rows: Sequence[int],
    *,
    count: int = 8,
    allowed_columns: Sequence[int] | None = None,
) -> list[int]:
    values = np.asarray(matrix, dtype=np.float64)
    candidates = (
        np.arange(values.shape[1], dtype=np.int64)
        if allowed_columns is None
        else np.asarray(allowed_columns, dtype=np.int64)
    )
    restricted = values[np.asarray(rows, dtype=np.int64), :]
    degrees = np.count_nonzero(restricted, axis=0).astype(np.float64)
    norms = np.linalg.norm(restricted, axis=0)
    global_norms = np.linalg.norm(values, axis=0)
    ranked = _rank_indices(candidates, degrees, norms, global_norms)
    active = [index for index in ranked if degrees[index] > 0]
    if len(active) < int(count):
        raise CandidateGenerationError("fewer than eight active columns for selected rows")
    return active[: int(count)]


def _repair_active_block(
    matrix: np.ndarray, rows: Sequence[int], columns: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically replace inactive local rows or columns without outcome data."""

    values = np.asarray(matrix, dtype=np.float64)
    selected_rows = list(dict.fromkeys(int(value) for value in rows))
    selected_columns = list(dict.fromkeys(int(value) for value in columns))
    if len(selected_rows) != 8 or len(selected_columns) != 8:
        raise CandidateGenerationError("candidate policies must select eight unique rows/columns")
    for _iteration in range(8):
        block = values[np.ix_(selected_rows, selected_columns)]
        inactive_rows = np.flatnonzero(~np.any(block != 0.0, axis=1))
        inactive_columns = np.flatnonzero(~np.any(block != 0.0, axis=0))
        if inactive_rows.size == 0 and inactive_columns.size == 0:
            order_rows = np.asarray(sorted(selected_rows), dtype=np.int64)
            order_columns = np.asarray(sorted(selected_columns), dtype=np.int64)
            return values[np.ix_(order_rows, order_columns)], order_rows, order_columns
        if inactive_rows.size:
            replacements = _top_rows_for_columns(values, selected_columns, count=8)
            available = [index for index in replacements if index not in selected_rows]
            for local in inactive_rows:
                if not available:
                    raise CandidateGenerationError("unable to repair inactive candidate row")
                selected_rows[int(local)] = available.pop(0)
        if inactive_columns.size:
            replacements = _top_columns_for_rows(values, selected_rows, count=8)
            available = [index for index in replacements if index not in selected_columns]
            for local in inactive_columns:
                if not available:
                    raise CandidateGenerationError("unable to repair inactive candidate column")
                selected_columns[int(local)] = available.pop(0)
    raise CandidateGenerationError("candidate activity repair did not converge")


def _type_indices(values: Sequence[str], accepted: Iterable[str]) -> list[int]:
    allowed = set(str(value) for value in accepted)
    return [index for index, value in enumerate(values) if str(value) in allowed]


def _state_type_indices(metadata: Mapping[str, Any], state_type: str) -> list[int]:
    angle_count = len(metadata.get("angle_state_buses", []))
    column_count = len(metadata.get("state_labels", []))
    if state_type == "angle":
        return list(range(angle_count))
    if state_type == "voltage":
        return list(range(angle_count, column_count))
    raise ValueError(f"unknown state type {state_type}")


def generate_structural_candidate(
    full_matrix: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    policy: str,
    variant: int,
    case_name: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one predeclared, outcome-independent 8x8 candidate block."""

    values = np.asarray(full_matrix, dtype=np.float64)
    if policy not in CANDIDATE_POLICIES:
        raise CandidateGenerationError(f"unknown candidate policy {policy}")
    if int(variant) < 0:
        raise CandidateGenerationError("candidate variant must be nonnegative")
    row_norms = np.linalg.norm(values, axis=1)
    column_norms = np.linalg.norm(values, axis=0)
    row_rank = _rank_indices(np.arange(values.shape[0]), row_norms)
    column_rank = _rank_indices(np.arange(values.shape[1]), column_norms)
    measurement_types = [str(value) for value in metadata["measurement_types"]]
    measurement_labels = [str(value) for value in metadata["measurement_labels"]]
    state_labels = [str(value) for value in metadata["state_labels"]]
    rows: list[int]
    columns: list[int]

    if policy == "topology_local":
        anchor_positions = np.linspace(0, len(column_rank) - 1, 4, dtype=int)
        anchor = int(column_rank[int(anchor_positions[int(variant) % 4])])
        support = values != 0.0
        overlap = np.sum(support & support[:, [anchor]], axis=0).astype(np.float64)
        anchor_rows = support[:, anchor]
        restricted_norm = np.linalg.norm(values[anchor_rows, :], axis=0)
        columns = _rank_indices(np.arange(values.shape[1]), overlap, restricted_norm, column_norms)
        columns = [anchor, *[value for value in columns if value != anchor]][:8]
        rows = _top_rows_for_columns(values, columns)
    elif policy == "measurement_balanced":
        rows = []
        for measurement_type in sorted(set(measurement_types)):
            candidates = _type_indices(measurement_types, [measurement_type])
            ranked = _rank_indices(candidates, row_norms)
            rows.append(ranked[int(variant) % len(ranked)])
        fill = [value for value in row_rank if value not in rows]
        offset = int(variant) * 3
        rows.extend(_circular_window(fill, count=3, offset=offset))
        columns = _top_columns_for_rows(values, rows)
    elif policy in {"angle_dominant", "voltage_dominant"}:
        state_type = "angle" if policy == "angle_dominant" else "voltage"
        candidates = _state_type_indices(metadata, state_type)
        ranked = _rank_indices(candidates, column_norms)
        columns = _circular_window(ranked, count=8, offset=int(variant) * 3)
        rows = _top_rows_for_columns(values, columns)
    elif policy == "mixed_state":
        angle = _rank_indices(_state_type_indices(metadata, "angle"), column_norms)
        voltage = _rank_indices(_state_type_indices(metadata, "voltage"), column_norms)
        columns = [
            *_circular_window(angle, count=4, offset=int(variant) * 2),
            *_circular_window(voltage, count=4, offset=int(variant) * 2),
        ]
        rows = _top_rows_for_columns(values, columns)
    elif policy in {"injection_heavy", "branch_flow_heavy"}:
        prefix = "" if policy == "injection_heavy" else "branch_flow"
        if policy == "injection_heavy":
            type_pairs = ("p_injection", "q_injection")
        else:
            type_pairs = ("p_branch_flow", "q_branch_flow")
        rows = []
        for measurement_type in type_pairs:
            candidates = _type_indices(measurement_types, [measurement_type])
            ranked = _rank_indices(candidates, row_norms)
            rows.extend(_circular_window(ranked, count=4, offset=int(variant) * 4))
        if len(set(rows)) != 8:
            raise CandidateGenerationError(f"duplicate {prefix} measurement rows")
        columns = _top_columns_for_rows(values, rows)
    elif policy == "voltage_measurement_heavy":
        voltage_rows = _rank_indices(
            _type_indices(measurement_types, ["voltage_magnitude"]), row_norms
        )
        branch_rows = _rank_indices(
            _type_indices(measurement_types, ["p_branch_flow", "q_branch_flow"]),
            row_norms,
        )
        rows = [
            *_circular_window(voltage_rows, count=4, offset=int(variant) * 3),
            *_circular_window(branch_rows, count=4, offset=int(variant) * 5),
        ]
        voltage_state_by_bus = {
            int(record["bus_id"]): int(record["full_state_index"])
            for record in _state_records(metadata, range(values.shape[1]))
            if record["state_type"] == "voltage" and record["bus_id"] is not None
        }
        forced: list[int] = []
        for row in rows[:4]:
            label = measurement_labels[row]
            try:
                bus_id = int(label.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if bus_id in voltage_state_by_bus:
                forced.append(voltage_state_by_bus[bus_id])
        forced = list(dict.fromkeys(forced))
        ranked = _top_columns_for_rows(values, rows, count=8)
        columns = [*forced, *[value for value in ranked if value not in forced]][:8]
    elif policy == "seeded_random":
        pool = config["candidate_pool"]
        case_multiplier = int(pool["seeded_random_case_multipliers"][case_name])
        seed = int(pool["seeded_random_base"]) + case_multiplier + int(variant)
        rng = np.random.default_rng(seed)
        for _attempt in range(int(pool["seeded_random_max_attempts"])):
            rows = rng.choice(values.shape[0], size=8, replace=False).tolist()
            columns = rng.choice(values.shape[1], size=8, replace=False).tolist()
            block = values[np.ix_(rows, columns)]
            if np.all(np.any(block != 0.0, axis=1)) and np.all(np.any(block != 0.0, axis=0)):
                break
        else:
            raise CandidateGenerationError("seeded-random active-block retry limit exceeded")
    else:  # lexicographic_metadata
        row_order = sorted(
            range(values.shape[0]),
            key=lambda index: (measurement_types[index], measurement_labels[index], index),
        )
        angle_count = len(metadata.get("angle_state_buses", []))
        column_order = sorted(
            range(values.shape[1]),
            key=lambda index: (
                "angle" if index < angle_count else "voltage",
                state_labels[index],
                index,
            ),
        )
        rows = _circular_window(row_order, count=8, offset=int(variant) * 11)
        columns = _circular_window(column_order, count=8, offset=int(variant) * 7)

    return _repair_active_block(values, rows, columns)


def build_structural_descriptor(
    *,
    candidate_id: str,
    case_name: str,
    rows: Sequence[int],
    columns: Sequence[int],
    matrix: np.ndarray,
    measurement_metadata: Sequence[Mapping[str, Any]],
    state_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the outcome-independent descriptor used for structural selection."""

    values = np.asarray(matrix, dtype=np.float64)
    support = values != 0.0
    singular_values = np.linalg.svd(values, compute_uv=False)
    measurement_histogram = dict(
        sorted(Counter(str(item["measurement_type"]) for item in measurement_metadata).items())
    )
    state_histogram = dict(
        sorted(Counter(str(item["state_type"]) for item in state_metadata).items())
    )
    full_support_cells = [
        [int(rows[local_row]), int(columns[local_column])]
        for local_row, local_column in np.argwhere(support)
    ]
    row_degrees = np.count_nonzero(support, axis=1).astype(int)
    column_degrees = np.count_nonzero(support, axis=0).astype(int)
    return {
        "candidate_id": str(candidate_id),
        "ieee_case": str(case_name),
        "selected_rows": [int(value) for value in rows],
        "selected_columns": [int(value) for value in columns],
        "row_set": sorted(int(value) for value in rows),
        "column_set": sorted(int(value) for value in columns),
        "support_cells": full_support_cells,
        "support_pattern": support.astype(int).tolist(),
        "support_pattern_fingerprint": stable_array_fingerprint(support.astype(float)),
        "measurement_type_histogram": measurement_histogram,
        "measurement_type_proportions": {
            key: float(value / len(measurement_metadata))
            for key, value in measurement_histogram.items()
        },
        "state_type_histogram": state_histogram,
        "state_type_proportions": {
            key: float(value / len(state_metadata)) for key, value in state_histogram.items()
        },
        "row_degree_profile": row_degrees.tolist(),
        "column_degree_profile": column_degrees.tolist(),
        "row_degree_histogram": dict(sorted(Counter(row_degrees.tolist()).items())),
        "column_degree_histogram": dict(sorted(Counter(column_degrees.tolist()).items())),
        "nonzeros": int(np.count_nonzero(values)),
        "rank": int(np.linalg.matrix_rank(values)),
        "condition_number": float(np.linalg.cond(values)),
        "spectral_norm": float(singular_values[0]),
        "frobenius_norm": float(np.linalg.norm(values)),
        "max_absolute_entry": float(np.max(np.abs(values))),
        "singular_values": singular_values.tolist(),
        "descriptor_uses_selector_or_output_results": False,
    }


def _set_jaccard_distance(first: Iterable[Any], second: Iterable[Any]) -> float:
    left = {tuple(value) if isinstance(value, list) else value for value in first}
    right = {tuple(value) if isinstance(value, list) else value for value in second}
    union = left | right
    return 0.0 if not union else float(1.0 - len(left & right) / len(union))


def _total_variation_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    keys = set(first) | set(second)
    return float(
        0.5 * sum(abs(float(first.get(key, 0.0)) - float(second.get(key, 0.0))) for key in keys)
    )


def composite_structural_distance(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[str, float]:
    components = {
        "row_distance": _set_jaccard_distance(first["row_set"], second["row_set"]),
        "column_distance": _set_jaccard_distance(first["column_set"], second["column_set"]),
        "support_distance": _set_jaccard_distance(first["support_cells"], second["support_cells"]),
        "measurement_distance": _total_variation_distance(
            first["measurement_type_proportions"],
            second["measurement_type_proportions"],
        ),
        "state_distance": _total_variation_distance(
            first["state_type_proportions"], second["state_type_proportions"]
        ),
    }
    composite = (
        float(weights["rows"]) * components["row_distance"]
        + float(weights["columns"]) * components["column_distance"]
        + float(weights["support"]) * components["support_distance"]
        + float(weights["measurement"]) * components["measurement_distance"]
        + float(weights["state"]) * components["state_distance"]
    )
    return {**components, "composite_distance": float(composite)}


def select_structurally_diverse_candidates(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    previous_descriptor: Mapping[str, Any],
    weights: Mapping[str, float],
    preferred_count: int,
    minimum_distance: float,
) -> list[dict[str, Any]]:
    """Deterministic farthest-point selection with lexicographic tie-breaking."""

    remaining = {str(item["candidate_id"]): dict(item) for item in descriptors}
    if not remaining:
        return []
    selected: list[dict[str, Any]] = []
    previous_distances = {
        candidate_id: composite_structural_distance(item, previous_descriptor, weights)[
            "composite_distance"
        ]
        for candidate_id, item in remaining.items()
    }
    seed_id = sorted(
        remaining,
        key=lambda candidate_id: (-previous_distances[candidate_id], candidate_id),
    )[0]
    seed = remaining.pop(seed_id)
    selected.append(
        {
            **seed,
            "selection_order": 1,
            "distance_from_previous_block": previous_distances[seed_id],
            "minimum_distance_to_earlier_selected": previous_distances[seed_id],
            "selection_rule": "maximum_distance_from_previous_evaluation_block",
        }
    )
    while remaining and len(selected) < int(preferred_count):
        scores: dict[str, float] = {}
        for candidate_id, item in remaining.items():
            scores[candidate_id] = min(
                composite_structural_distance(item, chosen, weights)["composite_distance"]
                for chosen in selected
            )
        chosen_id = sorted(
            remaining,
            key=lambda candidate_id: (-scores[candidate_id], candidate_id),
        )[0]
        if scores[chosen_id] < float(minimum_distance):
            break
        chosen = remaining.pop(chosen_id)
        selected.append(
            {
                **chosen,
                "selection_order": len(selected) + 1,
                "distance_from_previous_block": previous_distances[chosen_id],
                "minimum_distance_to_earlier_selected": scores[chosen_id],
                "selection_rule": "farthest_point_maximin",
            }
        )
    return selected


def _load_previous_structures(context: StructuralContext) -> dict[str, dict[str, Any]]:
    registry = pd.read_csv(context.root / PRIOR_BENCHMARK_DIR / "instance_registry.csv")
    result: dict[str, dict[str, Any]] = {}
    for case_name in context.config["required_cases"]:
        selected = registry[registry["ieee_case"] == case_name].sort_values(
            "instance_id", kind="stable"
        )
        if selected.empty:
            raise FileNotFoundError(f"previous structural block is missing for {case_name}")
        row = selected.iloc[0]
        payload = json.loads(
            (
                context.root / PRIOR_BENCHMARK_DIR / "instances" / f"{row['instance_id']}.json"
            ).read_text(encoding="utf-8")
        )
        result[case_name] = build_structural_descriptor(
            candidate_id=PREVIOUS_BLOCK_LABEL,
            case_name=case_name,
            rows=payload["selected_rows"],
            columns=payload["selected_columns"],
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
            measurement_metadata=payload["measurement_metadata"],
            state_metadata=payload["state_metadata"],
        )
    return result


def stage_audit(context: StructuralContext) -> dict[str, Any]:
    audit_path = context.output_dir / "implementation_audit.md"
    if not audit_path.is_file():
        raise FileNotFoundError("implementation_audit.md must exist before evaluation")
    snapshot = _protected_file_snapshot(context.root, context.config["protected_paths"])
    atomic_write_json(
        context.output_dir / "protected_path_snapshot.json",
        {
            "created_before_benchmark_evaluation": True,
            "root": str(context.root),
            "protected_paths": context.config["protected_paths"],
            "file_count": len(snapshot),
            "files": snapshot,
        },
    )
    return {
        "audit_exists": True,
        "protected_files_snapshotted": len(snapshot),
        "working_tree_status_lines": len(_git_status(context.root).splitlines()),
    }


def stage_freeze(context: StructuralContext) -> dict[str, Any]:
    previous_frozen = _load_prior_frozen_configuration(context.root)
    previous_study = _load_prior_study_configuration(context.root)
    frozen = build_frozen_method_configuration(context.config, previous_frozen, previous_study)
    atomic_write_json(context.output_dir / "frozen_method_configuration.json", frozen)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=context.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    study = {
        **context.config,
        "root": str(context.root),
        "branch": branch,
        "git_commit": git_commit_hash(),
        "max_workers_recorded": context.max_workers,
        "benchmark_configuration_fingerprint": configuration_fingerprint(context.config),
        "frozen_method_configuration_fingerprint": frozen["configuration_fingerprint"],
        "benchmark_evaluation_started": False,
    }
    atomic_write_json(context.output_dir / "study_configuration.json", study)
    return {
        "configuration_fingerprint": frozen["configuration_fingerprint"],
        "previous_configuration_fingerprint": frozen["previous_configuration_fingerprint"],
    }


def _candidate_feasibility(
    matrix: np.ndarray, context: StructuralContext, comparison: Mapping[str, Any]
) -> SupportSelectionResult:
    settings = context.config["milp_solver_options"]
    return select_resource_constrained_support(
        matrix,
        np.ones_like(matrix, dtype=np.float64),
        SupportConstraints(int(comparison["k_budget"]), int(comparison["slot_budget"]), True),
        time_limit_seconds=float(settings["time_limit_seconds"]),
        relative_mip_gap=float(settings["relative_mip_gap"]),
        tie_epsilon_relative=float(settings["deterministic_tie_epsilon_relative"]),
    )


def stage_candidates(context: StructuralContext) -> dict[str, Any]:
    pool = context.config["candidate_pool"]
    previous_registry = pd.read_csv(context.root / PRIOR_BENCHMARK_DIR / "instance_registry.csv")
    previous_fingerprints = set(previous_registry["matrix_fingerprint"].astype(str))
    previous_structures = {
        case_name: {
            (
                tuple(json.loads(row.selected_rows)),
                tuple(json.loads(row.selected_columns)),
            )
            for row in previous_registry[previous_registry["ieee_case"] == case_name].itertuples(
                index=False
            )
        }
        for case_name in context.config["required_cases"]
    }
    registry_rows: list[dict[str, Any]] = []
    seen_structures: dict[str, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {
        case_name: set() for case_name in context.config["required_cases"]
    }
    for case_name in context.config["required_cases"]:
        matrix_seed = int(pool["candidate_linearization_seeds"][case_name])
        full = extract_weighted_jacobian_matrix(
            case_name=case_name,
            mode="ac_weighted_jacobian",
            case_source="pypower",
            measurement_profile="default",
            normalize=False,
            seed=matrix_seed,
        )
        for policy in pool["policies"]:
            for variant in range(int(pool["variants_per_policy"])):
                candidate_id = f"{case_name}_{policy}_v{variant:02d}"
                common: dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "ieee_case": case_name,
                    "policy": policy,
                    "variant": variant,
                    "candidate_linearization_seed": matrix_seed,
                    "outcome_independent": True,
                    "selector_outcomes_used_for_inclusion": False,
                    "status": "excluded",
                    "failure_reason": "",
                }
                try:
                    matrix, rows, columns = generate_structural_candidate(
                        full.matrix,
                        full.metadata,
                        policy=str(policy),
                        variant=variant,
                        case_name=case_name,
                        config=context.config,
                    )
                    fingerprint = stable_array_fingerprint(matrix)
                    structure = (tuple(rows.tolist()), tuple(columns.tolist()))
                    reasons: list[str] = []
                    if structure in previous_structures[case_name]:
                        reasons.append("previous_block_row_column_identity")
                    if (
                        fingerprint in previous_fingerprints
                        or fingerprint == context.config["development_matrix_fingerprint"]
                    ):
                        reasons.append("previous_block_matrix_fingerprint")
                    if structure in seen_structures[case_name]:
                        reasons.append("duplicate_candidate_block")
                    seen_structures[case_name].add(structure)
                    if np.any(~np.any(matrix != 0.0, axis=1)) or np.any(
                        ~np.any(matrix != 0.0, axis=0)
                    ):
                        reasons.append("inactive_row_or_column")
                    if int(np.count_nonzero(matrix)) < int(pool["minimum_candidate_nonzeros"]):
                        reasons.append("insufficient_nonzeros")
                    primary = _candidate_feasibility(
                        matrix, context, context.config["primary_comparison"]
                    )
                    secondary = _candidate_feasibility(
                        matrix, context, context.config["secondary_comparison"]
                    )
                    if pool["require_primary_and_secondary_budget_feasibility"]:
                        if primary.status != "completed":
                            reasons.append("support_budget_infeasible:primary")
                        if secondary.status != "completed":
                            reasons.append("support_budget_infeasible:secondary")
                    regularization = context.config["regularization"]
                    reference_mu = float(np.max(np.abs(matrix)))
                    reference_beta = int(regularization["reference_slot_count"]) * reference_mu
                    alpha = float(regularization["lambda_ref"]) * reference_beta**2
                    try:
                        ridge_svd_solution(matrix, np.ones(8), alpha=alpha)
                    except Exception as exc:
                        reasons.append(
                            f"invalid_regularization_or_ridge_solve:{type(exc).__name__}"
                        )
                    state_metadata = _state_records(full.metadata, columns)
                    measurement_metadata = _selected_measurement_records(full.metadata, rows)
                    payload = {
                        **common,
                        "matrix": matrix,
                        "matrix_shape": list(matrix.shape),
                        "matrix_fingerprint": fingerprint,
                        "selected_rows": rows,
                        "selected_columns": columns,
                        "state_metadata": state_metadata,
                        "measurement_metadata": measurement_metadata,
                        "candidate_nonzeros": int(np.count_nonzero(matrix)),
                        "reference_mu": reference_mu,
                        "reference_beta": reference_beta,
                        "regularization_alpha": alpha,
                        "primary_structural_feasibility": primary.status,
                        "secondary_structural_feasibility": secondary.status,
                        "status": "included" if not reasons else "excluded",
                        "failure_reason": ";".join(reasons),
                    }
                    atomic_write_json(
                        context.output_dir / "candidates" / case_name / f"{candidate_id}.json",
                        payload,
                    )
                    registry_rows.append(
                        {
                            **common,
                            "selected_rows": _compact_json(rows),
                            "selected_columns": _compact_json(columns),
                            "matrix_fingerprint": fingerprint,
                            "candidate_nonzeros": int(np.count_nonzero(matrix)),
                            "measurement_type_composition": _compact_json(
                                Counter(item["measurement_type"] for item in measurement_metadata)
                            ),
                            "state_type_composition": _compact_json(
                                Counter(item["state_type"] for item in state_metadata)
                            ),
                            "primary_structural_feasibility": primary.status,
                            "secondary_structural_feasibility": secondary.status,
                            "status": "included" if not reasons else "excluded",
                            "failure_reason": ";".join(reasons),
                            "candidate_file": (
                                Path("candidates") / case_name / f"{candidate_id}.json"
                            ).as_posix(),
                        }
                    )
                except Exception as exc:
                    registry_rows.append(
                        {
                            **common,
                            "failure_reason": (
                                f"candidate_generation_failure:{type(exc).__name__}:{exc}"
                            ),
                            "candidate_file": "",
                        }
                    )
    registry = pd.DataFrame(registry_rows).sort_values(
        ["ieee_case", "policy", "variant"], kind="stable"
    )
    atomic_write_csv(context.output_dir / "candidate_registry.csv", registry)
    exclusions = registry[registry["status"] != "included"].copy()
    atomic_write_csv(context.output_dir / "candidate_exclusion_registry.csv", exclusions)
    expected = (
        len(context.config["required_cases"])
        * len(pool["policies"])
        * int(pool["variants_per_policy"])
    )
    if len(registry) != expected:
        raise RuntimeError(f"candidate registry contains {len(registry)} of {expected} rows")
    return {
        "candidate_rows": len(registry),
        "included": int((registry["status"] == "included").sum()),
        "excluded": len(exclusions),
        "cases": sorted(registry["ieee_case"].unique().tolist()),
    }


def _load_candidate_payload(context: StructuralContext, candidate_id: str) -> dict[str, Any]:
    registry = pd.read_csv(context.output_dir / "candidate_registry.csv")
    matched = registry[registry["candidate_id"] == candidate_id]
    if len(matched) != 1 or not str(matched.iloc[0]["candidate_file"]):
        raise FileNotFoundError(f"candidate payload unavailable for {candidate_id}")
    return json.loads(
        (context.output_dir / str(matched.iloc[0]["candidate_file"])).read_text(encoding="utf-8")
    )


def _descriptor_csv_record(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    json_columns = {
        "selected_rows",
        "selected_columns",
        "row_set",
        "column_set",
        "support_cells",
        "support_pattern",
        "measurement_type_histogram",
        "measurement_type_proportions",
        "state_type_histogram",
        "state_type_proportions",
        "row_degree_profile",
        "column_degree_profile",
        "row_degree_histogram",
        "column_degree_histogram",
        "singular_values",
    }
    return {
        key: _compact_json(value) if key in json_columns else value
        for key, value in descriptor.items()
    }


def stage_descriptors(context: StructuralContext) -> dict[str, Any]:
    registry = pd.read_csv(context.output_dir / "candidate_registry.csv")
    included = registry[registry["status"] == "included"]
    descriptors: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for row in included.itertuples(index=False):
        payload = _load_candidate_payload(context, str(row.candidate_id))
        descriptor = build_structural_descriptor(
            candidate_id=str(row.candidate_id),
            case_name=str(row.ieee_case),
            rows=payload["selected_rows"],
            columns=payload["selected_columns"],
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
            measurement_metadata=payload["measurement_metadata"],
            state_metadata=payload["state_metadata"],
        )
        descriptors[str(row.candidate_id)] = descriptor
        records.append(
            {
                **_descriptor_csv_record(descriptor),
                "policy": row.policy,
                "variant": int(row.variant),
                "descriptor_data_scope": "matrix_structure_and_metadata_only",
            }
        )
    previous = _load_previous_structures(context)
    weights = context.config["structural_selection"]["distance_weights"]
    pair_rows: list[dict[str, Any]] = []
    previous_rows: list[dict[str, Any]] = []
    for case_name in context.config["required_cases"]:
        local = sorted(
            [item for item in descriptors.values() if item["ieee_case"] == case_name],
            key=lambda item: str(item["candidate_id"]),
        )
        for index, first in enumerate(local):
            distance_previous = composite_structural_distance(first, previous[case_name], weights)
            previous_rows.append(
                {
                    "ieee_case": case_name,
                    "candidate_id": first["candidate_id"],
                    "reference_id": PREVIOUS_BLOCK_LABEL,
                    **distance_previous,
                    "outcome_data_used": False,
                }
            )
            for second in local[index + 1 :]:
                pair_rows.append(
                    {
                        "ieee_case": case_name,
                        "candidate_id_a": first["candidate_id"],
                        "candidate_id_b": second["candidate_id"],
                        **composite_structural_distance(first, second, weights),
                        "outcome_data_used": False,
                    }
                )
    atomic_write_csv(context.output_dir / "structural_descriptors.csv", pd.DataFrame(records))
    atomic_write_csv(
        context.output_dir / "candidate_pairwise_distances.csv", pd.DataFrame(pair_rows)
    )
    atomic_write_csv(
        context.output_dir / "candidate_distance_from_previous.csv",
        pd.DataFrame(previous_rows),
    )
    atomic_write_json(context.output_dir / "previous_block_descriptors.json", previous)
    return {
        "descriptor_rows": len(records),
        "pairwise_distance_rows": len(pair_rows),
        "previous_distance_rows": len(previous_rows),
        "selector_or_output_data_used": False,
    }


def _load_descriptor_frame(context: StructuralContext) -> list[dict[str, Any]]:
    frame = pd.read_csv(context.output_dir / "structural_descriptors.csv")
    json_columns = (
        "selected_rows",
        "selected_columns",
        "row_set",
        "column_set",
        "support_cells",
        "support_pattern",
        "measurement_type_histogram",
        "measurement_type_proportions",
        "state_type_histogram",
        "state_type_proportions",
        "row_degree_profile",
        "column_degree_profile",
        "row_degree_histogram",
        "column_degree_histogram",
        "singular_values",
    )
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        for column in json_columns:
            row[column] = json.loads(row[column])
        records.append(row)
    return records


def stage_structural_selection(context: StructuralContext) -> dict[str, Any]:
    descriptors = _load_descriptor_frame(context)
    previous = _load_previous_structures(context)
    settings = context.config["structural_selection"]
    weights = settings["distance_weights"]
    selected_records: list[dict[str, Any]] = []
    for case_name in context.config["required_cases"]:
        local = [item for item in descriptors if item["ieee_case"] == case_name]
        selected = select_structurally_diverse_candidates(
            local,
            previous_descriptor=previous[case_name],
            weights=weights,
            preferred_count=int(settings["preferred_groups_per_case"]),
            minimum_distance=float(settings["minimum_pairwise_distance"]),
        )
        if len(selected) < int(settings["minimum_groups_per_case"]):
            raise RuntimeError(
                f"insufficient_structural_diversity: {case_name} selected {len(selected)} groups"
            )
        for chosen in selected:
            order = int(chosen["selection_order"])
            group_id = deterministic_structural_group_id(case_name, order)
            payload = _load_candidate_payload(context, str(chosen["candidate_id"]))
            record = {
                "structural_group_id": group_id,
                "ieee_case": case_name,
                "selection_order": order,
                "candidate_id": chosen["candidate_id"],
                "candidate_policy": chosen["policy"],
                "candidate_variant": int(chosen["variant"]),
                "selected_rows": _compact_json(chosen["selected_rows"]),
                "selected_columns": _compact_json(chosen["selected_columns"]),
                "support_pattern_fingerprint": chosen["support_pattern_fingerprint"],
                "measurement_type_composition": _compact_json(chosen["measurement_type_histogram"]),
                "state_type_composition": _compact_json(chosen["state_type_histogram"]),
                "nonzeros": int(chosen["nonzeros"]),
                "distance_from_previous_block": float(chosen["distance_from_previous_block"]),
                "minimum_distance_to_earlier_selected": float(
                    chosen["minimum_distance_to_earlier_selected"]
                ),
                "selection_rule": chosen["selection_rule"],
                "minimum_distance_threshold": float(settings["minimum_pairwise_distance"]),
                "selector_outcomes_used_for_selection": False,
                "status": "included",
                "failure_reason": "",
            }
            selected_records.append(record)
            atomic_write_json(
                context.output_dir / "groups" / group_id / "structural_definition.json",
                _json_ready(
                    {
                        **record,
                        "selected_rows": payload["selected_rows"],
                        "selected_columns": payload["selected_columns"],
                        "reference_matrix": payload["matrix"],
                        "reference_matrix_fingerprint": payload["matrix_fingerprint"],
                        "measurement_metadata": payload["measurement_metadata"],
                        "state_metadata": payload["state_metadata"],
                        "descriptor": chosen,
                        "selection_completed_before_selector_evaluation": True,
                    }
                ),
            )
    selected_frame = pd.DataFrame(selected_records).sort_values(
        ["ieee_case", "selection_order"], kind="stable"
    )
    atomic_write_csv(context.output_dir / "structural_group_registry.csv", selected_frame)
    atomic_write_csv(context.output_dir / "selected_structural_groups.csv", selected_frame.copy())
    study_path = context.output_dir / "study_configuration.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["benchmark_evaluation_started"] = True
    study["selected_structural_group_ids"] = selected_frame["structural_group_id"].tolist()
    study["structural_group_registry_fingerprint"] = configuration_fingerprint(
        {"groups": selected_frame.to_dict(orient="records")}
    )
    atomic_write_json(study_path, study)
    counts = selected_frame.groupby("ieee_case").size().to_dict()
    return {
        "selected_groups": len(selected_frame),
        "groups_per_case": {key: int(value) for key, value in counts.items()},
        "minimum_selected_distance": float(
            selected_frame.loc[
                selected_frame["selection_order"] > 1,
                "minimum_distance_to_earlier_selected",
            ].min()
        ),
        "selector_outcomes_used": False,
    }


def stage_realizations(context: StructuralContext) -> dict[str, Any]:
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    regularization = context.config["regularization"]
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for group in groups.itertuples(index=False):
        group_id = str(group.structural_group_id)
        case_name = str(group.ieee_case)
        definition = json.loads(
            (context.output_dir / "groups" / group_id / "structural_definition.json").read_text(
                encoding="utf-8"
            )
        )
        selected_rows = np.asarray(definition["selected_rows"], dtype=np.int64)
        selected_columns = np.asarray(definition["selected_columns"], dtype=np.int64)
        seeds = [
            int(value) for value in context.config["realizations"]["linearization_seeds"][case_name]
        ]
        for realization_order, matrix_seed in enumerate(seeds, start=1):
            instance_id = deterministic_realization_id(group_id, realization_order, matrix_seed)
            try:
                full = extract_weighted_jacobian_matrix(
                    case_name=case_name,
                    mode="ac_weighted_jacobian",
                    case_source="pypower",
                    measurement_profile="default",
                    normalize=False,
                    seed=matrix_seed,
                )
                matrix = np.asarray(
                    full.matrix[np.ix_(selected_rows, selected_columns)],
                    dtype=np.float64,
                )
                reasons: list[str] = []
                if matrix.shape != (8, 8) or not np.all(np.isfinite(matrix)):
                    reasons.append("matrix_realization_failure:invalid_shape_or_values")
                if np.any(~np.any(matrix != 0.0, axis=1)) or np.any(~np.any(matrix != 0.0, axis=0)):
                    reasons.append("matrix_realization_failure:inactive_row_or_column")
                if int(np.count_nonzero(matrix)) < int(
                    context.config["candidate_pool"]["minimum_candidate_nonzeros"]
                ):
                    reasons.append("insufficient_nonzeros")
                primary = _candidate_feasibility(
                    matrix, context, context.config["primary_comparison"]
                )
                secondary = _candidate_feasibility(
                    matrix, context, context.config["secondary_comparison"]
                )
                if primary.status != "completed":
                    reasons.append("support_budget_infeasible:primary")
                if secondary.status != "completed":
                    reasons.append("support_budget_infeasible:secondary")
                reference_mu = float(np.max(np.abs(matrix)))
                reference_beta = int(regularization["reference_slot_count"]) * reference_mu
                alpha = float(regularization["lambda_ref"]) * reference_beta**2
                ridge_svd_solution(matrix, np.ones(8), alpha=alpha)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                state_metadata = _state_records(full.metadata, selected_columns)
                measurement_metadata = _selected_measurement_records(full.metadata, selected_rows)
                fingerprint = stable_array_fingerprint(matrix)
                payload = {
                    "instance_id": instance_id,
                    "structural_group_id": group_id,
                    "structural_group_selection_order": int(group.selection_order),
                    "candidate_id": group.candidate_id,
                    "candidate_policy": group.candidate_policy,
                    "ieee_case": case_name,
                    "realization_order": realization_order,
                    "matrix_selection_seed": matrix_seed,
                    "matrix": matrix,
                    "matrix_shape": list(matrix.shape),
                    "candidate_nonzeros": int(np.count_nonzero(matrix)),
                    "matrix_fingerprint": fingerprint,
                    "selected_rows": selected_rows,
                    "selected_columns": selected_columns,
                    "state_metadata": state_metadata,
                    "measurement_metadata": measurement_metadata,
                    "condition_number": float(np.linalg.cond(matrix)),
                    "spectral_norm": float(singular_values[0]),
                    "regularization_alpha": alpha,
                    "regularization_policy": regularization["policy"],
                    "reference_mu": reference_mu,
                    "reference_beta": reference_beta,
                    "lambda_ref": regularization["lambda_ref"],
                    "development_or_evaluation": "structural_evaluation",
                    "inclusion_status": "included" if not reasons else "excluded",
                    "inclusion_reasons": reasons,
                    "primary_structural_feasibility": primary.status,
                    "secondary_structural_feasibility": secondary.status,
                    "extraction_policy": {
                        "strategy": "fixed_preselected_structural_group_rows_columns",
                        "rows_columns_fixed_within_group": True,
                        "outcome_independent": True,
                        "candidate_id": group.candidate_id,
                    },
                    "selector_outcomes_used_for_inclusion": False,
                }
                atomic_write_json(
                    context.output_dir / "instances" / f"{instance_id}.json",
                    _json_ready(payload),
                )
                atomic_write_json(
                    context.output_dir
                    / "groups"
                    / group_id
                    / "realizations"
                    / f"{instance_id}.json",
                    {
                        "instance_id": instance_id,
                        "matrix_seed": matrix_seed,
                        "matrix_fingerprint": fingerprint,
                        "instance_file": (Path("instances") / f"{instance_id}.json").as_posix(),
                        "status": payload["inclusion_status"],
                        "failure_reasons": reasons,
                    },
                )
                record = {
                    key: _json_ready(payload[key])
                    for key in (
                        "instance_id",
                        "structural_group_id",
                        "structural_group_selection_order",
                        "candidate_id",
                        "candidate_policy",
                        "ieee_case",
                        "realization_order",
                        "matrix_selection_seed",
                        "matrix_shape",
                        "candidate_nonzeros",
                        "matrix_fingerprint",
                        "selected_rows",
                        "selected_columns",
                        "state_metadata",
                        "measurement_metadata",
                        "condition_number",
                        "spectral_norm",
                        "regularization_alpha",
                        "regularization_policy",
                        "reference_mu",
                        "reference_beta",
                        "lambda_ref",
                        "development_or_evaluation",
                        "inclusion_status",
                        "primary_structural_feasibility",
                        "secondary_structural_feasibility",
                        "selector_outcomes_used_for_inclusion",
                    )
                }
                for column in (
                    "matrix_shape",
                    "selected_rows",
                    "selected_columns",
                    "state_metadata",
                    "measurement_metadata",
                ):
                    record[column] = _compact_json(record[column])
                if reasons:
                    exclusions.append(
                        {
                            "instance_id": instance_id,
                            "structural_group_id": group_id,
                            "ieee_case": case_name,
                            "matrix_selection_seed": matrix_seed,
                            "stage": "realizations",
                            "status": "excluded",
                            "failure_reason": ";".join(reasons),
                            "matrix_fingerprint": fingerprint,
                        }
                    )
                else:
                    rows.append(record)
            except Exception as exc:
                exclusions.append(
                    {
                        "instance_id": instance_id,
                        "structural_group_id": group_id,
                        "ieee_case": case_name,
                        "matrix_selection_seed": matrix_seed,
                        "stage": "realizations",
                        "status": "excluded",
                        "failure_reason": (
                            f"matrix_realization_failure:{type(exc).__name__}:{exc}"
                        ),
                        "matrix_fingerprint": "",
                    }
                )
    registry = pd.DataFrame(rows).sort_values(
        ["ieee_case", "structural_group_selection_order", "realization_order"],
        kind="stable",
    )
    exclusion_columns = [
        "instance_id",
        "structural_group_id",
        "ieee_case",
        "matrix_selection_seed",
        "stage",
        "status",
        "failure_reason",
        "matrix_fingerprint",
    ]
    exclusion_frame = pd.DataFrame(exclusions, columns=exclusion_columns)
    atomic_write_csv(context.output_dir / "instance_registry.csv", registry)
    atomic_write_csv(context.output_dir / "instance_exclusion_registry.csv", exclusion_frame)
    expected_per_group = int(context.config["realizations"]["per_group"])
    counts = registry.groupby("structural_group_id").size()
    if len(counts) != len(groups) or not bool((counts == expected_per_group).all()):
        raise RuntimeError(
            "matrix_realization_failure: every structural group must retain two realizations"
        )
    study_path = context.output_dir / "study_configuration.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["instance_registry_fingerprint"] = configuration_fingerprint(
        {"instances": registry.to_dict(orient="records")}
    )
    study["realization_count"] = len(registry)
    atomic_write_json(study_path, study)
    return {
        "realizations": len(registry),
        "structural_groups": int(registry["structural_group_id"].nunique()),
        "realizations_per_group": expected_per_group,
        "excluded": len(exclusion_frame),
    }


def _augment_instance_metadata(
    context: StructuralContext, relative_path: str, *, group_summary: bool = False
) -> pd.DataFrame:
    path = context.output_dir / relative_path
    frame = pd.read_csv(path)
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    columns = [
        "instance_id",
        "structural_group_id",
        "structural_group_selection_order",
        "realization_order",
        "matrix_selection_seed",
        "candidate_id",
        "candidate_policy",
    ]
    additions = [column for column in columns if column not in frame.columns]
    if additions and "instance_id" in frame.columns:
        frame = frame.merge(instances[["instance_id", *additions]], on="instance_id", how="left")
    if group_summary and "structural_group_id" not in frame.columns:
        raise RuntimeError(f"unable to attach structural group to {relative_path}")
    atomic_write_csv(path, frame)
    return frame


def stage_functionals(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_functionals(context)
    frame = _augment_instance_metadata(context, "functional_registry.csv")
    if not bool(frame["selection_data_used"].eq("state_metadata_only_no_output_accuracy").all()):
        raise RuntimeError("functional_metadata_failure: outcome data entered functional design")
    result["structural_groups"] = int(frame["structural_group_id"].nunique())
    return result


def stage_residuals(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_residuals(context)
    frame = _augment_instance_metadata(context, "residual_registry.csv")
    overlap = 0
    for _instance_id, group in frame.groupby("instance_id", sort=True):
        training = set(group.loc[group["split"] == "training", "residual_seed"])
        held_out = set(group.loc[group["split"] == "held_out", "residual_seed"])
        overlap += len(training & held_out)
    if overlap:
        raise RuntimeError(f"data isolation failure: {overlap} residual-seed overlaps")
    result["training_heldout_seed_overlap"] = overlap
    return result


def stage_supports(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_supports(context)
    for path in (
        "support_registry.csv",
        "support_selection_results.csv",
        "training_instance_summary.csv",
        "refinement_traces.csv",
        "entry_scores.csv",
    ):
        frame = pd.read_csv(context.output_dir / path)
        if not frame.empty and "instance_id" in frame.columns:
            _augment_instance_metadata(context, path)
    support = pd.read_csv(context.output_dir / "support_registry.csv")
    leakage = support["selection_data_split"].astype(str).str.contains("held", case=False, na=False)
    if bool(leakage.any()):
        raise RuntimeError("data isolation failure: held-out data entered support selection")
    result["structural_groups"] = int(support["structural_group_id"].nunique())
    return result


def stage_heldout(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_heldout(context)
    for path in (
        "heldout_results.csv",
        "heldout_instance_summary.csv",
        "heldout_case_summary.csv",
        "heldout_functional_summary.csv",
    ):
        frame = pd.read_csv(context.output_dir / path)
        if not frame.empty and "instance_id" in frame.columns:
            _augment_instance_metadata(context, path)
    heldout = pd.read_csv(context.output_dir / "heldout_results.csv")
    if set(heldout["split"].dropna().astype(str)) != {"held_out"}:
        raise RuntimeError("held-out registry contains a non-held-out result")
    result["structural_groups"] = int(heldout["structural_group_id"].nunique())
    return result


def _group_comparison_rows(
    context: StructuralContext,
    comparison: Mapping[str, Any],
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_summary = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    functional_summary = pd.read_csv(context.output_dir / "heldout_functional_summary.csv")
    candidate = str(comparison["candidate_selector"])
    baseline = str(comparison["baseline_selector"])
    k_budget = int(comparison["k_budget"])
    slot_budget = int(comparison["slot_budget"])
    selected = support_summary[
        (support_summary["k_budget"] == k_budget)
        & (support_summary["slot_budget"] == slot_budget)
        & support_summary["selector"].isin([candidate, baseline])
        & (support_summary["status"] == "completed")
    ].copy()
    realization = selected.pivot(
        index=[
            "instance_id",
            "structural_group_id",
            "ieee_case",
            "realization_order",
        ],
        columns="selector",
        values="median_normalized_error",
    ).reset_index()
    realization = realization.dropna(subset=[candidate, baseline]).rename(
        columns={candidate: "candidate_error", baseline: "baseline_error"}
    )
    realization["paired_difference_candidate_minus_baseline"] = (
        realization["candidate_error"] - realization["baseline_error"]
    )
    grouped = (
        selected.groupby(["structural_group_id", "ieee_case", "selector"], sort=True)
        .agg(
            group_error=("median_normalized_error", "mean"),
            realization_count=("instance_id", "nunique"),
            minimum_realization_error=("median_normalized_error", "min"),
            maximum_realization_error=("median_normalized_error", "max"),
        )
        .reset_index()
    )
    expected_realizations = int(context.config["realizations"]["per_group"])
    if not bool((grouped["realization_count"] == expected_realizations).all()):
        raise RuntimeError("primary comparison lacks the frozen realizations per group")
    pairs = grouped.pivot(
        index=["structural_group_id", "ieee_case"],
        columns="selector",
        values="group_error",
    ).reset_index()
    pairs = pairs.dropna(subset=[candidate, baseline]).rename(
        columns={
            candidate: "candidate_group_normalized_error",
            baseline: "baseline_group_normalized_error",
        }
    )
    pairs["paired_difference_candidate_minus_baseline"] = (
        pairs["candidate_group_normalized_error"] - pairs["baseline_group_normalized_error"]
    )
    tolerance = float(
        comparison.get(
            "tie_relative_tolerance",
            context.config["primary_comparison"]["tie_relative_tolerance"],
        )
    )
    epsilon = float(
        comparison.get("tie_epsilon", context.config["primary_comparison"]["tie_epsilon"])
    )
    pairs["outcome"] = [
        classify_matched_errors(
            candidate_error,
            baseline_error,
            relative_tolerance=tolerance,
            epsilon=epsilon,
        )
        for candidate_error, baseline_error in zip(
            pairs["candidate_group_normalized_error"],
            pairs["baseline_group_normalized_error"],
            strict=True,
        )
    ]
    pairs["comparison_label"] = label
    pairs["candidate_selector"] = candidate
    pairs["baseline_selector"] = baseline
    pairs["k_budget"] = k_budget
    pairs["slot_budget"] = slot_budget
    pairs["realization_combination"] = "mean_of_realization_medians"

    functional_selected = functional_summary[
        (functional_summary["k_budget"] == k_budget)
        & (functional_summary["slot_budget"] == slot_budget)
        & functional_summary["selector"].isin([candidate, baseline])
    ].copy()
    functional_grouped = (
        functional_selected.groupby(
            [
                "structural_group_id",
                "ieee_case",
                "functional_id",
                "selector",
            ],
            sort=True,
        )
        .agg(
            group_error=("median_normalized_error", "mean"),
            realization_count=("instance_id", "nunique"),
        )
        .reset_index()
    )
    functional_pairs = functional_grouped.pivot(
        index=["structural_group_id", "ieee_case", "functional_id"],
        columns="selector",
        values="group_error",
    ).reset_index()
    functional_pairs = functional_pairs.dropna(subset=[candidate, baseline]).rename(
        columns={
            candidate: "candidate_group_normalized_error",
            baseline: "baseline_group_normalized_error",
        }
    )
    functional_pairs["paired_difference_candidate_minus_baseline"] = (
        functional_pairs["candidate_group_normalized_error"]
        - functional_pairs["baseline_group_normalized_error"]
    )
    functional_pairs["outcome"] = [
        classify_matched_errors(
            candidate_error,
            baseline_error,
            relative_tolerance=tolerance,
            epsilon=epsilon,
        )
        for candidate_error, baseline_error in zip(
            functional_pairs["candidate_group_normalized_error"],
            functional_pairs["baseline_group_normalized_error"],
            strict=True,
        )
    ]
    functional_pairs["comparison_label"] = label
    return pairs, functional_pairs, realization


def _outcome_counts(frame: pd.DataFrame, group_column: str | None = None) -> dict[str, Any]:
    def count(local: pd.DataFrame) -> dict[str, int]:
        observed = local["outcome"].value_counts()
        return {outcome: int(observed.get(outcome, 0)) for outcome in ("win", "tie", "loss")}

    if group_column is None:
        return count(frame)
    return {str(key): count(group) for key, group in frame.groupby(group_column, sort=True)}


def structural_group_bootstrap(
    pairs: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    case_stratified: bool,
) -> pd.DataFrame:
    required = {
        "structural_group_id",
        "ieee_case",
        "paired_difference_candidate_minus_baseline",
    }
    if missing := required.difference(pairs.columns):
        raise ValueError(f"structural bootstrap missing columns {sorted(missing)}")
    if pairs.empty or int(samples) <= 0:
        raise ValueError("structural bootstrap requires pairs and positive samples")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    values = pairs["paired_difference_candidate_minus_baseline"].to_numpy(dtype=np.float64)
    case_values = {
        str(case): group["paired_difference_candidate_minus_baseline"].to_numpy(dtype=np.float64)
        for case, group in pairs.groupby("ieee_case", sort=True)
    }
    for sample in range(int(samples)):
        if case_stratified:
            draws = [
                local[rng.integers(0, len(local), size=len(local))]
                for local in case_values.values()
            ]
            sampled = np.concatenate(draws)
        else:
            sampled = values[rng.integers(0, len(values), size=len(values))]
        rows.append(
            {
                "bootstrap_sample": sample,
                "structural_group_count": len(sampled),
                "median_paired_difference_sensitivity_minus_magnitude": float(np.median(sampled)),
                "mean_paired_difference_sensitivity_minus_magnitude": float(np.mean(sampled)),
                "resampling_unit": "structural_group",
                "bootstrap_mode": ("case_stratified" if case_stratified else "unstratified"),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def stage_primary_test(context: StructuralContext) -> dict[str, Any]:
    comparison = context.config["primary_comparison"]
    pairs, functional_pairs, realization_pairs = _group_comparison_rows(
        context, comparison, label="primary"
    )
    bootstrap = structural_group_bootstrap(
        pairs,
        samples=int(comparison["bootstrap_samples"]),
        seed=int(comparison["bootstrap_seed"]),
        case_stratified=False,
    )
    stratified = structural_group_bootstrap(
        pairs,
        samples=int(comparison["bootstrap_samples"]),
        seed=int(comparison["case_stratified_bootstrap_seed"]),
        case_stratified=True,
    )
    column = "median_paired_difference_sensitivity_minus_magnitude"
    ci = np.quantile(bootstrap[column], [0.025, 0.975])
    stratified_ci = np.quantile(stratified[column], [0.025, 0.975])
    differences = pairs["paired_difference_candidate_minus_baseline"].to_numpy(dtype=np.float64)
    payload = {
        "comparison_declaration": comparison,
        "structural_groups": len(pairs),
        "numerical_realizations": len(realization_pairs),
        "overall_win_tie_loss": _outcome_counts(pairs),
        "case_win_tie_loss": _outcome_counts(pairs, "ieee_case"),
        "functional_win_tie_loss": _outcome_counts(functional_pairs, "functional_id"),
        "functional_case_win_tie_loss": {
            f"{case}:{functional}": _outcome_counts(group)
            for (case, functional), group in functional_pairs.groupby(
                ["ieee_case", "functional_id"], sort=True
            )
        },
        "median_paired_difference_sensitivity_minus_magnitude": float(np.median(differences)),
        "mean_paired_difference_sensitivity_minus_magnitude": float(np.mean(differences)),
        "bootstrap_confidence_interval_95": [float(ci[0]), float(ci[1])],
        "case_stratified_bootstrap_confidence_interval_95": [
            float(stratified_ci[0]),
            float(stratified_ci[1]),
        ],
        "bootstrap_probability_median_difference_below_zero": float(
            np.mean(bootstrap[column] < 0.0)
        ),
        "case_stratified_probability_median_difference_below_zero": float(
            np.mean(stratified[column] < 0.0)
        ),
        "bootstrap_resampling_unit": "structural_group",
        "numerical_realizations_treated_as_independent_structures": False,
        "development_or_previous_instances_in_score": False,
        "metric_substitution_after_results": False,
        "status": "completed",
    }
    atomic_write_json(context.output_dir / "structural_primary_test.json", payload)
    atomic_write_csv(context.output_dir / "structural_primary_matched_pairs.csv", pairs)
    atomic_write_csv(
        context.output_dir / "structural_primary_functional_pairs.csv",
        functional_pairs,
    )
    atomic_write_csv(
        context.output_dir / "structural_primary_realization_pairs.csv",
        realization_pairs,
    )
    atomic_write_csv(context.output_dir / "structural_group_bootstrap.csv", bootstrap)
    atomic_write_csv(context.output_dir / "structural_case_stratified_bootstrap.csv", stratified)
    return {
        "structural_groups": len(pairs),
        **_outcome_counts(pairs),
        "median_paired_difference": float(np.median(differences)),
        "bootstrap_ci_low": float(ci[0]),
        "bootstrap_ci_high": float(ci[1]),
        "bootstrap_probability_below_zero": payload[
            "bootstrap_probability_median_difference_below_zero"
        ],
    }


def stage_secondary_test(context: StructuralContext) -> dict[str, Any]:
    comparison = {
        **context.config["secondary_comparison"],
        "tie_relative_tolerance": context.config["primary_comparison"]["tie_relative_tolerance"],
        "tie_epsilon": context.config["primary_comparison"]["tie_epsilon"],
    }
    pairs, functional_pairs, realization_pairs = _group_comparison_rows(
        context, comparison, label="secondary"
    )
    differences = pairs["paired_difference_candidate_minus_baseline"].to_numpy(dtype=np.float64)
    payload = {
        "comparison_declaration": context.config["secondary_comparison"],
        "structural_groups": len(pairs),
        "numerical_realizations": len(realization_pairs),
        "overall_win_tie_loss": _outcome_counts(pairs),
        "case_win_tie_loss": _outcome_counts(pairs, "ieee_case"),
        "functional_win_tie_loss": _outcome_counts(functional_pairs, "functional_id"),
        "median_paired_difference_refined_minus_magnitude": float(np.median(differences)),
        "mean_paired_difference_refined_minus_magnitude": float(np.mean(differences)),
        "numerical_realizations_treated_as_independent_structures": False,
        "status": "completed",
    }
    atomic_write_json(context.output_dir / "structural_secondary_test.json", payload)
    atomic_write_csv(context.output_dir / "structural_secondary_matched_pairs.csv", pairs)
    atomic_write_csv(
        context.output_dir / "structural_secondary_functional_pairs.csv",
        functional_pairs,
    )
    return {
        "structural_groups": len(pairs),
        **_outcome_counts(pairs),
        "median_paired_difference": float(np.median(differences)),
    }


def stage_structural_analysis(context: StructuralContext) -> dict[str, Any]:
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    pairs = pd.read_csv(context.output_dir / "structural_primary_matched_pairs.csv")
    functional = pd.read_csv(context.output_dir / "structural_primary_functional_pairs.csv")
    merged = groups.merge(
        pairs,
        on=["structural_group_id", "ieee_case"],
        how="inner",
        validate="one_to_one",
    )
    merged["sensitivity_improvement_over_magnitude"] = -merged[
        "paired_difference_candidate_minus_baseline"
    ]
    atomic_write_csv(context.output_dir / "structural_performance_analysis.csv", merged)
    from scipy.stats import spearmanr

    association_rows: list[dict[str, Any]] = []
    for scope, local in [
        ("overall", merged),
        *[(str(case), frame) for case, frame in merged.groupby("ieee_case", sort=True)],
    ]:
        if len(local) < 2 or local["distance_from_previous_block"].nunique() < 2:
            statistic = np.nan
            pvalue = np.nan
        else:
            result = spearmanr(
                local["distance_from_previous_block"],
                local["sensitivity_improvement_over_magnitude"],
            )
            statistic = float(result.statistic)
            pvalue = float(result.pvalue)
        association_rows.append(
            {
                "scope": scope,
                "structural_groups": len(local),
                "spearman_distance_vs_sensitivity_improvement": statistic,
                "two_sided_pvalue_descriptive_only": pvalue,
                "association_semantics": (
                    "descriptive_group_level_association_not_causal_or_confirmatory"
                ),
            }
        )
    association = pd.DataFrame(association_rows)
    atomic_write_csv(context.output_dir / "structural_distance_association.csv", association)
    functional_summary = (
        functional.groupby(["ieee_case", "functional_id", "outcome"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for outcome in ("win", "tie", "loss"):
        if outcome not in functional_summary:
            functional_summary[outcome] = 0
    atomic_write_csv(
        context.output_dir / "structural_case_functional_summary.csv",
        functional_summary,
    )
    structures_differ = True
    for _case, local in groups.groupby("ieee_case", sort=True):
        row_sets = set(local["selected_rows"].astype(str))
        column_sets = set(local["selected_columns"].astype(str))
        support_sets = set(local["support_pattern_fingerprint"].astype(str))
        structures_differ &= (
            len(row_sets) == len(local)
            and len(column_sets) == len(local)
            and len(support_sets) == len(local)
        )
    return {
        "structural_groups": len(merged),
        "row_column_support_structures_all_distinct_within_case": bool(structures_differ),
        "wins_by_policy": {
            str(key): int((local["outcome"] == "win").sum())
            for key, local in merged.groupby("candidate_policy", sort=True)
        },
        "losses_by_policy": {
            str(key): int((local["outcome"] == "loss").sum())
            for key, local in merged.groupby("candidate_policy", sort=True)
        },
    }


def stage_stability(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_stability(context)
    frame = _augment_instance_metadata(context, "support_stability.csv")
    summary = pd.read_csv(context.output_dir / "support_stability_summary.csv")
    group_summary = (
        frame.groupby(["structural_group_id", "ieee_case", "selector"], sort=True)
        .agg(
            subset_comparisons=("jaccard_similarity", "count"),
            median_jaccard=("jaccard_similarity", "median"),
            worst_jaccard=("jaccard_similarity", "min"),
            mean_jaccard=("jaccard_similarity", "mean"),
            heldout_median_normalized_error=(
                "heldout_median_normalized_error_full_support",
                "median",
            ),
        )
        .reset_index()
    )
    atomic_write_csv(context.output_dir / "support_stability_group_summary.csv", group_summary)
    result["summary_rows"] = len(summary)
    result["group_summary_rows"] = len(group_summary)
    return result


def stage_certificates(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_certificates(context)
    _augment_instance_metadata(context, "certificate_results.csv")
    return result


def stage_resources(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_resources(context)
    frame = _augment_instance_metadata(context, "resource_registry.csv")
    completed = frame[frame["status"] == "completed"]
    group_summary = (
        completed.groupby(["structural_group_id", "ieee_case", "selector"], sort=True)
        .agg(
            realization_count=("instance_id", "nunique"),
            support_records=("support_id", "count"),
            median_nonzeros=("actual_nonzeros", "median"),
            median_slots=("slot_count", "median"),
            median_gates=("signal_unitary_gate_count", "median"),
            median_depth=("signal_unitary_depth", "median"),
            median_cx=("cx_count", "median"),
            median_controlled_rotations=("controlled_rotations", "median"),
            minimum_reconstruction_error=("wrapper_reconstruction_error", "min"),
            maximum_reconstruction_error=("wrapper_reconstruction_error", "max"),
        )
        .reset_index()
    )
    atomic_write_csv(context.output_dir / "resource_group_summary.csv", group_summary)
    failed = frame[frame["status"] != "completed"]
    validation_failures = failed[
        failed["failure_reason"].fillna("").str.startswith(
            ("resource_compilation_limit", "other_verified_failure")
        )
    ]
    result["unavailable_support_resource_records"] = len(failed) - len(
        validation_failures
    )
    result["executed_wrapper_validation_failures"] = len(validation_failures)
    result["reconstruction_error_range"] = [
        float(completed["wrapper_reconstruction_error"].min()),
        float(completed["wrapper_reconstruction_error"].max()),
    ]
    return result


def stage_pareto(context: StructuralContext) -> dict[str, Any]:
    result = _generalization.stage_pareto(context)
    heldout = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    completed = heldout[heldout["status"] == "completed"].copy()
    base_columns = [
        "instance_id",
        "structural_group_id",
        "ieee_case",
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "actual_nonzeros",
        "slot_count",
        "median_normalized_error",
        "median_absolute_error",
        "random_replicate",
    ]
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    depth = completed[base_columns].merge(
        resources[
            [
                "support_id",
                "signal_unitary_depth",
                "resource_record_type",
                "status",
            ]
        ],
        on="support_id",
        how="left",
        suffixes=("", "_resource"),
    )
    depth["missing_resource_cost_is_zero"] = False
    candidates, frontier = grouped_pareto_frontier(
        depth,
        group_columns=["instance_id"],
        error_column="median_normalized_error",
        cost_column="signal_unitary_depth",
        tie_columns=["support_id"],
    )
    atomic_write_csv(context.output_dir / "pareto_candidates_error_depth.csv", candidates)
    atomic_write_csv(context.output_dir / "pareto_frontier_error_depth.csv", frontier)
    for path in (
        "pareto_candidates_error_nnz.csv",
        "pareto_frontier_error_nnz.csv",
        "pareto_candidates_error_slots.csv",
        "pareto_frontier_error_slots.csv",
        "pareto_candidates_error_gates.csv",
        "pareto_frontier_error_gates.csv",
    ):
        frame = pd.read_csv(context.output_dir / path)
        if not frame.empty and "instance_id" in frame.columns:
            _augment_instance_metadata(context, path)
    result["error_depth_candidates"] = len(candidates)
    result["error_depth_frontier"] = len(frontier)
    return result


def _qsvt_instance_ids(context: StructuralContext) -> list[str]:
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    selected_ids: list[str] = []
    for case_name in context.config["required_cases"]:
        selected_groups = (
            groups[groups["ieee_case"] == case_name]
            .sort_values("selection_order", kind="stable")
            .head(int(context.config["qsvt"]["groups_per_case"]))
        )
        for group_id in selected_groups["structural_group_id"]:
            matched = instances[
                (instances["structural_group_id"] == group_id)
                & (instances["realization_order"] == 1)
            ]
            if len(matched) != 1:
                raise RuntimeError(
                    f"qsvt_statevector_failure: first realization missing for {group_id}"
                )
            selected_ids.append(str(matched.iloc[0]["instance_id"]))
    return selected_ids


def stage_qsvt(context: StructuralContext) -> dict[str, Any]:
    expected_ids = _qsvt_instance_ids(context)
    context.config["qsvt"]["predeclared_instance_ids"] = expected_ids
    result = _generalization.stage_qsvt(context)
    frame = _augment_instance_metadata(context, "qsvt_validation_results.csv")
    designs_path = context.output_dir / "qsvt_instance_designs.json"
    designs = json.loads(designs_path.read_text(encoding="utf-8"))
    instances = pd.read_csv(context.output_dir / "instance_registry.csv").set_index("instance_id")
    design_rows: list[dict[str, Any]] = []
    for instance_id, design in designs.items():
        instance = instances.loc[instance_id]
        design["study_id"] = STUDY_ID
        design["structural_group_id"] = instance["structural_group_id"]
        design["realization_order"] = int(instance["realization_order"])
        design["common_design_applies_to_all_supports"] = True
        design_rows.append(
            {
                "structural_group_id": instance["structural_group_id"],
                "instance_id": instance_id,
                "ieee_case": design["ieee_case"],
                "common_design_fingerprint": design["common_design_fingerprint"],
                "common_mu": design["common_mu"],
                "common_beta": design["common_beta"],
                "physical_alpha": design["physical_alpha"],
                "common_lambda": design["common_lambda"],
                "common_C": design["common_C"],
                "degree": design["degree"],
                "phase_count": design["phase_count"],
                "support_count": len(design["support_subset"]),
                "per_support_phase_refit": False,
                "fit_max_abs_error": design["fit_max_abs_error"],
            }
        )
    atomic_write_json(designs_path, designs)
    atomic_write_csv(
        context.output_dir / "qsvt_common_design_summary.csv",
        pd.DataFrame(design_rows),
    )
    expected_groups = len(context.config["required_cases"]) * int(
        context.config["qsvt"]["groups_per_case"]
    )
    if frame["structural_group_id"].nunique() != expected_groups:
        raise RuntimeError("qsvt_statevector_failure: structural QSVT subset incomplete")
    result["structural_groups"] = int(frame["structural_group_id"].nunique())
    return result


def stage_finite_shot(context: StructuralContext) -> dict[str, Any]:
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    completed = qsvt[qsvt["status"] == "completed"].drop_duplicates("support_id")
    selected: list[pd.Series] = []
    for case_name in context.config["required_cases"]:
        local = completed[completed["ieee_case"] == case_name]
        first_group = sorted(local["structural_group_id"].unique())[0]
        for selector in ("balanced_magnitude", "sensitivity_initial_mean"):
            matched = local[
                (local["structural_group_id"] == first_group) & (local["selector"] == selector)
            ]
            if len(matched) != 1:
                raise RuntimeError("finite-shot projection subset is incomplete")
            selected.append(matched.iloc[0])
    settings = context.config["finite_shot"]
    shots = int(settings["attempted_shots_per_seed"])
    seed_count = int(settings["seed_count"])
    projected_attempts = len(selected) * shots * seed_count
    projected_gates = int(
        sum(
            float(row["component_composed_gates_per_attempt"]) * shots * seed_count
            for row in selected
        )
    )
    ceilings_hold = projected_attempts <= int(
        settings["attempted_shot_ceiling"]
    ) and projected_gates <= int(settings["projected_gate_application_ceiling"])
    execute_requested = bool(settings["execute_by_default"])
    status = "skipped_under_frozen_cost_ceiling"
    if not execute_requested:
        reason = (
            "Finite-shot execution was not enabled in the predeclared configuration; "
            "the already-frozen integrated readout evidence is not duplicated."
        )
    elif not ceilings_hold:
        reason = (
            "finite_shot_cost_ceiling_exceeded: projected attempts or component-composed "
            "gate applications exceed the frozen ceiling."
        )
    else:
        status = "partially_executed"
        reason = (
            "Execution was enabled and cost ceilings held, but no independent counts were "
            "generated by this structural-only benchmark."
        )
    rows = [
        {
            "instance_id": row["instance_id"],
            "structural_group_id": row["structural_group_id"],
            "ieee_case": row["ieee_case"],
            "support_id": row["support_id"],
            "selector": row["selector"],
            "attempted_shots_per_seed": shots,
            "seed_count": seed_count,
            "projected_attempted_shots": shots * seed_count,
            "projected_gate_applications": int(
                float(row["component_composed_gates_per_attempt"]) * shots * seed_count
            ),
            "actual_attempted_shots": np.nan,
            "actual_postselected_shots": np.nan,
            "status": status,
            "failure_reason": reason,
        }
        for row in selected
    ]
    atomic_write_csv(context.output_dir / "finite_shot_results.csv", pd.DataFrame(rows))
    _atomic_write_text(
        context.output_dir / "finite_shot_status.md",
        "\n".join(
            [
                "# Finite-Shot Status",
                "",
                f"- Status: `{status}`",
                f"- Projected attempted shots: {projected_attempts}",
                f"- Frozen attempted-shot ceiling: {settings['attempted_shot_ceiling']}",
                f"- Projected component-composed gate applications: {projected_gates}",
                "- Frozen gate-application ceiling: "
                f"{settings['projected_gate_application_ceiling']}",
                f"- Cost ceilings hold: {ceilings_hold}",
                f"- Reason: {reason}",
                "- No finite-shot counts were fabricated.",
                "",
            ]
        ),
    )
    return {
        "status": status,
        "projected_attempted_shots": projected_attempts,
        "projected_gate_applications": projected_gates,
        "cost_ceilings_hold": ceilings_hold,
        "executed_counts": False,
    }


def _selector_structural_summary(context: StructuralContext) -> pd.DataFrame:
    support = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    certificates = pd.read_csv(context.output_dir / "certificate_results.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    stability = pd.read_csv(context.output_dir / "support_stability_summary.csv")
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    primary = context.config["primary_comparison"]
    k_budget = int(primary["k_budget"])
    slot_budget = int(primary["slot_budget"])
    selectors = [*_generalization.DETERMINISTIC_SELECTORS, _generalization.RANDOM_SELECTOR]
    rows: list[dict[str, Any]] = []
    for scope in [*context.config["required_cases"], "overall"]:
        for selector in selectors:
            registry = support[support["selector"] == selector]
            selected = heldout[
                (heldout["selector"] == selector)
                & (heldout["k_budget"] == k_budget)
                & (heldout["slot_budget"] == slot_budget)
                & (heldout["status"] == "completed")
            ]
            resource = resources[
                (resources["selector"] == selector)
                & (resources["k_budget"] == k_budget)
                & (resources["slot_budget"] == slot_budget)
                & (resources["status"] == "completed")
            ]
            certificate = certificates[
                (certificates["selector"] == selector)
                & (certificates["k_budget"] == k_budget)
                & (certificates["slot_budget"] == slot_budget)
            ]
            qsvt_rows = qsvt[qsvt["selector"] == selector]
            if scope != "overall":
                registry = registry[registry["ieee_case"] == scope]
                selected = selected[selected["ieee_case"] == scope]
                resource = resource[resource["ieee_case"] == scope]
                certificate = certificate[certificate["ieee_case"] == scope]
                qsvt_rows = qsvt_rows[qsvt_rows["ieee_case"] == scope]
            groups = (
                selected.groupby(["structural_group_id", "ieee_case"], sort=True)
                .agg(
                    group_error=("median_normalized_error", "mean"),
                    group_absolute_error=("median_absolute_error", "mean"),
                    group_failure_fraction=("failure_fraction", "mean"),
                )
                .reset_index()
            )
            baseline = heldout[
                (heldout["selector"] == "balanced_magnitude")
                & (heldout["k_budget"] == k_budget)
                & (heldout["slot_budget"] == slot_budget)
                & (heldout["status"] == "completed")
            ]
            if scope != "overall":
                baseline = baseline[baseline["ieee_case"] == scope]
            baseline_groups = (
                baseline.groupby("structural_group_id", sort=True)["median_normalized_error"]
                .mean()
                .rename("baseline_error")
                .reset_index()
            )
            matched = groups.merge(baseline_groups, on="structural_group_id", how="inner")
            outcomes = [
                classify_matched_errors(
                    candidate_error,
                    baseline_error,
                    relative_tolerance=float(primary["tie_relative_tolerance"]),
                    epsilon=float(primary["tie_epsilon"]),
                )
                for candidate_error, baseline_error in zip(
                    matched["group_error"], matched["baseline_error"], strict=True
                )
            ]
            outcome_counts = pd.Series(outcomes, dtype=str).value_counts()
            stability_row = stability[stability["selector"] == selector]
            rows.append(
                {
                    "summary_scope": scope,
                    "selector": selector,
                    "k_budget": k_budget,
                    "slot_budget": slot_budget,
                    "structural_groups_evaluated": int(groups["structural_group_id"].nunique()),
                    "realizations_evaluated": int(selected["instance_id"].nunique()),
                    "feasible_support_fraction_all_budgets": float(
                        (registry["status"] == "completed").mean()
                    ),
                    "median_group_normalized_error": float(groups["group_error"].median()),
                    "p90_group_normalized_error": float(groups["group_error"].quantile(0.9)),
                    "worst_group_normalized_error": float(groups["group_error"].max()),
                    "median_group_absolute_error": float(groups["group_absolute_error"].median()),
                    "failure_fraction": float(groups["group_failure_fraction"].mean()),
                    "wins_vs_magnitude": int(outcome_counts.get("win", 0)),
                    "ties_vs_magnitude": int(outcome_counts.get("tie", 0)),
                    "losses_vs_magnitude": int(outcome_counts.get("loss", 0)),
                    "median_signal_unitary_gate_count": float(
                        resource["signal_unitary_gate_count"].median()
                    ),
                    "median_signal_unitary_depth": float(resource["signal_unitary_depth"].median()),
                    "median_support_stability_jaccard": (
                        float(stability_row.iloc[0]["median_jaccard"])
                        if not stability_row.empty
                        else np.nan
                    ),
                    "certificate_coverage": float(
                        certificate["certificate_holds"].astype(bool).mean()
                    ),
                    "median_certificate_tightness": float(
                        certificate["certificate_tightness"].median()
                    ),
                    "qsvt_validation_status": (
                        "passed"
                        if not qsvt_rows.empty and bool((qsvt_rows["status"] == "completed").all())
                        else "not_in_predeclared_subset"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _collect_failure_registry(context: StructuralContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = (
        ("candidate_exclusion_registry.csv", "candidate_id", "candidates"),
        ("instance_exclusion_registry.csv", "instance_id", "realizations"),
        ("support_registry.csv", "support_id", "supports"),
        ("resource_registry.csv", "support_id", "resources"),
        ("qsvt_validation_results.csv", "support_id", "qsvt"),
    )
    for relative_path, identifier_column, stage in sources:
        path = context.output_dir / relative_path
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if stage in {"candidates", "realizations"}:
            failed = frame.copy()
        else:
            failed = frame[frame["status"] != "completed"].copy()
        for record in failed.to_dict(orient="records"):
            reason = str(record.get("failure_reason", ""))
            if not reason or reason == "nan":
                reason = "unspecified_retained_failure"
            category = reason.split(":", 1)[0].split(";", 1)[0]
            rows.append(
                {
                    "stage": stage,
                    "identifier": str(record.get(identifier_column, "")),
                    "structural_group_id": str(record.get("structural_group_id", "")).replace(
                        "nan", ""
                    ),
                    "instance_id": str(record.get("instance_id", "")).replace("nan", ""),
                    "ieee_case": str(record.get("ieee_case", "")).replace("nan", ""),
                    "failure_category": category,
                    "failure_reason": reason,
                    "retained_not_silently_dropped": True,
                }
            )
    columns = [
        "stage",
        "identifier",
        "structural_group_id",
        "instance_id",
        "ieee_case",
        "failure_category",
        "failure_reason",
        "retained_not_silently_dropped",
    ]
    return pd.DataFrame(rows, columns=columns)


def structural_generalization_verdict(
    *,
    primary: Mapping[str, Any],
    qsvt: pd.DataFrame,
    structural_requirements_hold: bool,
) -> str:
    required_cases = {"ieee14", "ieee30", "ieee57"}
    qsvt_passes = set(qsvt["ieee_case"].astype(str)) == required_cases and bool(
        (qsvt["status"] == "completed").all()
    )
    outcomes = primary["overall_win_tie_loss"]
    group_count = int(primary["structural_groups"])
    clear_majority = int(outcomes["win"]) > group_count / 2 and int(outcomes["win"]) > int(
        outcomes["loss"]
    )
    bootstrap_favors = (
        float(primary["median_paired_difference_sensitivity_minus_magnitude"]) < 0.0
        and float(primary["bootstrap_probability_median_difference_below_zero"]) > 0.5
    )
    represented = set(primary["case_win_tie_loss"]) == required_cases
    if not (represented and structural_requirements_hold and qsvt_passes):
        return "Structural benchmark incomplete due to documented blockers"
    if not (int(outcomes["win"]) > int(outcomes["loss"]) and bootstrap_favors):
        return "Previous value-level gains did not generalize reliably across structural blocks"
    case_consistent = all(
        int(counts["win"]) > int(counts["loss"]) for counts in primary["case_win_tie_loss"].values()
    )
    functional_consistent = all(
        int(counts["win"]) >= int(counts["loss"])
        for counts in primary["functional_win_tie_loss"].values()
    )
    if clear_majority and case_consistent and functional_consistent:
        return (
            "Output-aware selection generalized across structurally diverse PSSE-derived workloads"
        )
    return "Structural generalization is supported but remains case- or functional-dependent"


def _scientific_definition_of_done(
    context: StructuralContext,
    *,
    verdict: str,
) -> dict[str, Any]:
    candidates = pd.read_csv(context.output_dir / "candidate_registry.csv")
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    primary = json.loads(
        (context.output_dir / "structural_primary_test.json").read_text(encoding="utf-8")
    )
    certificates = pd.read_csv(context.output_dir / "certificate_results.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    finite = pd.read_csv(context.output_dir / "finite_shot_results.csv")
    per_case_groups = groups.groupby("ieee_case").size()
    structural_distinct = all(
        local["selected_rows"].nunique() == len(local)
        and local["selected_columns"].nunique() == len(local)
        and local["support_pattern_fingerprint"].nunique() == len(local)
        for _case, local in groups.groupby("ieee_case", sort=True)
    )
    criteria = {
        "candidate_pool_contains_120_predeclared_blocks": len(candidates) == 120,
        "candidate_formation_is_outcome_independent": bool(
            candidates["outcome_independent"].astype(bool).all()
            and (~candidates["selector_outcomes_used_for_inclusion"].astype(bool)).all()
        ),
        "all_three_ieee_cases_represented": set(groups["ieee_case"])
        == {"ieee14", "ieee30", "ieee57"},
        "at_least_three_structural_groups_per_case": bool((per_case_groups >= 3).all()),
        "row_column_and_support_structures_differ_within_cases": structural_distinct,
        "two_realizations_per_structural_group": bool(
            (instances.groupby("structural_group_id").size() == 2).all()
        ),
        "frozen_primary_comparison_completed_at_group_unit": int(
            primary["structural_groups"]
        )
        == len(groups)
        and primary["bootstrap_resampling_unit"] == "structural_group",
        "group_level_and_case_stratified_bootstraps_executed": (
            context.output_dir / "structural_group_bootstrap.csv"
        ).is_file()
        and (
            context.output_dir / "structural_case_stratified_bootstrap.csv"
        ).is_file(),
        "certificates_cover_all_completed_rows_without_violation": bool(
            certificates["certificate_holds"].astype(bool).all()
        ),
        "completed_deterministic_resource_wrappers_valid_and_failures_retained": bool(
            resources.loc[
                resources["selector"].isin(_generalization.DETERMINISTIC_SELECTORS)
                & resources["status"].eq("completed"),
                "wrapper_reconstruction_holds",
            ]
            .astype(bool)
            .all()
            and resources.loc[
                resources["selector"].isin(_generalization.DETERMINISTIC_SELECTORS)
                & resources["status"].ne("completed"),
                "failure_reason",
            ]
            .fillna("")
            .str.len()
            .gt(0)
            .all()
        ),
        "four_resource_matched_pareto_frontiers_exist": all(
            (context.output_dir / f"pareto_frontier_error_{cost}.csv").is_file()
            for cost in ("nnz", "slots", "gates", "depth")
        ),
        "common_design_qsvt_passes_all_cases": set(qsvt["ieee_case"])
        == {"ieee14", "ieee30", "ieee57"}
        and bool((qsvt["status"] == "completed").all()),
        "finite_shot_status_is_explicit_and_nonfabricated": set(finite["status"])
        <= {
            "executed",
            "partially_executed",
            "skipped_under_frozen_cost_ceiling",
        }
        and bool(finite["actual_attempted_shots"].isna().all()),
        "final_status_is_one_of_predeclared_choices": verdict
        in {
            "Output-aware selection generalized across structurally diverse PSSE-derived workloads",
            "Structural generalization is supported but remains case- or functional-dependent",
            "Previous value-level gains did not generalize reliably across structural blocks",
            "Structural benchmark incomplete due to documented blockers",
        },
    }
    return {
        "criteria": {
            name: {"status": "PASS" if passed else "FAIL", "passed": bool(passed)}
            for name, passed in criteria.items()
        },
        "all_scientific_criteria_pass": all(criteria.values()),
        "final_status": verdict,
    }


def stage_summary(context: StructuralContext) -> dict[str, Any]:
    selector_summary = _selector_structural_summary(context)
    atomic_write_csv(context.output_dir / "selector_structural_summary.csv", selector_summary)
    failures = _collect_failure_registry(context)
    atomic_write_csv(context.output_dir / "failure_registry.csv", failures)
    primary = json.loads(
        (context.output_dir / "structural_primary_test.json").read_text(encoding="utf-8")
    )
    secondary = json.loads(
        (context.output_dir / "structural_secondary_test.json").read_text(encoding="utf-8")
    )
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    structural_requirements_hold = all(
        local["selected_rows"].nunique() == len(local)
        and local["selected_columns"].nunique() == len(local)
        and local["support_pattern_fingerprint"].nunique() == len(local)
        for _case, local in groups.groupby("ieee_case", sort=True)
    ) and bool((groups.groupby("ieee_case").size() >= 3).all())
    verdict = structural_generalization_verdict(
        primary=primary,
        qsvt=qsvt,
        structural_requirements_hold=structural_requirements_hold,
    )
    done = _scientific_definition_of_done(context, verdict=verdict)
    atomic_write_json(context.output_dir / "definition_of_done.json", done)
    functional = primary["functional_win_tie_loss"]
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    resource_validation_failures = resources[
        resources["status"].ne("completed")
        & resources["failure_reason"].fillna("").str.startswith(
            ("resource_compilation_limit", "other_verified_failure")
        )
    ]
    certificate = pd.read_csv(context.output_dir / "certificate_results.csv")
    stability = pd.read_csv(context.output_dir / "support_stability_summary.csv")
    finite = pd.read_csv(context.output_dir / "finite_shot_results.csv")
    lines = [
        "# Structurally Diverse Output-Aware Sparse Selection Summary",
        "",
        f"- Structural groups: {len(groups)}",
        f"- Matrix realizations: {len(instances)}",
        f"- Cases: {', '.join(sorted(groups['ieee_case'].unique()))}",
        f"- Primary win/tie/loss: {primary['overall_win_tie_loss']}",
        "- Primary median paired difference (sensitivity minus magnitude): "
        f"{primary['median_paired_difference_sensitivity_minus_magnitude']:.12g}",
        f"- Primary group-bootstrap 95% interval: {primary['bootstrap_confidence_interval_95']}",
        "- Case-stratified group-bootstrap 95% interval: "
        f"{primary['case_stratified_bootstrap_confidence_interval_95']}",
        f"- Secondary win/tie/loss: {secondary['overall_win_tie_loss']}",
        f"- Functional win/tie/loss: {functional}",
        f"- Certificate violations: {int((~certificate['certificate_holds'].astype(bool)).sum())}",
        f"- Executed resource-wrapper validation failures: {len(resource_validation_failures)}",
        "- Resource records unavailable because supports were infeasible: "
        f"{int((resources['status'] != 'completed').sum()) - len(resource_validation_failures)}",
        f"- QSVT structural groups: {qsvt['structural_group_id'].nunique()}",
        f"- QSVT failures: {int((qsvt['status'] != 'completed').sum())}",
        f"- Minimum stability Jaccard: {stability['worst_jaccard'].min():.6g}",
        f"- Finite-shot status: {finite.iloc[0]['status']}",
        f"- Final status: **{verdict}**",
        "",
        "Generated measurements are controlled PYPOWER benchmark-model calculations, not "
        "field PMU/SCADA data. QSVT results are local statevector simulations, not hardware "
        "execution. Numerical realizations are not counted as independent structures.",
        "",
    ]
    _atomic_write_text(context.output_dir / "summary.md", "\n".join(lines))
    metrics = {
        "study_id": STUDY_ID,
        "final_status": verdict,
        "cases": sorted(groups["ieee_case"].unique().tolist()),
        "structural_groups": len(groups),
        "matrix_realizations": len(instances),
        "primary": primary,
        "secondary": secondary,
        "qsvt_structural_groups": int(qsvt["structural_group_id"].nunique()),
        "finite_shot_status": str(finite.iloc[0]["status"]),
        "failure_records": len(failures),
    }
    atomic_write_json(context.output_dir / "summary_metrics.json", metrics)
    for group in groups.itertuples(index=False):
        group_id = str(group.structural_group_id)
        group_pairs = pd.read_csv(context.output_dir / "structural_primary_matched_pairs.csv")
        matched = group_pairs[group_pairs["structural_group_id"] == group_id]
        group_resources = resources[resources["structural_group_id"] == group_id]
        group_qsvt = qsvt[qsvt["structural_group_id"] == group_id]
        atomic_write_json(
            context.output_dir / "groups" / group_id / "group_summary.json",
            {
                "structural_group_id": group_id,
                "ieee_case": group.ieee_case,
                "primary_comparison": (matched.iloc[0].to_dict() if not matched.empty else None),
                "resource_records": len(group_resources),
                "resource_failures": int((group_resources["status"] != "completed").sum()),
                "qsvt_rows": len(group_qsvt),
                "qsvt_status": (
                    "passed"
                    if not group_qsvt.empty and bool((group_qsvt["status"] == "completed").all())
                    else "not_in_predeclared_subset"
                ),
            },
        )
    return {
        "selector_summary_rows": len(selector_summary),
        "failure_records": len(failures),
        "final_status": verdict,
        "scientific_definition_of_done": done["all_scientific_criteria_pass"],
    }


def _compare_protected_snapshot(context: StructuralContext) -> dict[str, Any]:
    initial = json.loads(
        (context.output_dir / "protected_path_snapshot.json").read_text(encoding="utf-8")
    )
    original = dict(initial["files"])
    current = _protected_file_snapshot(context.root, initial["protected_paths"])
    changed = sorted(
        path for path in set(original) & set(current) if original[path] != current[path]
    )
    added = sorted(set(current).difference(original))
    deleted = sorted(set(original).difference(current))
    return {
        "pass": not changed and not added and not deleted,
        "initial_file_count": len(original),
        "current_file_count": len(current),
        "changed": changed,
        "added": added,
        "deleted": deleted,
    }


def _internal_verification_checks(context: StructuralContext) -> dict[str, Any]:
    frozen = json.loads(
        (context.output_dir / "frozen_method_configuration.json").read_text(encoding="utf-8")
    )
    candidates = pd.read_csv(context.output_dir / "candidate_registry.csv")
    descriptors = pd.read_csv(context.output_dir / "structural_descriptors.csv")
    groups = pd.read_csv(context.output_dir / "structural_group_registry.csv")
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    functionals = pd.read_csv(context.output_dir / "functional_registry.csv")
    residuals = pd.read_csv(context.output_dir / "residual_registry.csv")
    supports = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout = pd.read_csv(context.output_dir / "heldout_results.csv")
    certificates = pd.read_csv(context.output_dir / "certificate_results.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    primary = json.loads(
        (context.output_dir / "structural_primary_test.json").read_text(encoding="utf-8")
    )
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    protected = _compare_protected_snapshot(context)
    split_overlap = 0
    for _instance_id, local in residuals.groupby("instance_id", sort=True):
        split_overlap += len(
            set(local.loc[local["split"] == "training", "residual_seed"])
            & set(local.loc[local["split"] == "held_out", "residual_seed"])
        )
    support_constraint_failures = 0
    for record in supports[supports["status"] == "completed"].itertuples(index=False):
        payload = json.loads(
            (context.output_dir / "instances" / f"{record.instance_id}.json").read_text(
                encoding="utf-8"
            )
        )
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        selected = _generalization.load_generalization_support(context, str(record.support_file))
        report = support_constraint_report(
            matrix,
            selected,
            SupportConstraints(int(record.k_budget), int(record.slot_budget), True),
        )
        support_constraint_failures += int(not report["valid"])
    structures_distinct = all(
        local["selected_rows"].nunique() == len(local)
        and local["selected_columns"].nunique() == len(local)
        and local["support_pattern_fingerprint"].nunique() == len(local)
        for _case, local in groups.groupby("ieee_case", sort=True)
    )
    rows_columns_fixed = True
    for _group_id, local in instances.groupby("structural_group_id", sort=True):
        rows_columns_fixed &= (
            local["selected_rows"].nunique() == 1 and local["selected_columns"].nunique() == 1
        )
    deterministic_resources = resources[
        resources["selector"].isin(_generalization.DETERMINISTIC_SELECTORS)
    ]
    completed_deterministic_resources = deterministic_resources[
        deterministic_resources["status"] == "completed"
    ]
    failed_deterministic_resources = deterministic_resources[
        deterministic_resources["status"] != "completed"
    ]
    checks = {
        "frozen_method_fingerprint_valid": frozen["configuration_fingerprint"]
        == configuration_fingerprint(frozen),
        "previous_method_configuration_reused": frozen["previous_configuration_fingerprint"]
        == json.loads(
            (context.root / PRIOR_BENCHMARK_DIR / "frozen_selector_configuration.json").read_text(
                encoding="utf-8"
            )
        )["configuration_fingerprint"],
        "candidate_registry_has_120_rows": len(candidates) == 120,
        "candidate_policy_balance_holds": bool(
            (candidates.groupby(["ieee_case", "policy"]).size() == 4).all()
        ),
        "candidate_and_descriptor_outcome_isolation": bool(
            candidates["outcome_independent"].astype(bool).all()
            and (~candidates["selector_outcomes_used_for_inclusion"].astype(bool)).all()
            and (~descriptors["descriptor_uses_selector_or_output_results"].astype(bool)).all()
        ),
        "minimum_structural_groups_per_case": bool((groups.groupby("ieee_case").size() >= 3).all()),
        "required_cases_represented": set(groups["ieee_case"]) == {"ieee14", "ieee30", "ieee57"},
        "structural_rows_columns_supports_distinct": bool(structures_distinct),
        "structural_selection_outcome_independent": bool(
            (~groups["selector_outcomes_used_for_selection"].astype(bool)).all()
        ),
        "two_realizations_per_group": bool(
            (instances.groupby("structural_group_id").size() == 2).all()
        ),
        "rows_columns_fixed_within_group": bool(rows_columns_fixed),
        "realizations_not_counted_as_structures": primary[
            "numerical_realizations_treated_as_independent_structures"
        ]
        is False,
        "three_functionals_per_realization": len(functionals) == 3 * len(instances),
        "functional_design_outcome_independent": bool(
            functionals["selection_data_used"].eq("state_metadata_only_no_output_accuracy").all()
        ),
        "residual_split_overlap_zero": split_overlap == 0,
        "residual_counts_complete": bool(
            (
                residuals[residuals["generation_status"] == "completed"]
                .groupby(["instance_id", "split"])
                .size()
                == 20
            ).all()
        ),
        "support_selection_has_no_heldout_leakage": bool(
            ~supports["selection_data_split"]
            .astype(str)
            .str.contains("held", case=False, na=False)
            .any()
        ),
        "support_constraints_hold": support_constraint_failures == 0,
        "heldout_rows_only": set(heldout["split"].dropna()) == {"held_out"},
        "primary_uses_structural_group_unit": primary["bootstrap_resampling_unit"]
        == "structural_group",
        "primary_contains_all_selected_groups": int(primary["structural_groups"]) == len(groups),
        "certificate_formula_unchanged": bool(
            certificates["certificate_formula_version"]
            .eq(context.config["certificate"]["formula_version"])
            .all()
        ),
        "certificate_actual_error_not_used": bool(
            (~certificates["certificate_used_actual_error_in_computation"].astype(bool)).all()
        ),
        "certificate_violations_zero": bool(certificates["certificate_holds"].astype(bool).all()),
        "deterministic_resources_measured": bool(
            completed_deterministic_resources["signal_unitary_gate_count"].gt(0).all()
            and completed_deterministic_resources["signal_unitary_depth"].gt(0).all()
            and completed_deterministic_resources["wrapper_reconstruction_holds"]
            .astype(bool)
            .all()
            and failed_deterministic_resources["failure_reason"]
            .fillna("")
            .str.len()
            .gt(0)
            .all()
        ),
        "resource_costs_not_zero_filled": bool(
            (~completed_deterministic_resources["missing_cost_is_zero"].astype(bool)).all()
        ),
        "all_four_pareto_frontiers_exist": all(
            (context.output_dir / f"pareto_frontier_error_{cost}.csv").is_file()
            for cost in ("nnz", "slots", "gates", "depth")
        ),
        "qsvt_two_groups_per_case": bool(
            (qsvt.drop_duplicates("structural_group_id").groupby("ieee_case").size() == 2).all()
        ),
        "qsvt_failures_zero": bool((qsvt["status"] == "completed").all()),
        "qsvt_no_per_support_refit": bool((~qsvt["per_support_phase_refit"].astype(bool)).all()),
        "qsvt_same_sparse_matrix": bool(
            qsvt["ridge_qsvt_identical_sparse_matrix"].astype(bool).all()
        ),
        "support_and_qsvt_errors_separate": bool(
            qsvt["support_error_separate_from_qsvt_error"].astype(bool).all()
        ),
        "protected_paths_unchanged": bool(protected["pass"]),
    }
    return {
        "checks": checks,
        "all_internal_checks_pass": all(checks.values()),
        "protected_path_comparison": protected,
        "counts": {
            "candidates": len(candidates),
            "structural_groups": len(groups),
            "realizations": len(instances),
            "functionals": len(functionals),
            "residuals": len(residuals),
            "supports": len(supports),
            "heldout_rows": len(heldout),
            "certificate_rows": len(certificates),
            "resource_rows": len(resources),
            "qsvt_rows": len(qsvt),
        },
    }


def stage_verify(context: StructuralContext) -> dict[str, Any]:
    report = _internal_verification_checks(context)
    external_path = context.output_dir / "verification_commands.json"
    external = (
        json.loads(external_path.read_text(encoding="utf-8"))
        if external_path.is_file()
        else {"status": "not_recorded_yet", "commands": []}
    )
    external_commands = external.get("commands", [])
    required_categories = {"targeted", "dependent", "full_filewise", "ruff"}
    passed_categories = {
        str(item.get("category")) for item in external_commands if item.get("status") == "PASS"
    }
    external_required_pass = required_categories <= passed_categories
    done = json.loads((context.output_dir / "definition_of_done.json").read_text(encoding="utf-8"))
    verification_criteria = {
        "internal_verification_passes": report["all_internal_checks_pass"],
        "targeted_tests_pass": "targeted" in passed_categories,
        "dependent_tests_pass": "dependent" in passed_categories,
        "full_filewise_suite_pass": "full_filewise" in passed_categories,
        "ruff_pass": "ruff" in passed_categories,
        "protected_paths_unchanged": report["checks"]["protected_paths_unchanged"],
    }
    done["verification_criteria"] = {
        name: {"status": "PASS" if passed else "FAIL", "passed": bool(passed)}
        for name, passed in verification_criteria.items()
    }
    done["all_verification_criteria_pass"] = all(verification_criteria.values())
    done["definition_of_done_pass"] = bool(
        done["all_scientific_criteria_pass"] and done["all_verification_criteria_pass"]
    )
    atomic_write_json(context.output_dir / "definition_of_done.json", done)
    lines = [
        "# Structural-Generalization Verification Report",
        "",
        f"- Generated: {now_iso()}",
        f"- Internal verification: {'PASS' if report['all_internal_checks_pass'] else 'FAIL'}",
        f"- Required external command categories: {'PASS' if external_required_pass else 'FAIL'}",
        "- Protected paths unchanged: "
        f"{'PASS' if report['checks']['protected_paths_unchanged'] else 'FAIL'}",
        f"- Definition of done: {'PASS' if done['definition_of_done_pass'] else 'FAIL'}",
        "",
        "## Internal checks",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Protected paths",
            "",
            f"- Initial files: {report['protected_path_comparison']['initial_file_count']}",
            f"- Current files: {report['protected_path_comparison']['current_file_count']}",
            f"- Changed: {len(report['protected_path_comparison']['changed'])}",
            f"- Added: {len(report['protected_path_comparison']['added'])}",
            f"- Deleted: {len(report['protected_path_comparison']['deleted'])}",
            "",
            "## Executed command results",
            "",
            "```json",
            json.dumps(external, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    _atomic_write_text(context.output_dir / "verification_report.md", "\n".join(lines))
    atomic_write_json(context.output_dir / "internal_verification.json", report)
    if not report["all_internal_checks_pass"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"internal verification failed: {failed}")
    if not external_required_pass:
        missing = sorted(required_categories.difference(passed_categories))
        raise RuntimeError(f"external verification categories missing or failed: {missing}")
    return {
        "all_internal_checks_pass": True,
        "all_required_external_commands_pass": True,
        "protected_paths_unchanged": True,
        "definition_of_done_pass": done["definition_of_done_pass"],
        **report["counts"],
    }


def refresh_manifest_and_checksums(context: StructuralContext) -> None:
    manifest_path = context.output_dir / "manifest.json"
    checksum_path = context.output_dir / "checksums.sha256"
    artifact_paths = [
        path
        for path in sorted(context.output_dir.rglob("*"))
        if path.is_file() and path not in {manifest_path, checksum_path}
    ]
    manifest = {
        "study_id": STUDY_ID,
        "generated_at": now_iso(),
        "artifact_count": len(artifact_paths),
        "top_level_artifact_count": sum(
            path.parent == context.output_dir for path in artifact_paths
        ),
        "grouped_artifact_count": sum(
            "groups" in path.relative_to(context.output_dir).parts for path in artifact_paths
        ),
        "artifacts": [
            {
                "path": path.relative_to(context.output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    atomic_write_json(manifest_path, manifest)
    targets = [
        path
        for path in sorted(context.output_dir.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(context.root).as_posix()}" for path in targets
    ]
    _atomic_write_text(checksum_path, "\n".join(lines) + "\n")


STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "audit": ("implementation_audit.md", "protected_path_snapshot.json"),
    "freeze": ("frozen_method_configuration.json", "study_configuration.json"),
    "candidates": ("candidate_registry.csv", "candidate_exclusion_registry.csv"),
    "descriptors": (
        "structural_descriptors.csv",
        "candidate_pairwise_distances.csv",
        "candidate_distance_from_previous.csv",
        "previous_block_descriptors.json",
    ),
    "structural-selection": (
        "structural_group_registry.csv",
        "selected_structural_groups.csv",
    ),
    "realizations": ("instance_registry.csv", "instance_exclusion_registry.csv"),
    "functionals": ("functional_registry.csv",),
    "residuals": ("residual_registry.csv",),
    "supports": (
        "support_registry.csv",
        "support_selection_results.csv",
        "refinement_traces.csv",
        "entry_scores.csv",
        "training_instance_summary.csv",
    ),
    "heldout": (
        "heldout_results.csv",
        "heldout_instance_summary.csv",
        "heldout_case_summary.csv",
        "heldout_functional_summary.csv",
    ),
    "primary-test": (
        "structural_primary_test.json",
        "structural_primary_matched_pairs.csv",
        "structural_group_bootstrap.csv",
        "structural_case_stratified_bootstrap.csv",
    ),
    "secondary-test": (
        "structural_secondary_test.json",
        "structural_secondary_matched_pairs.csv",
    ),
    "structural-analysis": (
        "structural_performance_analysis.csv",
        "structural_distance_association.csv",
        "structural_case_functional_summary.csv",
    ),
    "stability": (
        "support_stability.csv",
        "support_stability_summary.csv",
        "support_stability_group_summary.csv",
    ),
    "certificates": ("certificate_results.csv", "certificate_case_summary.csv"),
    "resources": (
        "resource_registry.csv",
        "resource_case_summary.csv",
        "resource_group_summary.csv",
    ),
    "pareto": (
        "pareto_frontier_error_nnz.csv",
        "pareto_frontier_error_slots.csv",
        "pareto_frontier_error_gates.csv",
        "pareto_frontier_error_depth.csv",
    ),
    "qsvt": (
        "qsvt_instance_designs.json",
        "qsvt_validation_results.csv",
        "qsvt_common_design_summary.csv",
    ),
    "finite-shot": ("finite_shot_status.md", "finite_shot_results.csv"),
    "summary": (
        "selector_structural_summary.csv",
        "failure_registry.csv",
        "summary.md",
        "summary_metrics.json",
        "definition_of_done.json",
    ),
    "verify": (
        "verification_report.md",
        "internal_verification.json",
        "definition_of_done.json",
    ),
}


STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "freeze": stage_freeze,
    "candidates": stage_candidates,
    "descriptors": stage_descriptors,
    "structural-selection": stage_structural_selection,
    "realizations": stage_realizations,
    "functionals": stage_functionals,
    "residuals": stage_residuals,
    "supports": stage_supports,
    "heldout": stage_heldout,
    "primary-test": stage_primary_test,
    "secondary-test": stage_secondary_test,
    "structural-analysis": stage_structural_analysis,
    "stability": stage_stability,
    "certificates": stage_certificates,
    "resources": stage_resources,
    "pareto": stage_pareto,
    "qsvt": stage_qsvt,
    "finite-shot": stage_finite_shot,
    "summary": stage_summary,
    "verify": stage_verify,
}


def run_structural_generalization_study(
    context: StructuralContext, *, stage: str = "all"
) -> dict[str, Any]:
    if stage != "all" and stage not in STAGES:
        raise ValueError(f"unknown stage {stage}; choose from all or {STAGES}")
    selected_stages = list(STAGES) if stage == "all" else [stage]
    results: dict[str, Any] = {}
    for stage_name in selected_stages:
        if context.resume and not context.force and context.checkpoint.is_complete(stage_name):
            results[stage_name] = {
                **context.checkpoint.read()["stages"][stage_name].get("result", {}),
                "checkpoint_status": "resumed_from_checkpoint",
            }
            continue
        if context.force:
            context.checkpoint.clear(stage_name)
        started = time.perf_counter()
        result = STAGE_FUNCTIONS[stage_name](context)
        elapsed = time.perf_counter() - started
        context.checkpoint.mark_complete(
            stage_name,
            outputs=STAGE_OUTPUTS[stage_name],
            result=result,
            elapsed_seconds=elapsed,
        )
        if stage_name == "verify":
            refresh_manifest_and_checksums(context)
        results[stage_name] = {
            **result,
            "checkpoint_status": "completed",
            "elapsed_seconds": elapsed,
        }
    return results
