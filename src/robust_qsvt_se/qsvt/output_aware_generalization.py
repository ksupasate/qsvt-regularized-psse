"""Multi-instance generalization benchmark for output-aware sparse selection.

The benchmark reuses the exact Ridge sensitivity, resource-constrained selectors,
one-swap refinement, conservative perturbation certificate, sparse wrapper, and
QSVT implementation from the completed single-instance study.  Its orchestration
keeps matrix extraction, training data, held-out data, resource measurement, and
QSVT validation auditable and explicitly separated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    validate_complete_wrapper,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.tqe_revision_support_common import git_commit_hash, now_iso
from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    assign_slot_permutations,
    minimum_slot_count,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.engineering_utils import (
    build_engineering_system,
    ridge_svd_solution,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    SupportSelectionResult,
    _as_quantized_block,
    atomic_write_csv,
    atomic_write_json,
    build_common_padded_wrapper,
    compute_output_aware_entry_scores,
    deterministic_ridge_leverage_scores,
    exact_task_errors,
    refine_support_one_swap,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.qsvt.research_matrix import extract_weighted_jacobian_matrix
from robust_qsvt_se.qsvt.ridge_output_certificate import (
    compute_ridge_selected_output_certificate,
    ridge_selected_output_gradient,
    validate_ridge_selected_output_certificate,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint

DEFAULT_OUTPUT_DIR = Path("outputs/output_aware_generalization")
DEFAULT_CONFIG_PATH = Path("configs/output_aware_generalization.json")
PRIOR_STUDY_DIR = Path("outputs/output_aware_sparse_selection")
STUDY_ID = "output_aware_generalization_v1"

FUNCTIONAL_IDS = (
    "coordinate_e0",
    "signed_difference_e0_minus_e1",
    "aggregate_e0_to_e3",
)
INITIAL_SELECTORS = (
    "balanced_magnitude",
    "leverage_weighted",
    "sensitivity_initial_mean",
    "sensitivity_initial_worst_case",
)
REFINED_SELECTORS = (
    "sensitivity_refined_mean",
    "sensitivity_refined_worst_case",
)
DETERMINISTIC_SELECTORS = (*INITIAL_SELECTORS, *REFINED_SELECTORS)
RANDOM_SELECTOR = "random_objective_feasible"

STAGES = (
    "audit",
    "freeze",
    "instances",
    "functionals",
    "residuals",
    "supports",
    "heldout",
    "primary-test",
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
    "instance_extraction_failure",
    "insufficient_nonzeros",
    "functional_metadata_unavailable",
    "residual_generation_failure",
    "support_budget_infeasible",
    "milp_failure",
    "slot_assignment_failure",
    "certificate_violation",
    "common_normalization_failure",
    "polynomial_fit_failure",
    "qsvt_statevector_failure",
    "finite_shot_cost_ceiling_exceeded",
    "resource_compilation_limit",
    "other_verified_failure",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_ready(dict(payload)), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def configuration_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over a configuration excluding its fingerprint field."""

    values = dict(payload)
    values.pop("configuration_fingerprint", None)
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_generalization_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"generalization configuration must use study_id={STUDY_ID}")
    if not payload.get("declared_before_benchmark_evaluation", False):
        raise ValueError("benchmark configuration must be declared before evaluation")
    cases = tuple(payload.get("required_cases", ()))
    if cases != ("ieee14", "ieee30", "ieee57"):
        raise ValueError("required cases must be IEEE-14, IEEE-30, and IEEE-57")
    if payload["support_budgets"] != [8, 12, 16, 24, 32]:
        raise ValueError("support budgets differ from the frozen development study")
    if payload["slot_budgets"] != [2, 3, 4, 6, 8]:
        raise ValueError("slot budgets differ from the frozen development study")
    return payload


def deterministic_instance_id(case_name: str, matrix_seed: int) -> str:
    case = str(case_name).lower().replace("-", "").replace("_", "")
    if case not in {"ieee14", "ieee30", "ieee57"}:
        raise ValueError(f"unsupported benchmark case {case_name}")
    return f"{case}_eval_seed_{int(matrix_seed)}_block_8x8"


def _top_by_keys(*, count: int, primary: np.ndarray, secondary: np.ndarray,
                 tertiary: np.ndarray) -> np.ndarray:
    indices = np.arange(primary.size, dtype=np.int64)
    order = np.lexsort(
        (indices, -np.asarray(tertiary), -np.asarray(secondary), -np.asarray(primary))
    )
    return np.sort(order[: int(count)])


def extract_coverage_preserving_block(
    full_matrix: np.ndarray,
    *,
    row_count: int = 8,
    column_count: int = 8,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a deterministic dense structural block without outcome information.

    Rows are initialized by global norm.  Columns and rows are then updated
    alternately by nonzero count, restricted norm, global norm, and index.  The
    procedure is deterministic and uses neither Ridge outputs nor selector results.
    """

    values = np.asarray(full_matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("full matrix must be a finite real two-dimensional array")
    if row_count <= 0 or column_count <= 0:
        raise ValueError("block dimensions must be positive")
    if row_count > values.shape[0] or column_count > values.shape[1]:
        raise ValueError("requested block exceeds the full matrix")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    row_norms = np.linalg.norm(values, axis=1)
    column_norms = np.linalg.norm(values, axis=0)
    rows = _top_by_keys(
        count=row_count,
        primary=row_norms,
        secondary=np.zeros_like(row_norms),
        tertiary=np.zeros_like(row_norms),
    )
    columns: np.ndarray | None = None
    for _iteration in range(int(max_iterations)):
        restricted_columns = values[rows, :]
        new_columns = _top_by_keys(
            count=column_count,
            primary=np.count_nonzero(restricted_columns, axis=0).astype(np.float64),
            secondary=np.linalg.norm(restricted_columns, axis=0),
            tertiary=column_norms,
        )
        restricted_rows = values[:, new_columns]
        new_rows = _top_by_keys(
            count=row_count,
            primary=np.count_nonzero(restricted_rows, axis=1).astype(np.float64),
            secondary=np.linalg.norm(restricted_rows, axis=1),
            tertiary=row_norms,
        )
        if (
            columns is not None
            and np.array_equal(new_columns, columns)
            and np.array_equal(new_rows, rows)
        ):
            rows, columns = new_rows, new_columns
            break
        rows, columns = new_rows, new_columns
    if columns is None:
        raise RuntimeError("block extraction did not select columns")
    block = values[np.ix_(rows, columns)]
    return block, rows.astype(np.int64), columns.astype(np.int64)


def _state_records(metadata: Mapping[str, Any], columns: Sequence[int]) -> list[dict[str, Any]]:
    angle_buses = [int(value) for value in metadata.get("angle_state_buses", [])]
    voltage_buses = [int(value) for value in metadata.get("voltage_state_buses", [])]
    labels = list(metadata.get("state_labels", []))
    records: list[dict[str, Any]] = []
    for local_index, raw_column in enumerate(columns):
        column = int(raw_column)
        if column < len(angle_buses):
            state_type = "angle"
            bus_id: int | None = angle_buses[column]
        else:
            voltage_position = column - len(angle_buses)
            if 0 <= voltage_position < len(voltage_buses):
                state_type = "voltage"
                bus_id = voltage_buses[voltage_position]
            else:
                state_type = "unknown"
                bus_id = None
        records.append(
            {
                "local_index": int(local_index),
                "full_state_index": column,
                "state_type": state_type,
                "bus_id": bus_id,
                "label": labels[column] if column < len(labels) else f"state_{column}",
            }
        )
    return records


def build_instance_functionals(
    state_records: Sequence[Mapping[str, Any]], dimension: int = 8
) -> dict[str, dict[str, Any]]:
    """Construct the three frozen local probes with explicit metadata semantics."""

    if dimension < 4 or len(state_records) != dimension:
        raise ValueError("functional construction requires complete metadata for an 8D block")
    first = dict(state_records[0])
    second = dict(state_records[1])
    metadata_complete = all(record.get("state_type") != "unknown" for record in state_records)

    coordinate = np.zeros(dimension, dtype=np.float64)
    coordinate[0] = 1.0
    difference = np.zeros(dimension, dtype=np.float64)
    difference[0] = 1.0 / np.sqrt(2.0)
    difference[1] = -1.0 / np.sqrt(2.0)
    aggregate = np.zeros(dimension, dtype=np.float64)
    aggregate[:4] = 0.5

    compatible = (
        metadata_complete
        and first.get("state_type") == second.get("state_type")
        and first.get("state_type") in {"angle", "voltage"}
    )
    coordinate_semantics = (
        "metadata_grounded_state_coordinate"
        if metadata_complete
        else "nonsemantic_deterministic_probe"
    )
    difference_semantics = (
        "metadata_grounded_compatible_state_difference"
        if compatible
        else "nonsemantic_deterministic_probe"
    )
    aggregate_semantics = (
        "metadata_grounded_state_aggregate"
        if metadata_complete
        else "nonsemantic_deterministic_probe"
    )
    return {
        "coordinate_e0": {
            "vector": coordinate,
            "functional_family": "coordinate",
            "semantic_status": coordinate_semantics,
            "local_indices": [0],
            "state_records": [first],
        },
        "signed_difference_e0_minus_e1": {
            "vector": difference,
            "functional_family": "difference",
            "semantic_status": difference_semantics,
            "local_indices": [0, 1],
            "state_records": [first, second],
        },
        "aggregate_e0_to_e3": {
            "vector": aggregate,
            "functional_family": "aggregate",
            "semantic_status": aggregate_semantics,
            "local_indices": [0, 1, 2, 3],
            "state_records": [dict(record) for record in state_records[:4]],
        },
    }


def residual_seed_ids(
    config: Mapping[str, Any], *, case_name: str, instance_position: int, split: str
) -> list[int]:
    residual = config["residuals"]
    base = int(residual["case_seed_bases"][case_name])
    base += int(instance_position) * int(residual["instance_stride"])
    if split == "training":
        count = int(residual["training_count_per_instance"])
        offset = int(residual["training_offset"])
    elif split == "held_out":
        count = int(residual["held_out_count_per_instance"])
        offset = int(residual["held_out_offset"])
    else:
        raise ValueError("split must be training or held_out")
    return [base + offset + index for index in range(count)]


def jaccard_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("supports must have the same shape")
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left & right) / union)


def classify_matched_errors(
    candidate_error: float,
    baseline_error: float,
    *,
    relative_tolerance: float,
    epsilon: float,
) -> str:
    candidate = float(candidate_error)
    baseline = float(baseline_error)
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        raise ValueError("matched errors must be finite")
    relative = abs(candidate - baseline) / max(abs(candidate), abs(baseline), float(epsilon))
    if relative <= float(relative_tolerance):
        return "tie"
    return "win" if candidate < baseline else "loss"


def paired_instance_bootstrap(
    paired_differences: Sequence[float], *, samples: int, seed: int
) -> pd.DataFrame:
    differences = np.asarray(paired_differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0 or not np.all(np.isfinite(differences)):
        raise ValueError("paired bootstrap requires finite instance-level differences")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(int(samples), differences.size))
    medians = np.median(differences[indices], axis=1)
    return pd.DataFrame(
        {
            "bootstrap_sample": np.arange(int(samples), dtype=np.int64),
            "instance_count": int(differences.size),
            "median_paired_difference_sensitivity_minus_magnitude": medians,
            "resampling_unit": "instance",
            "bootstrap_seed": int(seed),
        }
    )


def grouped_pareto_frontier(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    error_column: str,
    cost_column: str,
    tie_columns: Sequence[str] = ("support_id",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic within-group two-objective Pareto registries."""

    required = {*group_columns, error_column, cost_column, *tie_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pareto frame is missing columns {sorted(missing)}")
    candidates = frame.copy()
    candidates["nondominated"] = False
    grouping: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for _key, indices in candidates.groupby(grouping, sort=True).groups.items():
        local = candidates.loc[indices]
        finite = local[np.isfinite(local[error_column]) & np.isfinite(local[cost_column])]
        for index, row in finite.iterrows():
            dominated = bool(
                (
                    (finite[error_column] <= float(row[error_column]))
                    & (finite[cost_column] <= float(row[cost_column]))
                    & (
                        (finite[error_column] < float(row[error_column]))
                        | (finite[cost_column] < float(row[cost_column]))
                    )
                ).any()
            )
            candidates.loc[index, "nondominated"] = not dominated
    order = [*group_columns, cost_column, error_column, *tie_columns]
    candidates = candidates.sort_values(order, kind="stable", na_position="last").reset_index(
        drop=True
    )
    frontier = candidates[candidates["nondominated"]].reset_index(drop=True)
    return candidates, frontier


def _git_status(root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _protected_file_snapshot(root: Path, configured_paths: Sequence[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative in configured_paths:
        selected = root / relative
        if selected.is_file():
            snapshot[selected.relative_to(root).as_posix()] = _sha256_file(selected)
            continue
        if not selected.exists():
            continue
        for path in sorted(selected.rglob("*")):
            if path.is_file():
                snapshot[path.relative_to(root).as_posix()] = _sha256_file(path)
    return snapshot


class GeneralizationCheckpoint:
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
class GeneralizationContext:
    root: Path
    output_dir: Path
    config_path: Path
    config: dict[str, Any]
    checkpoint: GeneralizationCheckpoint
    resume: bool
    force: bool
    max_workers: int
    seed: int

    def part_path(self, stage: str, instance_id: str, suffix: str = ".json") -> Path:
        return self.output_dir / "checkpoint_parts" / stage / f"{instance_id}{suffix}"


def make_context(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    resume: bool = False,
    force: bool = False,
    max_workers: int = 1,
    seed: int | None = None,
) -> GeneralizationContext:
    root = Path.cwd().resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    config = load_generalization_configuration(config_path)
    frozen_seed = int(config["random_objective"]["base_seed"])
    selected_seed = frozen_seed if seed is None else int(seed)
    if selected_seed != frozen_seed:
        raise ValueError(
            f"--seed is frozen at {frozen_seed}; received {selected_seed}. "
            "Create a new predeclared study configuration for another seed."
        )
    return GeneralizationContext(
        root=root,
        output_dir=destination,
        config_path=Path(config_path),
        config=config,
        checkpoint=GeneralizationCheckpoint(destination),
        resume=bool(resume),
        force=bool(force),
        max_workers=max(1, int(max_workers)),
        seed=selected_seed,
    )


def _load_prior_study_configuration(root: Path) -> dict[str, Any]:
    path = root / PRIOR_STUDY_DIR / "study_configuration.json"
    if not path.is_file():
        raise FileNotFoundError(f"completed development configuration is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_frozen_selector_configuration(
    benchmark_config: Mapping[str, Any], prior: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy and cross-check every frozen algorithmic field from the prior study."""

    expected = {
        "development_matrix_fingerprint": prior["matrix_fingerprint"],
        "development_physical_alpha": prior["physical_alpha"],
        "development_physical_alpha_policy": prior["physical_alpha_policy"],
        "physical_alpha_policy": benchmark_config["regularization"]["policy"],
        "lambda_ref": benchmark_config["regularization"]["lambda_ref"],
        "sensitivity_score_definition": benchmark_config["sensitivity_score_definition"],
        "task_score_normalization": benchmark_config["task_score_normalization"],
        "normalization_epsilon": benchmark_config["score_normalization_epsilon"],
        "normalized_error_floor": benchmark_config["normalized_error_floor"],
        "normalized_error_failure_threshold": benchmark_config[
            "normalized_error_failure_threshold"
        ],
        "mean_objective": benchmark_config["mean_objective"],
        "worst_case_objective": benchmark_config["worst_case_objective"],
        "support_coverage_policy": benchmark_config["support_coverage_policy"],
        "milp_solver_options": benchmark_config["milp_solver_options"],
        "refinement_max_iterations": benchmark_config["refinement"]["max_iterations"],
        "refinement_improvement_tolerance": benchmark_config["refinement"][
            "strict_improvement_tolerance"
        ],
        "refinement_tie_breaking": benchmark_config["refinement"]["tie_breaking"],
        "refinement_candidate_order": benchmark_config["refinement"]["candidate_order"],
        "random_support_count": benchmark_config["random_objective"][
            "support_count_per_budget"
        ],
        "random_support_seeds": benchmark_config["random_objective"],
        "certificate_formula_version": benchmark_config["certificate"]["formula_version"],
        "certificate_validation": benchmark_config["certificate"],
        "support_budgets": benchmark_config["support_budgets"],
        "slot_budgets": benchmark_config["slot_budgets"],
        "primary_comparison_budget": benchmark_config["primary_comparison"],
        "secondary_comparison_budget": benchmark_config["secondary_comparison"],
        "qsvt_uniform_error_tolerance": benchmark_config["qsvt"][
            "uniform_approximation_tolerance"
        ],
        "qsvt_candidate_degrees": benchmark_config["qsvt"]["candidate_degrees"],
        "qsvt_phase_refit_policy": benchmark_config["qsvt"]["phase_refit_policy"],
        "selected_output_evaluation_metrics": [
            "absolute_error",
            "normalized_error",
            "median_absolute_error",
            "mean_absolute_error",
            "p90_absolute_error",
            "worst_absolute_error",
            "median_normalized_error",
            "mean_normalized_error",
            "failure_fraction",
        ],
        "source_study_id": prior["study_id"],
        "source_study_git_commit": prior["git_commit"],
        "immutable_after_benchmark_evaluation_begins": True,
    }
    cross_checks = {
        "normalization_epsilon": prior["score_normalization_epsilon"],
        "normalized_error_floor": prior["y_floor"],
        "normalized_error_failure_threshold": prior["normalized_error_failure_threshold"],
        "support_budgets": prior["support_budgets"],
        "slot_budgets": prior["slot_budgets"],
        "support_coverage_policy": prior["coverage_policy"],
        "milp_solver_options": prior["milp"],
        "refinement_max_iterations": prior["refinement"]["max_iterations"],
        "refinement_improvement_tolerance": prior["refinement"][
            "strict_improvement_tolerance"
        ],
        "random_support_count": prior["random_supports_per_budget"],
        "qsvt_uniform_error_tolerance": prior["common_qsvt"][
            "uniform_approximation_tolerance"
        ],
        "qsvt_candidate_degrees": prior["common_qsvt"]["candidate_degrees"],
    }
    for key, prior_value in cross_checks.items():
        if expected[key] != prior_value:
            raise ValueError(f"frozen configuration mismatch for {key}")
    expected["configuration_fingerprint"] = configuration_fingerprint(expected)
    return expected


def stage_audit(context: GeneralizationContext) -> dict[str, Any]:
    audit = context.output_dir / "implementation_audit.md"
    if not audit.is_file():
        raise FileNotFoundError("required implementation audit must exist before the campaign")
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


def stage_freeze(context: GeneralizationContext) -> dict[str, Any]:
    prior = _load_prior_study_configuration(context.root)
    frozen = build_frozen_selector_configuration(context.config, prior)
    atomic_write_json(context.output_dir / "frozen_selector_configuration.json", frozen)
    study = {
        **context.config,
        "root": str(context.root),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=context.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_commit": git_commit_hash(),
        "max_workers_recorded": context.max_workers,
        "frozen_configuration_fingerprint": frozen["configuration_fingerprint"],
        "benchmark_configuration_fingerprint": configuration_fingerprint(context.config),
        "benchmark_evaluation_started": False,
    }
    atomic_write_json(context.output_dir / "study_configuration.json", study)
    return {
        "configuration_fingerprint": frozen["configuration_fingerprint"],
        "development_matrix_fingerprint": frozen["development_matrix_fingerprint"],
    }


def _selected_measurement_records(
    metadata: Mapping[str, Any], rows: Sequence[int]
) -> list[dict[str, Any]]:
    labels = list(metadata.get("measurement_labels", []))
    types = list(metadata.get("measurement_types", []))
    return [
        {
            "local_index": local,
            "full_measurement_index": int(row),
            "measurement_label": labels[int(row)] if int(row) < len(labels) else f"row_{row}",
            "measurement_type": types[int(row)] if int(row) < len(types) else "unknown",
        }
        for local, row in enumerate(rows)
    ]


def _structural_feasibility(
    matrix: np.ndarray, context: GeneralizationContext, *, k_budget: int, slot_budget: int
) -> SupportSelectionResult:
    settings = context.config["milp_solver_options"]
    return select_resource_constrained_support(
        matrix,
        np.ones_like(matrix, dtype=np.float64),
        SupportConstraints(int(k_budget), int(slot_budget), True),
        time_limit_seconds=float(settings["time_limit_seconds"]),
        relative_mip_gap=float(settings["relative_mip_gap"]),
        tie_epsilon_relative=float(settings["deterministic_tie_epsilon_relative"]),
    )


def stage_instances(context: GeneralizationContext) -> dict[str, Any]:
    extraction = context.config["instance_extraction"]
    regularization = context.config["regularization"]
    primary = context.config["primary_comparison"]
    secondary = context.config["secondary_comparison"]
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for case_name in context.config["required_cases"]:
        for matrix_seed in context.config["instance_seeds"][case_name]:
            instance_id = deterministic_instance_id(case_name, int(matrix_seed))
            try:
                full = extract_weighted_jacobian_matrix(
                    case_name=case_name,
                    mode=str(extraction["mode"]),
                    case_source=str(extraction["case_source"]),
                    measurement_profile=str(extraction["measurement_profile"]),
                    normalize=False,
                    seed=int(matrix_seed),
                )
                matrix, selected_rows, selected_columns = extract_coverage_preserving_block(
                    full.matrix,
                    row_count=int(extraction["target_shape"][0]),
                    column_count=int(extraction["target_shape"][1]),
                    max_iterations=int(extraction["strategy_iterations"]),
                )
                matrix_fingerprint = stable_array_fingerprint(matrix)
                candidate_nonzeros = int(np.count_nonzero(matrix))
                row_degrees = np.count_nonzero(matrix, axis=1)
                column_degrees = np.count_nonzero(matrix, axis=0)
                reasons: list[str] = []
                if matrix.shape != tuple(extraction["target_shape"]):
                    reasons.append("instance_extraction_failure: incorrect block shape")
                if not np.all(np.isfinite(matrix)):
                    reasons.append("instance_extraction_failure: nonfinite matrix entry")
                if candidate_nonzeros < int(extraction["minimum_candidate_nonzeros"]):
                    reasons.append("insufficient_nonzeros")
                if np.any(row_degrees == 0) or np.any(column_degrees == 0):
                    reasons.append("instance_extraction_failure: inactive selected row or column")
                if matrix_fingerprint == context.config["development_matrix_fingerprint"]:
                    reasons.append("development_matrix_excluded")

                primary_feasibility = _structural_feasibility(
                    matrix,
                    context,
                    k_budget=int(primary["k_budget"]),
                    slot_budget=int(primary["slot_budget"]),
                )
                secondary_feasibility = _structural_feasibility(
                    matrix,
                    context,
                    k_budget=int(secondary["k_budget"]),
                    slot_budget=int(secondary["slot_budget"]),
                )
                if extraction["require_primary_and_secondary_budget_feasibility"]:
                    if primary_feasibility.status != "completed":
                        reasons.append("support_budget_infeasible: primary comparison")
                    if secondary_feasibility.status != "completed":
                        reasons.append("support_budget_infeasible: secondary comparison")

                reference_mu = float(np.max(np.abs(matrix)))
                reference_beta = int(regularization["reference_slot_count"]) * reference_mu
                alpha = float(regularization["lambda_ref"]) * reference_beta**2
                if not np.isfinite(alpha) or alpha <= 0.0:
                    reasons.append("instance_extraction_failure: invalid regularization alpha")
                try:
                    np.linalg.solve(
                        matrix.T @ matrix + alpha * np.eye(matrix.shape[1]),
                        matrix.T @ np.ones(matrix.shape[0]),
                    )
                except np.linalg.LinAlgError as exc:
                    reasons.append(f"instance_extraction_failure: Ridge solve: {exc}")

                state_records = _state_records(full.metadata, selected_columns)
                measurement_records = _selected_measurement_records(full.metadata, selected_rows)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                condition_number = float(np.linalg.cond(matrix))
                included = not reasons
                payload = {
                    "instance_id": instance_id,
                    "ieee_case": case_name,
                    "matrix_selection_seed": int(matrix_seed),
                    "matrix": matrix,
                    "matrix_shape": list(matrix.shape),
                    "candidate_nonzeros": candidate_nonzeros,
                    "matrix_fingerprint": matrix_fingerprint,
                    "selected_rows": selected_rows,
                    "selected_columns": selected_columns,
                    "state_metadata": state_records,
                    "measurement_metadata": measurement_records,
                    "condition_number": condition_number,
                    "spectral_norm": float(singular_values[0]),
                    "regularization_alpha": alpha,
                    "regularization_policy": regularization["policy"],
                    "reference_mu": reference_mu,
                    "reference_beta": reference_beta,
                    "lambda_ref": regularization["lambda_ref"],
                    "development_or_evaluation": "evaluation",
                    "inclusion_status": "included" if included else "excluded",
                    "inclusion_reasons": reasons,
                    "primary_structural_feasibility": primary_feasibility.status,
                    "secondary_structural_feasibility": secondary_feasibility.status,
                    "extraction_policy": extraction,
                    "selector_outcomes_used_for_inclusion": False,
                }
                destination = context.output_dir / "instances" / f"{instance_id}.json"
                atomic_write_json(destination, payload)
                common = {
                    key: _json_ready(payload[key])
                    for key in (
                        "instance_id",
                        "ieee_case",
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
                for key in (
                    "matrix_shape",
                    "selected_rows",
                    "selected_columns",
                    "state_metadata",
                    "measurement_metadata",
                ):
                    common[key] = json.dumps(common[key], separators=(",", ":"))
                if included:
                    records.append(common)
                else:
                    exclusions.append(
                        {
                            "instance_id": instance_id,
                            "ieee_case": case_name,
                            "matrix_selection_seed": int(matrix_seed),
                            "stage": "instances",
                            "status": "excluded",
                            "failure_reason": "; ".join(reasons),
                            "last_completed_checkpoint": "freeze",
                            "matrix_fingerprint": matrix_fingerprint,
                        }
                    )
            except Exception as exc:
                exclusions.append(
                    {
                        "instance_id": instance_id,
                        "ieee_case": case_name,
                        "matrix_selection_seed": int(matrix_seed),
                        "stage": "instances",
                        "status": "excluded",
                        "failure_reason": (
                            f"instance_extraction_failure: {type(exc).__name__}: {exc}"
                        ),
                        "last_completed_checkpoint": "freeze",
                        "matrix_fingerprint": "",
                    }
                )
    registry = (
        pd.DataFrame(records)
        .sort_values("instance_id", kind="stable")
        .reset_index(drop=True)
    )
    exclusion_columns = [
        "instance_id",
        "ieee_case",
        "matrix_selection_seed",
        "stage",
        "status",
        "failure_reason",
        "last_completed_checkpoint",
        "matrix_fingerprint",
    ]
    exclusion_frame = pd.DataFrame(exclusions, columns=exclusion_columns)
    atomic_write_csv(context.output_dir / "instance_registry.csv", registry)
    atomic_write_csv(context.output_dir / "instance_exclusion_registry.csv", exclusion_frame)
    study_path = context.output_dir / "study_configuration.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["benchmark_evaluation_started"] = True
    study["instance_registry_fingerprint"] = stable_array_fingerprint(
        np.frombuffer(
            "\n".join(registry["matrix_fingerprint"].astype(str)).encode("utf-8"),
            dtype=np.uint8,
        )
    )
    atomic_write_json(study_path, study)
    return {
        "candidates": sum(len(values) for values in context.config["instance_seeds"].values()),
        "included": len(registry),
        "excluded": len(exclusion_frame),
        "cases": sorted(registry["ieee_case"].unique().tolist()),
    }


def _load_instance_payload(context: GeneralizationContext, instance_id: str) -> dict[str, Any]:
    return json.loads(
        (context.output_dir / "instances" / f"{instance_id}.json").read_text(
            encoding="utf-8"
        )
    )


def _included_instances(context: GeneralizationContext) -> pd.DataFrame:
    frame = pd.read_csv(context.output_dir / "instance_registry.csv")
    return frame.sort_values("instance_id", kind="stable").reset_index(drop=True)


def stage_functionals(context: GeneralizationContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for instance in _included_instances(context).itertuples(index=False):
        payload = _load_instance_payload(context, str(instance.instance_id))
        functionals = build_instance_functionals(payload["state_metadata"], dimension=8)
        for functional_id in FUNCTIONAL_IDS:
            record = functionals[functional_id]
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "ieee_case": instance.ieee_case,
                    "functional_id": functional_id,
                    "functional_family": record["functional_family"],
                    "functional_vector": json.dumps(
                        record["vector"].tolist(), separators=(",", ":")
                    ),
                    "functional_norm": float(np.linalg.norm(record["vector"])),
                    "local_indices": json.dumps(record["local_indices"], separators=(",", ":")),
                    "state_metadata": json.dumps(
                        record["state_records"], separators=(",", ":"), sort_keys=True
                    ),
                    "semantic_status": record["semantic_status"],
                    "selection_data_used": "state_metadata_only_no_output_accuracy",
                    "status": "completed",
                    "failure_reason": "",
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["instance_id", "functional_id"], kind="stable"
    )
    atomic_write_csv(context.output_dir / "functional_registry.csv", frame)
    return {
        "functional_rows": len(frame),
        "metadata_grounded": int(frame["semantic_status"].str.startswith("metadata").sum()),
        "nonsemantic": int((frame["semantic_status"] == "nonsemantic_deterministic_probe").sum()),
    }


def _instance_functionals(
    context: GeneralizationContext, instance_id: str
) -> dict[str, np.ndarray]:
    registry = pd.read_csv(context.output_dir / "functional_registry.csv")
    selected = registry[registry["instance_id"] == instance_id]
    if set(selected["functional_id"]) != set(FUNCTIONAL_IDS):
        raise RuntimeError(f"functional registry is incomplete for {instance_id}")
    return {
        str(row.functional_id): np.asarray(json.loads(row.functional_vector), dtype=np.float64)
        for row in selected.itertuples(index=False)
    }


def stage_residuals(context: GeneralizationContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    instances = _included_instances(context)
    for case_name in context.config["required_cases"]:
        case_instances = instances[instances["ieee_case"] == case_name].sort_values(
            "instance_id", kind="stable"
        )
        for position, instance in enumerate(case_instances.itertuples(index=False)):
            instance_id = str(instance.instance_id)
            payload = _load_instance_payload(context, instance_id)
            selected_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
            split_payload: dict[str, Any] = {
                "instance_id": instance_id,
                "ieee_case": case_name,
                "matrix_fingerprint": payload["matrix_fingerprint"],
                "declared_before_selector_evaluation": True,
                "residual_generation_config": context.config["residuals"],
                "training_seed_ids": residual_seed_ids(
                    context.config,
                    case_name=case_name,
                    instance_position=position,
                    split="training",
                ),
                "held_out_seed_ids": residual_seed_ids(
                    context.config,
                    case_name=case_name,
                    instance_position=position,
                    split="held_out",
                ),
                "residual_fingerprints": {},
                "failures": [],
            }
            if set(split_payload["training_seed_ids"]) & set(split_payload["held_out_seed_ids"]):
                raise RuntimeError(f"data leakage: residual seed overlap for {instance_id}")
            for split, seed_ids in (
                ("training", split_payload["training_seed_ids"]),
                ("held_out", split_payload["held_out_seed_ids"]),
            ):
                for seed_id in seed_ids:
                    destination = (
                        context.output_dir
                        / "residual_banks"
                        / instance_id
                        / f"{split}_seed_{int(seed_id)}.npy"
                    )
                    try:
                        system, source = build_engineering_system(
                            {
                                "case_name": case_name,
                                "case_source": "pypower",
                                "matrix_source": "weighted_jacobian",
                                "seed": int(seed_id),
                            }
                        )
                        residual = np.asarray(system.r_tilde, dtype=np.float64)[selected_rows]
                        if residual.shape != (8,) or not np.all(np.isfinite(residual)):
                            raise RuntimeError("controlled residual has invalid shape or values")
                        if float(np.linalg.norm(residual)) <= 1.0e-15:
                            raise RuntimeError("controlled residual is numerically zero")
                        _atomic_save_array(destination, residual)
                        fingerprint = stable_array_fingerprint(residual)
                        split_payload["residual_fingerprints"][str(seed_id)] = fingerprint
                        rows.append(
                            {
                                "instance_id": instance_id,
                                "ieee_case": case_name,
                                "residual_seed": int(seed_id),
                                "split": split,
                                "residual_fingerprint": fingerprint,
                                "residual_norm": float(np.linalg.norm(residual)),
                                "residual_file": destination.relative_to(
                                    context.output_dir
                                ).as_posix(),
                                "generation_source": source,
                                "generation_status": "completed",
                                "failure_reason": "",
                            }
                        )
                    except Exception as exc:
                        reason = f"residual_generation_failure: {type(exc).__name__}: {exc}"
                        split_payload["failures"].append(
                            {"seed": int(seed_id), "split": split, "failure_reason": reason}
                        )
                        rows.append(
                            {
                                "instance_id": instance_id,
                                "ieee_case": case_name,
                                "residual_seed": int(seed_id),
                                "split": split,
                                "residual_fingerprint": "",
                                "residual_norm": np.nan,
                                "residual_file": "",
                                "generation_source": "",
                                "generation_status": "failed",
                                "failure_reason": reason,
                            }
                        )
            atomic_write_json(
                context.output_dir / "residual_splits" / f"{instance_id}.json",
                split_payload,
            )
    registry = pd.DataFrame(rows).sort_values(
        ["instance_id", "split", "residual_seed"], kind="stable"
    )
    atomic_write_csv(context.output_dir / "residual_registry.csv", registry)
    failures = int((registry["generation_status"] != "completed").sum())
    completed = registry[registry["generation_status"] == "completed"]
    counts = completed.groupby(["instance_id", "split"]).size()
    if failures or (counts < 10).any():
        raise RuntimeError(
            f"residual_generation_failure: {failures} failures or insufficient split count"
        )
    return {
        "records": len(registry),
        "completed": len(completed),
        "failed": failures,
        "training": int((completed["split"] == "training").sum()),
        "held_out": int((completed["split"] == "held_out").sum()),
    }


def _load_residual_tasks(
    context: GeneralizationContext,
    instance_id: str,
    *,
    split: str,
    positions: Sequence[int] | None = None,
) -> list[RidgeTask]:
    if split not in {"training", "held_out"}:
        raise ValueError("task split must be training or held_out")
    registry = pd.read_csv(context.output_dir / "residual_registry.csv")
    selected = registry[
        (registry["instance_id"] == instance_id)
        & (registry["split"] == split)
        & (registry["generation_status"] == "completed")
    ].sort_values("residual_seed", kind="stable")
    if positions is not None:
        selected = selected.iloc[list(positions)]
    functionals = _instance_functionals(context, instance_id)
    tasks: list[RidgeTask] = []
    for row in selected.itertuples(index=False):
        residual = np.asarray(
            np.load(context.output_dir / str(row.residual_file)), dtype=np.float64
        )
        for functional_id in FUNCTIONAL_IDS:
            tasks.append(
                RidgeTask(
                    task_id=f"{instance_id}_{split}_seed{int(row.residual_seed)}_{functional_id}",
                    seed_id=int(row.residual_seed),
                    split=split,
                    residual=residual,
                    functional_id=functional_id,
                    functional=functionals[functional_id],
                )
            )
    if not tasks:
        raise RuntimeError(f"no completed {split} tasks for {instance_id}")
    return tasks


def _support_identifier(
    instance_id: str,
    selector: str,
    k_budget: int,
    slot_budget: int,
    support: np.ndarray | None,
    *,
    random_replicate: int | None = None,
) -> str:
    if support is None:
        digest = "failed"
    else:
        digest = stable_array_fingerprint(np.asarray(support, dtype=np.float64))[:12]
    replicate = "" if random_replicate is None else f"_r{int(random_replicate):02d}"
    return (
        f"{instance_id}__{selector}_k{int(k_budget)}_s{int(slot_budget)}"
        f"{replicate}_{digest}"
    )


def _write_support_payload(
    context: GeneralizationContext,
    *,
    instance_id: str,
    selector: str,
    support_id: str,
    support: np.ndarray,
    matrix: np.ndarray,
    record: Mapping[str, Any],
    subdirectory: str = "",
) -> str:
    destination = context.output_dir / "supports"
    if subdirectory:
        destination = destination / subdirectory
    destination = destination / instance_id / f"{support_id}.json"
    sparse_matrix = np.where(support, matrix, 0.0)
    atomic_write_json(
        destination,
        {
            "instance_id": instance_id,
            "support_id": support_id,
            "selector": selector,
            "support": np.asarray(support, dtype=bool).astype(int).tolist(),
            "selected_entries": [
                {"row": int(row), "column": int(column), "value": float(matrix[row, column])}
                for row, column in np.argwhere(support)
            ],
            "support_fingerprint": stable_array_fingerprint(
                np.asarray(support, dtype=np.float64)
            ),
            "sparse_matrix_fingerprint": stable_array_fingerprint(sparse_matrix),
            "selection_record": _json_ready(dict(record)),
        },
    )
    return destination.relative_to(context.output_dir).as_posix()


def load_generalization_support(
    context: GeneralizationContext, support_file: str
) -> np.ndarray:
    payload = json.loads((context.output_dir / support_file).read_text(encoding="utf-8"))
    support = np.asarray(payload["support"], dtype=bool)
    if support.shape != (8, 8):
        raise ValueError("stored support does not have shape 8x8")
    fingerprint = stable_array_fingerprint(support.astype(np.float64))
    if fingerprint != payload["support_fingerprint"]:
        raise RuntimeError("stored support fingerprint mismatch")
    return support


def _support_selection_record(
    context: GeneralizationContext,
    *,
    instance_id: str,
    ieee_case: str,
    matrix: np.ndarray,
    selector: str,
    constraints: SupportConstraints,
    result: SupportSelectionResult,
    selection_data_split: str,
    random_seed: int | None = None,
    random_replicate: int | None = None,
    refinement_iterations: int = 0,
    refinement_initial_support_id: str = "",
    refinement_objective: str = "",
) -> dict[str, Any]:
    support_id = _support_identifier(
        instance_id,
        selector,
        constraints.k_budget,
        constraints.slot_budget,
        result.support,
        random_replicate=random_replicate,
    )
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "ieee_case": ieee_case,
        "support_id": support_id,
        "selector": selector,
        "k_budget": int(constraints.k_budget),
        "slot_budget": int(constraints.slot_budget),
        "actual_nonzeros": np.nan,
        "actual_row_degree": np.nan,
        "actual_column_degree": np.nan,
        "actual_max_row_degree": np.nan,
        "actual_max_column_degree": np.nan,
        "slot_count": np.nan,
        "support_fingerprint": "",
        "sparse_matrix_fingerprint": "",
        "support_file": "",
        "training_objective": np.nan,
        "training_objective_name": "",
        "training_mean_normalized_error": np.nan,
        "training_worst_normalized_error": np.nan,
        "training_median_absolute_error": np.nan,
        "selection_data_split": selection_data_split,
        "solver_used": result.solver_used,
        "solver_status": result.solver_status,
        "solver_gap": result.optimality_gap,
        "selection_runtime": float(result.runtime_seconds),
        "selection_objective_value": result.objective_value,
        "fallback_used": bool(result.fallback_used),
        "random_seed": random_seed,
        "random_replicate": random_replicate,
        "refinement_iterations": int(refinement_iterations),
        "refinement_initial_support_id": refinement_initial_support_id,
        "refinement_objective": refinement_objective,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "stage": "supports" if not selector.startswith("sensitivity_refined") else "refinement",
        "last_completed_checkpoint": "residuals",
    }
    if result.support is None:
        if not record["failure_reason"]:
            record["failure_reason"] = "support_budget_infeasible"
        return record
    report = support_constraint_report(matrix, result.support, constraints)
    if not report["valid"]:
        raise RuntimeError(
            f"selector returned an invalid support: {report['failure_reasons']}"
        )
    sparse_matrix = np.where(result.support, matrix, 0.0)
    pattern = sparse_matrix.T != 0.0
    slot_count = minimum_slot_count(pattern)
    assignment = assign_slot_permutations(pattern, slots=slot_count)
    slot_validation = validate_slot_assignment(pattern, assignment)
    if not slot_validation["valid"]:
        raise RuntimeError("slot_assignment_failure: selected support decomposition invalid")
    record.update(
        {
            "actual_nonzeros": int(report["actual_nonzeros"]),
            "actual_row_degree": int(report["actual_max_row_degree"]),
            "actual_column_degree": int(report["actual_max_column_degree"]),
            "actual_max_row_degree": int(report["actual_max_row_degree"]),
            "actual_max_column_degree": int(report["actual_max_column_degree"]),
            "slot_count": int(slot_count),
            "support_fingerprint": stable_array_fingerprint(
                result.support.astype(np.float64)
            ),
            "sparse_matrix_fingerprint": stable_array_fingerprint(sparse_matrix),
            "last_completed_checkpoint": record["stage"],
        }
    )
    record["support_file"] = _write_support_payload(
        context,
        instance_id=instance_id,
        selector=selector,
        support_id=support_id,
        support=result.support,
        matrix=matrix,
        record=record,
    )
    return record


def _training_metrics_for_support(
    matrix: np.ndarray,
    support: np.ndarray,
    tasks: Sequence[RidgeTask],
    *,
    alpha: float,
    y_floor: float,
) -> dict[str, float]:
    sparse = np.where(support, matrix, 0.0)
    full, selected, normalized = exact_task_errors(
        matrix, sparse, tasks, alpha=float(alpha), y_floor=float(y_floor)
    )
    absolute = np.abs(selected - full)
    return {
        "training_mean_normalized_error": float(np.mean(normalized)),
        "training_worst_normalized_error": float(np.max(normalized)),
        "training_median_absolute_error": float(np.median(absolute)),
    }


def _annotate_training_metrics(
    context: GeneralizationContext,
    records: list[dict[str, Any]],
    *,
    matrix: np.ndarray,
    tasks: Sequence[RidgeTask],
    alpha: float,
) -> None:
    y_floor = float(context.config["normalized_error_floor"])
    for record in records:
        if record["status"] != "completed":
            continue
        support = load_generalization_support(context, str(record["support_file"]))
        metrics = _training_metrics_for_support(
            matrix, support, tasks, alpha=alpha, y_floor=y_floor
        )
        record.update(metrics)
        selector = str(record["selector"])
        if selector.endswith("worst_case"):
            record["training_objective_name"] = "worst_case_normalized_error"
            record["training_objective"] = metrics["training_worst_normalized_error"]
        else:
            record["training_objective_name"] = "mean_normalized_error"
            record["training_objective"] = metrics["training_mean_normalized_error"]


def _support_part_is_complete(context: GeneralizationContext, instance_id: str) -> bool:
    part = context.part_path("supports", instance_id, ".csv")
    if not part.is_file():
        return False
    frame = pd.read_csv(part)
    completed = frame[frame["status"] == "completed"]
    return all((context.output_dir / path).is_file() for path in completed["support_file"])


def _select_instance_supports(
    context: GeneralizationContext,
    *,
    instance: Any,
    instance_ordinal: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instance_id = str(instance.instance_id)
    payload = _load_instance_payload(context, instance_id)
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    alpha = float(payload["regularization_alpha"])
    tasks = _load_residual_tasks(context, instance_id, split="training")
    if any(task.split != "training" for task in tasks):
        raise RuntimeError("data leakage: held-out task entered support construction")
    scores = compute_output_aware_entry_scores(
        matrix,
        tasks,
        alpha=alpha,
        epsilon=float(context.config["score_normalization_epsilon"]),
    )
    _row_lev, _column_lev, leverage = deterministic_ridge_leverage_scores(
        matrix, alpha=alpha
    )
    magnitude = np.abs(matrix)
    score_rows: list[dict[str, Any]] = []
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            score_rows.append(
                {
                    "instance_id": instance_id,
                    "ieee_case": instance.ieee_case,
                    "row": row,
                    "column": column,
                    "matrix_value": float(matrix[row, column]),
                    "absolute_matrix_value": float(abs(matrix[row, column])),
                    "sensitivity_mean": float(scores.sensitivity_mean[row, column]),
                    "sensitivity_worst_case": float(
                        scores.sensitivity_worst_case[row, column]
                    ),
                    "candidate_status": "candidate" if matrix[row, column] != 0 else "zero",
                    "selection_data_split": "training_only",
                }
            )

    settings = context.config["milp_solver_options"]
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    pair_index = 0
    for raw_k in context.config["support_budgets"]:
        for raw_s in context.config["slot_budgets"]:
            k_budget, slot_budget = int(raw_k), int(raw_s)
            constraints = SupportConstraints(k_budget, slot_budget, True)
            objectives = {
                "balanced_magnitude": magnitude,
                "leverage_weighted": leverage,
                "sensitivity_initial_mean": scores.sensitivity_mean,
                "sensitivity_initial_worst_case": scores.sensitivity_worst_case,
            }
            initial_by_selector: dict[str, dict[str, Any]] = {}
            for selector, objective in objectives.items():
                result = select_resource_constrained_support(
                    matrix,
                    objective,
                    constraints,
                    time_limit_seconds=float(settings["time_limit_seconds"]),
                    relative_mip_gap=float(settings["relative_mip_gap"]),
                    tie_epsilon_relative=float(settings["deterministic_tie_epsilon_relative"]),
                )
                split = "training_only" if selector.startswith("sensitivity") else "matrix_only"
                record = _support_selection_record(
                    context,
                    instance_id=instance_id,
                    ieee_case=str(instance.ieee_case),
                    matrix=matrix,
                    selector=selector,
                    constraints=constraints,
                    result=result,
                    selection_data_split=split,
                )
                records.append(record)
                initial_by_selector[selector] = record

            random_settings = context.config["random_objective"]
            for replicate in range(int(random_settings["support_count_per_budget"])):
                random_seed = (
                    int(random_settings["base_seed"])
                    + int(instance_ordinal) * 1_000_000
                    + pair_index * 10_000
                    + int(replicate)
                )
                rng = np.random.default_rng(random_seed)
                random_scores = np.zeros_like(matrix)
                random_scores[matrix != 0.0] = rng.random(int(np.count_nonzero(matrix)))
                result = select_resource_constrained_support(
                    matrix,
                    random_scores,
                    constraints,
                    time_limit_seconds=float(settings["time_limit_seconds"]),
                    relative_mip_gap=float(settings["relative_mip_gap"]),
                    tie_epsilon_relative=float(settings["deterministic_tie_epsilon_relative"]),
                )
                records.append(
                    _support_selection_record(
                        context,
                        instance_id=instance_id,
                        ieee_case=str(instance.ieee_case),
                        matrix=matrix,
                        selector=RANDOM_SELECTOR,
                        constraints=constraints,
                        result=result,
                        selection_data_split="fixed_seed_random_objective_no_residuals",
                        random_seed=random_seed,
                        random_replicate=replicate,
                    )
                )

            refinement_settings = context.config["refinement"]
            for initial_selector, refined_selector, objective in (
                (
                    "sensitivity_initial_mean",
                    "sensitivity_refined_mean",
                    "mean_normalized_error",
                ),
                (
                    "sensitivity_initial_worst_case",
                    "sensitivity_refined_worst_case",
                    "worst_case_normalized_error",
                ),
            ):
                initial = initial_by_selector[initial_selector]
                if initial["status"] != "completed":
                    failed = SupportSelectionResult(
                        support=None,
                        objective_value=None,
                        solver_used="deterministic_exact_one_swap",
                        solver_status="not_run_missing_initial_support",
                        optimality_gap=None,
                        runtime_seconds=0.0,
                        fallback_used=False,
                        status="failed",
                        failure_reason=(
                            "support_budget_infeasible: initial sensitivity support unavailable"
                        ),
                    )
                    records.append(
                        _support_selection_record(
                            context,
                            instance_id=instance_id,
                            ieee_case=str(instance.ieee_case),
                            matrix=matrix,
                            selector=refined_selector,
                            constraints=constraints,
                            result=failed,
                            selection_data_split="training_only",
                            refinement_objective=objective,
                        )
                    )
                    continue
                initial_support = load_generalization_support(
                    context, str(initial["support_file"])
                )
                started = time.perf_counter()
                refined = refine_support_one_swap(
                    matrix,
                    initial_support,
                    tasks,
                    constraints,
                    alpha=alpha,
                    y_floor=float(context.config["normalized_error_floor"]),
                    objective=objective,
                    max_iterations=int(refinement_settings["max_iterations"]),
                    improvement_tolerance=float(
                        refinement_settings["strict_improvement_tolerance"]
                    ),
                )
                elapsed = time.perf_counter() - started
                result = SupportSelectionResult(
                    support=refined.support,
                    objective_value=refined.final_objective,
                    solver_used="deterministic_exhaustive_exact_loss_one_swap",
                    solver_status="locally_refined_feasible",
                    optimality_gap=None,
                    runtime_seconds=elapsed,
                    fallback_used=False,
                    status="completed",
                    failure_reason="",
                )
                record = _support_selection_record(
                    context,
                    instance_id=instance_id,
                    ieee_case=str(instance.ieee_case),
                    matrix=matrix,
                    selector=refined_selector,
                    constraints=constraints,
                    result=result,
                    selection_data_split="training_only",
                    refinement_iterations=refined.iterations_accepted,
                    refinement_initial_support_id=str(initial["support_id"]),
                    refinement_objective=objective,
                )
                records.append(record)
                for trace in refined.trace:
                    traces.append(
                        {
                            "instance_id": instance_id,
                            "ieee_case": instance.ieee_case,
                            "support_id": record["support_id"],
                            "selector": refined_selector,
                            "k_budget": k_budget,
                            "slot_budget": slot_budget,
                            "initial_support_id": initial["support_id"],
                            "objective": objective,
                            **trace,
                        }
                    )
            pair_index += 1

    _annotate_training_metrics(
        context, records, matrix=matrix, tasks=tasks, alpha=alpha
    )
    registry = pd.DataFrame(records).sort_values(
        ["k_budget", "slot_budget", "selector", "random_replicate", "support_id"],
        kind="stable",
        na_position="first",
    )
    trace_frame = pd.DataFrame(traces)
    score_frame = pd.DataFrame(score_rows)
    return registry, trace_frame, score_frame


def stage_supports(context: GeneralizationContext) -> dict[str, Any]:
    instances = _included_instances(context)
    registry_parts: list[pd.DataFrame] = []
    trace_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    resumed_instances = 0
    for ordinal, instance in enumerate(instances.itertuples(index=False)):
        instance_id = str(instance.instance_id)
        registry_part = context.part_path("supports", instance_id, ".csv")
        trace_part = context.part_path("refinement_traces", instance_id, ".csv")
        score_part = context.part_path("entry_scores", instance_id, ".csv")
        if context.resume and not context.force and _support_part_is_complete(context, instance_id):
            registry = pd.read_csv(registry_part)
            traces = pd.read_csv(trace_part) if trace_part.is_file() else pd.DataFrame()
            scores = pd.read_csv(score_part)
            resumed_instances += 1
        else:
            registry, traces, scores = _select_instance_supports(
                context, instance=instance, instance_ordinal=ordinal
            )
            atomic_write_csv(registry_part, registry)
            atomic_write_csv(trace_part, traces)
            atomic_write_csv(score_part, scores)
        registry_parts.append(registry)
        trace_parts.append(traces)
        score_parts.append(scores)
    registry = pd.concat(registry_parts, ignore_index=True).sort_values(
        ["instance_id", "k_budget", "slot_budget", "selector", "random_replicate", "support_id"],
        kind="stable",
        na_position="first",
    )
    traces = pd.concat(trace_parts, ignore_index=True) if trace_parts else pd.DataFrame()
    scores = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    atomic_write_csv(context.output_dir / "support_registry.csv", registry)
    atomic_write_csv(context.output_dir / "support_selection_results.csv", registry.copy())
    atomic_write_csv(context.output_dir / "refinement_traces.csv", traces)
    atomic_write_csv(context.output_dir / "entry_scores.csv", scores)
    training_columns = [
        "instance_id",
        "ieee_case",
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "actual_nonzeros",
        "slot_count",
        "training_mean_normalized_error",
        "training_worst_normalized_error",
        "training_median_absolute_error",
        "status",
        "failure_reason",
    ]
    atomic_write_csv(
        context.output_dir / "training_instance_summary.csv", registry[training_columns]
    )
    return {
        "support_records": len(registry),
        "completed": int((registry["status"] == "completed").sum()),
        "failed_or_infeasible": int((registry["status"] != "completed").sum()),
        "deterministic": int(registry["selector"].isin(DETERMINISTIC_SELECTORS).sum()),
        "random_objective": int((registry["selector"] == RANDOM_SELECTOR).sum()),
        "refinement_trace_rows": len(traces),
        "resumed_instance_parts": resumed_instances,
    }


def _heldout_instance_rows(
    context: GeneralizationContext,
    *,
    instance: Any,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    instance_id = str(instance.instance_id)
    payload = _load_instance_payload(context, instance_id)
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    alpha = float(payload["regularization_alpha"])
    tasks = _load_residual_tasks(context, instance_id, split="held_out")
    if any(task.split != "held_out" for task in tasks):
        raise RuntimeError("data leakage guard failed for held-out evaluation")
    gradients = np.stack(
        [
            ridge_selected_output_gradient(
                matrix, task.residual, task.functional, alpha
            )
            for task in tasks
        ],
        axis=0,
    )
    rows: list[dict[str, Any]] = []
    certificate_settings = context.config["certificate"]
    y_floor = float(context.config["normalized_error_floor"])
    failure_threshold = float(context.config["normalized_error_failure_threshold"])
    unit_residual = np.zeros(matrix.shape[0], dtype=np.float64)
    unit_residual[0] = 1.0
    unit_functional = np.zeros(matrix.shape[1], dtype=np.float64)
    unit_functional[0] = 1.0
    for record in registry.itertuples(index=False):
        common = {
            "instance_id": instance_id,
            "ieee_case": instance.ieee_case,
            "support_id": record.support_id,
            "selector": record.selector,
            "k_budget": int(record.k_budget),
            "slot_budget": int(record.slot_budget),
            "actual_nonzeros": record.actual_nonzeros,
            "slot_count": record.slot_count,
            "support_fingerprint": record.support_fingerprint,
            "random_replicate": record.random_replicate,
        }
        if record.status != "completed":
            rows.append(
                {
                    **common,
                    "status": "failed",
                    "failure_reason": record.failure_reason,
                    "stage": "heldout",
                    "last_completed_checkpoint": "supports",
                }
            )
            continue
        support = load_generalization_support(context, str(record.support_file))
        sparse_matrix = np.where(support, matrix, 0.0)
        full_outputs, sparse_outputs, normalized = exact_task_errors(
            matrix,
            sparse_matrix,
            tasks,
            alpha=alpha,
            y_floor=y_floor,
        )
        operator_certificate = compute_ridge_selected_output_certificate(
            matrix,
            sparse_matrix,
            unit_residual,
            unit_functional,
            alpha,
        )
        operator_bound = min(
            operator_certificate.operator_bound_forward,
            operator_certificate.operator_bound_reverse,
        )
        delta_matrix = sparse_matrix - matrix
        linearized = np.einsum("tij,ij->t", gradients, delta_matrix)
        for index, task in enumerate(tasks):
            signed_error = float(sparse_outputs[index] - full_outputs[index])
            absolute_error = abs(signed_error)
            certificate_bound = float(
                np.linalg.norm(task.functional)
                * np.linalg.norm(task.residual)
                * operator_bound
            )
            certificate = compute_ridge_selected_output_certificate(
                matrix,
                sparse_matrix,
                task.residual,
                task.functional,
                alpha,
            )
            if not np.isclose(
                certificate.selected_output_bound,
                certificate_bound,
                rtol=1.0e-13,
                atol=1.0e-15,
            ):
                raise RuntimeError("certificate formula cache cross-check failed")
            validated = validate_ridge_selected_output_certificate(
                certificate,
                absolute_error,
                absolute_tolerance=float(
                    certificate_settings["absolute_validation_tolerance"]
                ),
                relative_tolerance=float(
                    certificate_settings["relative_validation_tolerance"]
                ),
            )
            rows.append(
                {
                    **common,
                    "task_id": task.task_id,
                    "residual_seed": int(task.seed_id),
                    "split": "held_out",
                    "functional_id": task.functional_id,
                    "full_ridge_output": float(full_outputs[index]),
                    "sparse_ridge_output": float(sparse_outputs[index]),
                    "signed_error": signed_error,
                    "absolute_error": absolute_error,
                    "normalized_error": float(normalized[index]),
                    "failure_above_frozen_threshold": bool(
                        normalized[index] > failure_threshold
                    ),
                    "first_order_prediction": float(linearized[index]),
                    "first_order_semantics": "local_linear_prediction_not_certificate",
                    "matrix_delta_spectral": validated.matrix_delta_spectral,
                    "operator_bound_forward": validated.operator_bound_forward,
                    "operator_bound_reverse": validated.operator_bound_reverse,
                    "certificate_bound": validated.selected_output_bound,
                    "certificate_holds": validated.certificate_holds,
                    "certificate_tightness": validated.tightness_ratio,
                    "certificate_formula_version": certificate_settings["formula_version"],
                    "certificate_used_actual_error_in_computation": False,
                    "status": "completed",
                    "failure_reason": "",
                    "stage": "heldout",
                    "last_completed_checkpoint": "heldout",
                }
            )
    return pd.DataFrame(rows)


def _summarize_heldout_supports(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    group_columns = [
        "instance_id",
        "ieee_case",
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "actual_nonzeros",
        "slot_count",
        "support_fingerprint",
        "random_replicate",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in completed.groupby(group_columns, sort=True, dropna=False):
        common = dict(zip(group_columns, key, strict=True))
        rows.append(
            {
                **common,
                "heldout_task_count": len(group),
                "median_absolute_error": float(group["absolute_error"].median()),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "p90_absolute_error": float(group["absolute_error"].quantile(0.9)),
                "worst_absolute_error": float(group["absolute_error"].max()),
                "median_normalized_error": float(group["normalized_error"].median()),
                "mean_normalized_error": float(group["normalized_error"].mean()),
                "p90_normalized_error": float(group["normalized_error"].quantile(0.9)),
                "worst_normalized_error": float(group["normalized_error"].max()),
                "failure_fraction": float(
                    group["failure_above_frozen_threshold"].astype(bool).mean()
                ),
                "certificate_coverage": float(group["certificate_holds"].astype(bool).mean()),
                "median_certificate_tightness": float(
                    group["certificate_tightness"].dropna().median()
                ),
                "worst_certificate_tightness": float(
                    group["certificate_tightness"].dropna().max()
                ),
                "status": "completed",
                "failure_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _summarize_heldout_functionals(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    group_columns = [
        "instance_id",
        "ieee_case",
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "functional_id",
    ]
    return (
        completed.groupby(group_columns, sort=True, dropna=False)
        .agg(
            median_absolute_error=("absolute_error", "median"),
            p90_absolute_error=("absolute_error", lambda values: values.quantile(0.9)),
            worst_absolute_error=("absolute_error", "max"),
            median_normalized_error=("normalized_error", "median"),
            mean_normalized_error=("normalized_error", "mean"),
            failure_fraction=("failure_above_frozen_threshold", "mean"),
        )
        .reset_index()
    )


def _summarize_heldout_cases(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    group_columns = ["ieee_case", "selector", "k_budget", "slot_budget"]
    return (
        completed.groupby(group_columns, sort=True)
        .agg(
            instances_evaluated=("instance_id", "nunique"),
            supports_evaluated=("support_id", "nunique"),
            median_absolute_error=("absolute_error", "median"),
            mean_absolute_error=("absolute_error", "mean"),
            p90_absolute_error=("absolute_error", lambda values: values.quantile(0.9)),
            worst_absolute_error=("absolute_error", "max"),
            median_normalized_error=("normalized_error", "median"),
            mean_normalized_error=("normalized_error", "mean"),
            failure_fraction=("failure_above_frozen_threshold", "mean"),
        )
        .reset_index()
    )


def stage_heldout(context: GeneralizationContext) -> dict[str, Any]:
    instances = _included_instances(context)
    support_registry = pd.read_csv(context.output_dir / "support_registry.csv")
    parts: list[pd.DataFrame] = []
    resumed_instances = 0
    for instance in instances.itertuples(index=False):
        instance_id = str(instance.instance_id)
        part = context.part_path("heldout", instance_id, ".csv")
        if context.resume and not context.force and part.is_file():
            frame = pd.read_csv(part)
            resumed_instances += 1
        else:
            selected = support_registry[support_registry["instance_id"] == instance_id]
            frame = _heldout_instance_rows(context, instance=instance, registry=selected)
            atomic_write_csv(part, frame)
        parts.append(frame)
    heldout = pd.concat(parts, ignore_index=True).sort_values(
        ["instance_id", "support_id", "residual_seed", "functional_id"],
        kind="stable",
        na_position="last",
    )
    support_summary = _summarize_heldout_supports(heldout)
    functional_summary = _summarize_heldout_functionals(heldout)
    case_summary = _summarize_heldout_cases(heldout)
    atomic_write_csv(context.output_dir / "heldout_results.csv", heldout)
    atomic_write_csv(context.output_dir / "heldout_instance_summary.csv", support_summary)
    atomic_write_csv(context.output_dir / "heldout_functional_summary.csv", functional_summary)
    atomic_write_csv(context.output_dir / "heldout_case_summary.csv", case_summary)
    violations = int(
        (
            (heldout["status"] == "completed")
            & ~heldout["certificate_holds"].fillna(False).astype(bool)
        ).sum()
    )
    if violations:
        raise RuntimeError(f"certificate_violation: {violations} held-out rows")
    return {
        "heldout_rows": len(heldout),
        "completed_rows": int((heldout["status"] == "completed").sum()),
        "failed_support_rows": int((heldout["status"] != "completed").sum()),
        "support_summaries": len(support_summary),
        "certificate_violations": violations,
        "resumed_instance_parts": resumed_instances,
    }


def _matched_comparison_rows(
    support_summary: pd.DataFrame,
    functional_summary: pd.DataFrame,
    comparison: Mapping[str, Any],
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_selector = str(comparison["candidate_selector"])
    baseline_selector = str(comparison["baseline_selector"])
    k_budget = int(comparison["k_budget"])
    slot_budget = int(comparison["slot_budget"])
    selected = support_summary[
        (support_summary["k_budget"] == k_budget)
        & (support_summary["slot_budget"] == slot_budget)
        & support_summary["selector"].isin([candidate_selector, baseline_selector])
        & (support_summary["status"] == "completed")
    ]
    pivot = selected.pivot(
        index=["instance_id", "ieee_case"],
        columns="selector",
        values="median_normalized_error",
    ).reset_index()
    if candidate_selector not in pivot or baseline_selector not in pivot:
        raise RuntimeError(f"primary comparison is incomplete for {label}")
    pivot = pivot.dropna(subset=[candidate_selector, baseline_selector]).copy()
    tolerance = float(comparison.get("tie_relative_tolerance", 0.01))
    epsilon = float(comparison.get("tie_epsilon", 1.0e-15))
    pivot = pivot.rename(
        columns={
            candidate_selector: "candidate_median_normalized_error",
            baseline_selector: "baseline_median_normalized_error",
        }
    )
    pivot["paired_difference_candidate_minus_baseline"] = (
        pivot["candidate_median_normalized_error"]
        - pivot["baseline_median_normalized_error"]
    )
    pivot["outcome"] = [
        classify_matched_errors(
            candidate,
            baseline,
            relative_tolerance=tolerance,
            epsilon=epsilon,
        )
        for candidate, baseline in zip(
            pivot["candidate_median_normalized_error"],
            pivot["baseline_median_normalized_error"],
            strict=True,
        )
    ]
    pivot["comparison_label"] = label
    pivot["candidate_selector"] = candidate_selector
    pivot["baseline_selector"] = baseline_selector
    pivot["k_budget"] = k_budget
    pivot["slot_budget"] = slot_budget

    selected_functional = functional_summary[
        (functional_summary["k_budget"] == k_budget)
        & (functional_summary["slot_budget"] == slot_budget)
        & functional_summary["selector"].isin([candidate_selector, baseline_selector])
    ]
    functional_pivot = selected_functional.pivot(
        index=["instance_id", "ieee_case", "functional_id"],
        columns="selector",
        values="median_normalized_error",
    ).reset_index()
    functional_pivot = functional_pivot.dropna(
        subset=[candidate_selector, baseline_selector]
    ).copy()
    functional_pivot = functional_pivot.rename(
        columns={
            candidate_selector: "candidate_median_normalized_error",
            baseline_selector: "baseline_median_normalized_error",
        }
    )
    functional_pivot["outcome"] = [
        classify_matched_errors(
            candidate,
            baseline,
            relative_tolerance=tolerance,
            epsilon=epsilon,
        )
        for candidate, baseline in zip(
            functional_pivot["candidate_median_normalized_error"],
            functional_pivot["baseline_median_normalized_error"],
            strict=True,
        )
    ]
    functional_pivot["comparison_label"] = label
    return pivot, functional_pivot


def _outcome_counts(frame: pd.DataFrame, group_column: str | None = None) -> Any:
    def counts(values: pd.DataFrame) -> dict[str, int]:
        observed = values["outcome"].value_counts()
        return {name: int(observed.get(name, 0)) for name in ("win", "tie", "loss")}

    if group_column is None:
        return counts(frame)
    return {
        str(key): counts(group)
        for key, group in frame.groupby(group_column, sort=True)
    }


def stage_primary_test(context: GeneralizationContext) -> dict[str, Any]:
    support_summary = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    functional_summary = pd.read_csv(context.output_dir / "heldout_functional_summary.csv")
    primary_config = context.config["primary_comparison"]
    secondary_config = context.config["secondary_comparison"]
    primary, primary_functional = _matched_comparison_rows(
        support_summary, functional_summary, primary_config, label="primary"
    )
    secondary, secondary_functional = _matched_comparison_rows(
        support_summary,
        functional_summary,
        {**secondary_config, "tie_relative_tolerance": primary_config["tie_relative_tolerance"],
         "tie_epsilon": primary_config["tie_epsilon"]},
        label="secondary",
    )
    differences = primary["paired_difference_candidate_minus_baseline"].to_numpy()
    bootstrap = paired_instance_bootstrap(
        differences,
        samples=int(primary_config["bootstrap_samples"]),
        seed=int(primary_config["bootstrap_seed"]),
    )
    bootstrap_column = "median_paired_difference_sensitivity_minus_magnitude"
    ci_low, ci_high = np.quantile(bootstrap[bootstrap_column], [0.025, 0.975])
    payload = {
        "primary_comparison_declaration": primary_config,
        "secondary_comparison_declaration": secondary_config,
        "development_matrix_in_primary_score": False,
        "primary_instances": len(primary),
        "primary_overall_win_tie_loss": _outcome_counts(primary),
        "primary_case_win_tie_loss": _outcome_counts(primary, "ieee_case"),
        "primary_functional_win_tie_loss": _outcome_counts(
            primary_functional, "functional_id"
        ),
        "primary_median_paired_difference_sensitivity_minus_magnitude": float(
            np.median(differences)
        ),
        "primary_bootstrap_confidence_interval_95": [float(ci_low), float(ci_high)],
        "bootstrap_resampling_unit": "instance",
        "secondary_instances": len(secondary),
        "secondary_overall_win_tie_loss": _outcome_counts(secondary),
        "secondary_case_win_tie_loss": _outcome_counts(secondary, "ieee_case"),
        "secondary_functional_win_tie_loss": _outcome_counts(
            secondary_functional, "functional_id"
        ),
        "secondary_median_paired_difference_refined_minus_magnitude": float(
            np.median(secondary["paired_difference_candidate_minus_baseline"])
        ),
        "metric_substitution_after_results": False,
        "status": "completed",
    }
    atomic_write_json(context.output_dir / "generalization_primary_test.json", payload)
    atomic_write_csv(context.output_dir / "generalization_bootstrap.csv", bootstrap)
    atomic_write_csv(
        context.output_dir / "generalization_matched_pairs.csv",
        pd.concat([primary, secondary], ignore_index=True),
    )
    atomic_write_csv(
        context.output_dir / "generalization_functional_pairs.csv",
        pd.concat([primary_functional, secondary_functional], ignore_index=True),
    )
    return {
        "instances": len(primary),
        **{f"primary_{key}": value for key, value in _outcome_counts(primary).items()},
        "median_paired_difference": float(np.median(differences)),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
    }


def _select_stability_support(
    context: GeneralizationContext,
    *,
    matrix: np.ndarray,
    alpha: float,
    tasks: Sequence[RidgeTask],
    selector: str,
    constraints: SupportConstraints,
) -> tuple[np.ndarray, str, int]:
    if any(task.split != "training" for task in tasks):
        raise RuntimeError("data leakage: stability selection requires training tasks")
    settings = context.config["milp_solver_options"]
    if selector == "balanced_magnitude":
        objective = np.abs(matrix)
    else:
        scores = compute_output_aware_entry_scores(
            matrix,
            tasks,
            alpha=alpha,
            epsilon=float(context.config["score_normalization_epsilon"]),
        )
        objective = scores.sensitivity_mean
    result = select_resource_constrained_support(
        matrix,
        objective,
        constraints,
        time_limit_seconds=float(settings["time_limit_seconds"]),
        relative_mip_gap=float(settings["relative_mip_gap"]),
        tie_epsilon_relative=float(settings["deterministic_tie_epsilon_relative"]),
    )
    if result.status != "completed" or result.support is None:
        raise RuntimeError(f"support_budget_infeasible: stability {selector}")
    support = result.support
    iterations = 0
    if selector == "sensitivity_refined_mean":
        refined = refine_support_one_swap(
            matrix,
            support,
            tasks,
            constraints,
            alpha=alpha,
            y_floor=float(context.config["normalized_error_floor"]),
            objective="mean_normalized_error",
            max_iterations=int(context.config["refinement"]["max_iterations"]),
            improvement_tolerance=float(
                context.config["refinement"]["strict_improvement_tolerance"]
            ),
        )
        support = refined.support
        iterations = refined.iterations_accepted
    return support, result.solver_status, iterations


def stage_stability(context: GeneralizationContext) -> dict[str, Any]:
    stability = context.config["stability"]
    support_registry = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout_summary = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    constraints = SupportConstraints(
        int(stability["k_budget"]), int(stability["slot_budget"]), True
    )
    rows: list[dict[str, Any]] = []
    for instance in _included_instances(context).itertuples(index=False):
        instance_id = str(instance.instance_id)
        payload = _load_instance_payload(context, instance_id)
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        alpha = float(payload["regularization_alpha"])
        for selector in stability["selectors"]:
            matched = support_registry[
                (support_registry["instance_id"] == instance_id)
                & (support_registry["selector"] == selector)
                & (support_registry["k_budget"] == constraints.k_budget)
                & (support_registry["slot_budget"] == constraints.slot_budget)
                & (support_registry["status"] == "completed")
            ]
            if len(matched) != 1:
                raise RuntimeError(
                    f"stability full-training support missing for {instance_id}/{selector}"
                )
            full_record = matched.iloc[0]
            full_support = load_generalization_support(
                context, str(full_record["support_file"])
            )
            heldout = heldout_summary[
                heldout_summary["support_id"] == full_record["support_id"]
            ]
            heldout_error = float(heldout.iloc[0]["median_normalized_error"])
            for subset_name, positions in stability["training_subset_schedules"].items():
                tasks = _load_residual_tasks(
                    context,
                    instance_id,
                    split="training",
                    positions=[int(value) for value in positions],
                )
                subset_support, solver_status, iterations = _select_stability_support(
                    context,
                    matrix=matrix,
                    alpha=alpha,
                    tasks=tasks,
                    selector=str(selector),
                    constraints=constraints,
                )
                subset_id = (
                    f"{instance_id}__stability_{selector}_{subset_name}_"
                    f"{stable_array_fingerprint(subset_support.astype(float))[:12]}"
                )
                support_file = _write_support_payload(
                    context,
                    instance_id=instance_id,
                    selector=str(selector),
                    support_id=subset_id,
                    support=subset_support,
                    matrix=matrix,
                    record={
                        "training_subset": subset_name,
                        "training_positions": positions,
                        "held_out_data_used": False,
                    },
                    subdirectory="stability",
                )
                rows.append(
                    {
                        "instance_id": instance_id,
                        "ieee_case": instance.ieee_case,
                        "selector": selector,
                        "k_budget": constraints.k_budget,
                        "slot_budget": constraints.slot_budget,
                        "training_subset": subset_name,
                        "training_positions": json.dumps(positions, separators=(",", ":")),
                        "training_seed_ids": json.dumps(
                            sorted({int(task.seed_id) for task in tasks}), separators=(",", ":")
                        ),
                        "full_training_support_id": full_record["support_id"],
                        "full_training_support_fingerprint": full_record[
                            "support_fingerprint"
                        ],
                        "subset_support_id": subset_id,
                        "subset_support_fingerprint": stable_array_fingerprint(
                            subset_support.astype(float)
                        ),
                        "subset_support_file": support_file,
                        "jaccard_similarity": jaccard_similarity(
                            full_support, subset_support
                        ),
                        "heldout_median_normalized_error_full_support": heldout_error,
                        "solver_status": solver_status,
                        "refinement_iterations": iterations,
                        "selection_data_split": "training_only",
                        "held_out_used_for_support_construction": False,
                        "status": "completed",
                        "failure_reason": "",
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["instance_id", "selector", "training_subset"], kind="stable"
    )
    summary_rows: list[dict[str, Any]] = []
    from scipy.stats import spearmanr

    for selector, group in frame.groupby("selector", sort=True):
        instability = 1.0 - group["jaccard_similarity"].to_numpy(dtype=np.float64)
        errors = group["heldout_median_normalized_error_full_support"].to_numpy(
            dtype=np.float64
        )
        if np.allclose(instability, instability[0]) or np.allclose(errors, errors[0]):
            correlation = np.nan
        else:
            correlation = float(spearmanr(instability, errors).statistic)
        summary_rows.append(
            {
                "selector": selector,
                "instances": int(group["instance_id"].nunique()),
                "subset_comparisons": len(group),
                "median_jaccard": float(group["jaccard_similarity"].median()),
                "worst_jaccard": float(group["jaccard_similarity"].min()),
                "mean_jaccard": float(group["jaccard_similarity"].mean()),
                "spearman_instability_vs_heldout_error": correlation,
                "relation_semantics": (
                    "descriptive_instance_subset_association_not_causal_or_independent_test"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(context.output_dir / "support_stability.csv", frame)
    atomic_write_csv(context.output_dir / "support_stability_summary.csv", summary)
    return {
        "rows": len(frame),
        "instances": int(frame["instance_id"].nunique()),
        "selectors": sorted(frame["selector"].unique().tolist()),
        "minimum_jaccard": float(frame["jaccard_similarity"].min()),
    }


def stage_certificates(context: GeneralizationContext) -> dict[str, Any]:
    heldout = pd.read_csv(context.output_dir / "heldout_results.csv")
    completed = heldout[heldout["status"] == "completed"].copy()
    columns = [
        "instance_id",
        "ieee_case",
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "residual_seed",
        "functional_id",
        "matrix_delta_spectral",
        "operator_bound_forward",
        "operator_bound_reverse",
        "certificate_bound",
        "absolute_error",
        "certificate_holds",
        "certificate_tightness",
        "certificate_formula_version",
        "certificate_used_actual_error_in_computation",
    ]
    certificate_results = completed[columns].copy()
    violations = int((~certificate_results["certificate_holds"].astype(bool)).sum())
    if violations:
        atomic_write_csv(
            context.output_dir / "certificate_results.csv", certificate_results
        )
        raise RuntimeError(f"certificate_violation: {violations} rows")
    summary_rows: list[dict[str, Any]] = []
    dimensions: list[tuple[str, list[str]]] = [
        ("ieee_case", ["ieee_case"]),
        ("selector", ["selector"]),
        ("support_budget", ["k_budget", "slot_budget"]),
        ("overall", []),
    ]
    for dimension, group_columns in dimensions:
        groups: Iterable[tuple[Any, pd.DataFrame]]
        if group_columns:
            groups = certificate_results.groupby(group_columns, sort=True)
        else:
            groups = [("overall", certificate_results)]
        for key, group in groups:
            if not group_columns:
                key = ()
            elif not isinstance(key, tuple):
                key = (key,)
            row: dict[str, Any] = {
                "summary_dimension": dimension,
                "ieee_case": "",
                "selector": "",
                "k_budget": np.nan,
                "slot_budget": np.nan,
            }
            for name, value in zip(group_columns, key, strict=True):
                row[name] = value
            tightness = group["certificate_tightness"].dropna()
            row.update(
                {
                    "rows": len(group),
                    "supports": int(group["support_id"].nunique()),
                    "coverage": float(group["certificate_holds"].astype(bool).mean()),
                    "violations": int((~group["certificate_holds"].astype(bool)).sum()),
                    "median_tightness": float(tightness.median()),
                    "p90_tightness": float(tightness.quantile(0.9)),
                    "worst_tightness": float(tightness.max()),
                }
            )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(context.output_dir / "certificate_results.csv", certificate_results)
    atomic_write_csv(context.output_dir / "certificate_case_summary.csv", summary)
    return {
        "rows": len(certificate_results),
        "coverage": float(certificate_results["certificate_holds"].astype(bool).mean()),
        "violations": violations,
        "median_tightness": float(
            certificate_results["certificate_tightness"].dropna().median()
        ),
        "worst_tightness": float(
            certificate_results["certificate_tightness"].dropna().max()
        ),
    }


def _resource_instance_rows(
    context: GeneralizationContext,
    *,
    instance: Any,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    from qiskit import transpile

    instance_id = str(instance.instance_id)
    payload = _load_instance_payload(context, instance_id)
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    resource_config = context.config["resources"]
    selected = registry[
        registry["selector"].isin(DETERMINISTIC_SELECTORS)
        | (
            (registry["selector"] == RANDOM_SELECTOR)
            & (registry["random_replicate"] == 0)
            & (registry["k_budget"] == int(context.config["primary_comparison"]["k_budget"]))
            & (
                registry["slot_budget"]
                == int(context.config["primary_comparison"]["slot_budget"])
            )
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for record in selected.itertuples(index=False):
        common = {
            "instance_id": instance_id,
            "ieee_case": instance.ieee_case,
            "support_id": record.support_id,
            "selector": record.selector,
            "k_budget": int(record.k_budget),
            "slot_budget": int(record.slot_budget),
            "support_fingerprint": record.support_fingerprint,
            "sparse_matrix_fingerprint": record.sparse_matrix_fingerprint,
            "random_replicate": record.random_replicate,
            "resource_record_type": "executed_sparse_signal_unitary",
        }
        if record.status != "completed":
            rows.append(
                {
                    **common,
                    "status": "failed",
                    "failure_reason": record.failure_reason,
                    "stage": "resources",
                    "last_completed_checkpoint": "supports",
                }
            )
            continue
        cache_key = str(record.sparse_matrix_fingerprint)
        if cache_key in cache:
            rows.append({**common, **cache[cache_key], "resource_cache_hit": True})
            continue
        try:
            support = load_generalization_support(context, str(record.support_file))
            sparse_matrix = np.where(support, matrix, 0.0)
            pattern = sparse_matrix.T != 0.0
            minimum_slots = minimum_slot_count(pattern)
            wrapper = validate_complete_wrapper(
                _as_quantized_block(sparse_matrix, 53),
                encode_transpose=True,
                transpile_circuit=False,
            )
            slot_validation = validate_slot_assignment(pattern, wrapper.assignment)
            compiled = transpile(
                wrapper.circuit,
                basis_gates=list(resource_config["basis_gates"]),
                optimization_level=int(resource_config["optimization_level"]),
            )
            counts = {str(key): int(value) for key, value in compiled.count_ops().items()}
            if not slot_validation["valid"]:
                raise RuntimeError("slot_assignment_failure: invalid matching decomposition")
            reconstruction_tolerance = float(
                resource_config["wrapper_reconstruction_tolerance"]
            )
            reconstruction_holds = bool(
                wrapper.top_left_reconstruction_error <= reconstruction_tolerance
            )
            validation_reason = (
                ""
                if reconstruction_holds
                else (
                    "other_verified_failure: wrapper numerical reconstruction error "
                    f"{wrapper.top_left_reconstruction_error:.17g} exceeds frozen "
                    f"tolerance {reconstruction_tolerance:.17g}"
                )
            )
            measured = {
                "actual_nonzeros": int(np.count_nonzero(sparse_matrix)),
                "actual_max_row_degree": int(np.count_nonzero(sparse_matrix, axis=1).max()),
                "actual_max_column_degree": int(
                    np.count_nonzero(sparse_matrix, axis=0).max()
                ),
                "minimum_slot_count": int(minimum_slots),
                "slot_count": int(wrapper.slots),
                "normalization_mu": float(np.max(np.abs(sparse_matrix))),
                "native_beta": float(wrapper.normalization_factor),
                "signal_unitary_gate_count": int(sum(counts.values())),
                "signal_unitary_depth": int(compiled.depth()),
                "cx_count": int(counts.get("cx", 0)),
                "controlled_rotations": int(wrapper.slots * sparse_matrix.shape[1]),
                "wrapper_reconstruction_error": float(
                    wrapper.top_left_reconstruction_error
                ),
                "wrapper_reconstruction_tolerance": reconstruction_tolerance,
                "wrapper_reconstruction_holds": reconstruction_holds,
                "wrapper_statevector_error": float(wrapper.statevector_max_error),
                "wrapper_unitarity_error": float(wrapper.unitarity_error),
                "slot_assignment_valid": bool(slot_validation["valid"]),
                "real_edges_covered_exactly_once": bool(
                    slot_validation["real_edges_covered_exactly_once"]
                ),
                "resource_measurement": (
                    "actual_qiskit_transpile_u3_cx_optimization_level_1"
                ),
                "value_semantics": resource_config["value_policy"],
                "missing_cost_is_zero": False,
                "resource_fingerprint": configuration_fingerprint(
                    {
                        "sparse_matrix_fingerprint": cache_key,
                        "slot_count": int(wrapper.slots),
                        "gate_count": int(sum(counts.values())),
                        "depth": int(compiled.depth()),
                        "cx": int(counts.get("cx", 0)),
                    }
                ),
                "status": "completed" if reconstruction_holds else "failed",
                "failure_reason": validation_reason,
                "stage": "resources",
                "last_completed_checkpoint": "resources",
                "resource_cache_hit": False,
            }
            cache[cache_key] = measured
            rows.append({**common, **measured})
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "status": "failed",
                    "failure_reason": (
                        f"resource_compilation_limit: {type(exc).__name__}: {exc}"
                    ),
                    "stage": "resources",
                    "last_completed_checkpoint": "supports",
                    "resource_cache_hit": False,
                }
            )
    return pd.DataFrame(rows)


def stage_resources(context: GeneralizationContext) -> dict[str, Any]:
    instances = _included_instances(context)
    support_registry = pd.read_csv(context.output_dir / "support_registry.csv")
    parts: list[pd.DataFrame] = []
    resumed_instances = 0
    for instance in instances.itertuples(index=False):
        instance_id = str(instance.instance_id)
        part = context.part_path("resources", instance_id, ".csv")
        if context.resume and not context.force and part.is_file():
            frame = pd.read_csv(part)
            resumed_instances += 1
        else:
            selected = support_registry[support_registry["instance_id"] == instance_id]
            frame = _resource_instance_rows(context, instance=instance, registry=selected)
            atomic_write_csv(part, frame)
        parts.append(frame)
    resources = pd.concat(parts, ignore_index=True).sort_values(
        ["instance_id", "k_budget", "slot_budget", "selector", "random_replicate"],
        kind="stable",
        na_position="last",
    )
    deterministic = resources[resources["selector"].isin(DETERMINISTIC_SELECTORS)]
    failures = deterministic[deterministic["status"] != "completed"]
    atomic_write_csv(context.output_dir / "resource_registry.csv", resources)
    completed = resources[resources["status"] == "completed"]
    case_summary = (
        completed.groupby(["ieee_case", "selector"], sort=True)
        .agg(
            instances=("instance_id", "nunique"),
            support_records=("support_id", "count"),
            median_nonzeros=("actual_nonzeros", "median"),
            median_slot_count=("slot_count", "median"),
            median_signal_unitary_gates=("signal_unitary_gate_count", "median"),
            median_signal_unitary_depth=("signal_unitary_depth", "median"),
            median_cx_count=("cx_count", "median"),
            median_controlled_rotations=("controlled_rotations", "median"),
            maximum_wrapper_reconstruction_error=("wrapper_reconstruction_error", "max"),
        )
        .reset_index()
    )
    atomic_write_csv(context.output_dir / "resource_case_summary.csv", case_summary)
    return {
        "resource_records": len(resources),
        "deterministic_records": len(deterministic),
        "unique_measured_wrappers": int(
            completed["resource_fingerprint"].dropna().nunique()
        ),
        "deterministic_failures_retained": len(failures),
        "resumed_instance_parts": resumed_instances,
    }


def _write_pareto_pair(
    context: GeneralizationContext,
    name: str,
    frame: pd.DataFrame,
    *,
    error_column: str,
    cost_column: str,
) -> tuple[int, int]:
    candidates, frontier = grouped_pareto_frontier(
        frame,
        group_columns=["instance_id"],
        error_column=error_column,
        cost_column=cost_column,
        tie_columns=["support_id"],
    )
    atomic_write_csv(
        context.output_dir / f"pareto_candidates_{name}.csv", candidates
    )
    atomic_write_csv(
        context.output_dir / f"pareto_frontier_{name}.csv", frontier
    )
    return len(candidates), len(frontier)


def stage_pareto(context: GeneralizationContext) -> dict[str, Any]:
    heldout = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    completed = heldout[heldout["status"] == "completed"].copy()
    base_columns = [
        "instance_id",
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
    base = completed[base_columns].copy()
    base["accuracy_objective"] = "median_heldout_normalized_error_absolute_not_signed"
    nnz_counts = _write_pareto_pair(
        context,
        "error_nnz",
        base,
        error_column="median_normalized_error",
        cost_column="actual_nonzeros",
    )
    slot_counts = _write_pareto_pair(
        context,
        "error_slots",
        base,
        error_column="median_normalized_error",
        cost_column="slot_count",
    )
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    resource_columns = [
        "support_id",
        "signal_unitary_gate_count",
        "signal_unitary_depth",
        "controlled_rotations",
        "resource_record_type",
        "status",
    ]
    gates = base.merge(resources[resource_columns], on="support_id", how="left")
    gates = gates.rename(columns={"status": "resource_status"})
    gates["missing_resource_cost_is_zero"] = False
    gate_counts = _write_pareto_pair(
        context,
        "error_gates",
        gates,
        error_column="median_normalized_error",
        cost_column="signal_unitary_gate_count",
    )

    deterministic = gates[gates["selector"].isin(DETERMINISTIC_SELECTORS)].copy()
    comparison_columns = [
        "instance_id",
        "ieee_case",
        "selector",
        "k_budget",
        "slot_budget",
        "support_id",
        "median_normalized_error",
        "actual_nonzeros",
        "slot_count",
        "signal_unitary_gate_count",
        "signal_unitary_depth",
        "controlled_rotations",
    ]
    atomic_write_csv(
        context.output_dir / "resource_matched_selector_comparison.csv",
        deterministic[comparison_columns].sort_values(
            ["instance_id", "k_budget", "slot_budget", "selector"], kind="stable"
        ),
    )
    return {
        "error_nnz_candidates": nnz_counts[0],
        "error_nnz_frontier": nnz_counts[1],
        "error_slots_candidates": slot_counts[0],
        "error_slots_frontier": slot_counts[1],
        "error_gates_candidates": gate_counts[0],
        "error_gates_frontier": gate_counts[1],
        "accuracy_uses_signed_error": False,
    }


def _qsvt_support_subset(
    context: GeneralizationContext, instance_id: str, registry: pd.DataFrame
) -> list[dict[str, Any]]:
    qsvt = context.config["qsvt"]
    selected = registry[
        (registry["instance_id"] == instance_id)
        & (registry["k_budget"] == int(qsvt["k_budget"]))
        & (registry["slot_budget"] == int(qsvt["slot_budget"]))
        & (registry["status"] == "completed")
    ]
    subset: list[dict[str, Any]] = []
    for selector in qsvt["selectors"]:
        matched = selected[selected["selector"] == selector]
        if len(matched) != 1:
            raise RuntimeError(
                f"qsvt_statevector_failure: expected one {selector} support for {instance_id}"
            )
        row = matched.iloc[0]
        subset.append(
            {
                "instance_id": instance_id,
                "support_id": str(row["support_id"]),
                "selector": str(selector),
                "support_file": str(row["support_file"]),
                "support_fingerprint": str(row["support_fingerprint"]),
                "actual_slot_count": int(row["slot_count"]),
                "random_replicate": None,
                "selection_policy": "fixed_selector_name_no_performance_ranking",
            }
        )
    random = selected[
        (selected["selector"] == qsvt["random_selector"])
        & (selected["random_replicate"] == int(qsvt["random_replicate"]))
    ]
    if len(random) != 1:
        raise RuntimeError(
            f"qsvt_statevector_failure: expected random replicate zero for {instance_id}"
        )
    row = random.iloc[0]
    subset.append(
        {
            "instance_id": instance_id,
            "support_id": str(row["support_id"]),
            "selector": RANDOM_SELECTOR,
            "support_file": str(row["support_file"]),
            "support_fingerprint": str(row["support_fingerprint"]),
            "actual_slot_count": int(row["slot_count"]),
            "random_replicate": int(qsvt["random_replicate"]),
            "selection_policy": "fixed_random_replicate_no_performance_ranking",
        }
    )
    return subset


def validate_common_design_registry(
    designs: Mapping[str, Any], expected_instance_ids: Sequence[str]
) -> None:
    if set(designs) != set(expected_instance_ids):
        raise ValueError("QSVT common-design registry does not match the predeclared instances")
    for instance_id, design in designs.items():
        subset = design["support_subset"]
        if len(subset) != 4:
            raise ValueError(f"common design for {instance_id} must contain four supports")
        if any(item.get("per_support_phase_refit", False) for item in subset):
            raise ValueError("per-support phase refitting is forbidden")
        if int(design["phase_count"]) != int(design["degree"]) + 1:
            raise ValueError("QSVT phase count must equal degree plus one")
        if not design["degree_selected_before_support_specific_outputs"]:
            raise ValueError("QSVT degree was not selected before support outputs")


def _build_qsvt_designs(
    context: GeneralizationContext, registry: pd.DataFrame
) -> dict[str, Any]:
    from robust_qsvt_se.qsvt.phase_synthesis import (
        synthesize_pennylane_phases_cached,
        validate_qsvt_polynomial,
    )

    qsvt = context.config["qsvt"]
    designs: dict[str, Any] = {}
    for instance_id in qsvt["predeclared_instance_ids"]:
        payload = _load_instance_payload(context, instance_id)
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        subset = _qsvt_support_subset(context, instance_id, registry)
        common_slots = int(qsvt["slot_budget"])
        if any(item["actual_slot_count"] > common_slots for item in subset):
            raise RuntimeError(
                f"common_normalization_failure: support exceeds common slots for {instance_id}"
            )
        common_mu = float(np.max(np.abs(matrix)))
        common_beta = common_slots * common_mu
        alpha = float(payload["regularization_alpha"])
        common_lambda = alpha / common_beta**2
        selected_target: Any | None = None
        degree_trials: list[dict[str, Any]] = []
        for raw_degree in qsvt["candidate_degrees"]:
            degree = int(raw_degree)
            try:
                target = fit_codesigned_bounded_polynomial(
                    beta=common_beta,
                    alpha=alpha,
                    domain_min=float(qsvt["domain_min"]),
                    domain_max=float(qsvt["domain_max"]),
                    degree=degree,
                    margin=float(qsvt["target_margin"]),
                )
                validation = validate_qsvt_polynomial(
                    np.asarray(target.coefficients),
                    parity="odd",
                    bound_tolerance=float(qsvt["bound_tolerance"]),
                )
                accepted = bool(
                    target.fit_max_abs_error
                    <= float(qsvt["uniform_approximation_tolerance"])
                )
                degree_trials.append(
                    {
                        "degree": degree,
                        "fit_max_abs_error": float(target.fit_max_abs_error),
                        "bounded_max_abs": float(target.bounded_max_abs),
                        "accepted": accepted,
                        "validation": validation,
                        "failure_reason": "",
                    }
                )
                if accepted:
                    selected_target = target
                    break
            except Exception as exc:
                degree_trials.append(
                    {
                        "degree": degree,
                        "accepted": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        if selected_target is None:
            raise RuntimeError(
                f"polynomial_fit_failure: no degree met tolerance for {instance_id}"
            )
        coefficients = np.asarray(selected_target.coefficients, dtype=np.float64)
        phase_result = synthesize_pennylane_phases_cached(
            coefficients,
            angle_solver=str(qsvt["phase_solver"]),
            cache_dir=context.output_dir / "qsvt_phase_cache",
            cache_metadata={
                "study_id": STUDY_ID,
                "instance_id": instance_id,
                "beta": common_beta,
                "alpha": alpha,
                "lambda": common_lambda,
                "degree": int(selected_target.degree),
                "track": "per_instance_common_design",
            },
        )
        phases = np.asarray(phase_result.phases, dtype=np.float64)
        if phases.size != int(selected_target.degree) + 1:
            raise RuntimeError("qsvt_statevector_failure: common phase count mismatch")
        for item in subset:
            item["per_support_phase_refit"] = False
        design: dict[str, Any] = {
            "instance_id": instance_id,
            "ieee_case": payload["ieee_case"],
            "matrix_fingerprint": payload["matrix_fingerprint"],
            "support_subset": subset,
            "common_slots": common_slots,
            "common_mu": common_mu,
            "common_beta": common_beta,
            "physical_alpha": alpha,
            "common_lambda": common_lambda,
            "common_C": float(selected_target.bound_C),
            "degree": int(selected_target.degree),
            "phase_count": int(phases.size),
            "phases": phases.tolist(),
            "phase_fingerprint": stable_array_fingerprint(phases),
            "polynomial_coefficients": coefficients.tolist(),
            "polynomial_fingerprint": stable_array_fingerprint(coefficients),
            "fit_max_abs_error": float(selected_target.fit_max_abs_error),
            "bounded_max_abs": float(selected_target.bounded_max_abs),
            "uniform_approximation_tolerance": qsvt["uniform_approximation_tolerance"],
            "degree_trials": degree_trials,
            "degree_selected_before_support_specific_outputs": True,
            "phase_refit_policy": qsvt["phase_refit_policy"],
            "phase_synthesis_metadata": phase_result.metadata,
            "phase_synthesis_cache_hit": phase_result.cache_hit,
            "validation_residual_policy": qsvt["validation_residual_policy"],
            "support_specific_outputs_observed_before_design": False,
        }
        design["common_design_fingerprint"] = configuration_fingerprint(design)
        designs[instance_id] = design
    validate_common_design_registry(designs, qsvt["predeclared_instance_ids"])
    return designs


def _execute_qsvt_design(
    context: GeneralizationContext,
    *,
    design: Mapping[str, Any],
) -> pd.DataFrame:
    from numpy.polynomial import Polynomial
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import (
        build_structured_qsvt_operator_circuit,
    )

    instance_id = str(design["instance_id"])
    payload = _load_instance_payload(context, instance_id)
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    alpha = float(payload["regularization_alpha"])
    tasks = _load_residual_tasks(context, instance_id, split="training")
    residual = np.asarray(tasks[0].residual, dtype=np.float64)
    residual_unit = residual / np.linalg.norm(residual)
    functionals = _instance_functionals(context, instance_id)
    full_update = ridge_svd_solution(matrix, residual, alpha=alpha)
    phases = np.asarray(design["phases"], dtype=np.float64)
    polynomial = Polynomial(np.asarray(design["polynomial_coefficients"], dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for selected in design["support_subset"]:
        common = {
            "instance_id": instance_id,
            "ieee_case": design["ieee_case"],
            "support_id": selected["support_id"],
            "selector": selected["selector"],
            "support_fingerprint": selected["support_fingerprint"],
            "k_budget": int(context.config["qsvt"]["k_budget"]),
            "slot_budget": int(context.config["qsvt"]["slot_budget"]),
            "common_design_fingerprint": design["common_design_fingerprint"],
            "phase_fingerprint": design["phase_fingerprint"],
            "polynomial_fingerprint": design["polynomial_fingerprint"],
            "beta": design["common_beta"],
            "lambda": design["common_lambda"],
            "C": design["common_C"],
            "degree": design["degree"],
            "phase_count": design["phase_count"],
            "per_support_phase_refit": False,
        }
        try:
            support = load_generalization_support(context, str(selected["support_file"]))
            sparse_matrix = np.where(support, matrix, 0.0)
            wrapper = build_common_padded_wrapper(
                sparse_matrix,
                slots=int(design["common_slots"]),
                mu=float(design["common_mu"]),
            )
            if wrapper.reconstruction_error > float(
                context.config["resources"]["wrapper_reconstruction_tolerance"]
            ):
                raise RuntimeError("common padded-wrapper reconstruction failed")
            bundle = build_structured_qsvt_operator_circuit(
                wrapper.unitary, phases, encoded_dimension=sparse_matrix.shape[1]
            )
            initial = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
            initial[: sparse_matrix.shape[0]] = residual_unit
            evolved = Statevector(initial).evolve(bundle.qsvt_operator_circuit).data
            encoded = np.asarray(evolved[: sparse_matrix.shape[1]], dtype=np.complex128)
            postselection = float(np.vdot(encoded, encoded).real)
            normalized_matrix = sparse_matrix.T / float(design["common_beta"])
            left, singular_values, right_t = np.linalg.svd(
                normalized_matrix, full_matrices=False
            )
            exact_action = (
                left @ np.diag(polynomial(singular_values)) @ right_t
            ) @ residual_unit
            action_error = float(
                np.linalg.norm(np.real(encoded) - exact_action)
                / max(np.linalg.norm(exact_action), 1.0e-30)
            )
            if action_error > float(context.config["qsvt"]["qsvt_action_relative_tolerance"]):
                raise RuntimeError(
                    f"QSVT circuit/exact polynomial mismatch {action_error}"
                )
            physical_scale = (
                float(design["common_C"])
                / float(design["common_beta"])
                * float(np.linalg.norm(residual))
            )
            exact_update = physical_scale * exact_action
            qsvt_update = physical_scale * np.real(encoded)
            sparse_update = ridge_svd_solution(sparse_matrix, residual, alpha=alpha)
            compiled_signal = transpile(
                wrapper.circuit,
                basis_gates=list(context.config["resources"]["basis_gates"]),
                optimization_level=int(context.config["resources"]["optimization_level"]),
            )
            signal_counts = compiled_signal.count_ops()
            signal_gates = int(sum(signal_counts.values()))
            signal_calls = int(design["degree"])
            gates_per_attempt = signal_calls * signal_gates + int(design["phase_count"])
            raw_qsvt_counts = bundle.qsvt_operator_circuit.count_ops()
            for functional_id in FUNCTIONAL_IDS:
                functional = functionals[functional_id]
                full_output = float(functional @ full_update)
                sparse_output = float(functional @ sparse_update)
                exact_output = float(functional @ exact_update)
                qsvt_output = float(functional @ qsvt_update)
                rows.append(
                    {
                        **common,
                        "functional_id": functional_id,
                        "validation_residual_seed": int(tasks[0].seed_id),
                        "full_matrix_ridge_output": full_output,
                        "sparse_matrix_ridge_output": sparse_output,
                        "exact_polynomial_svt_output": exact_output,
                        "sparse_qsvt_statevector_output": qsvt_output,
                        "support_selection_error": abs(sparse_output - full_output),
                        "qsvt_error_on_sparse_matrix": abs(qsvt_output - sparse_output),
                        "total_full_to_qsvt_error": abs(qsvt_output - full_output),
                        "postselection_probability": postselection,
                        "expected_attempts_per_postselection": (
                            1.0 / postselection if postselection > 0.0 else np.inf
                        ),
                        "qsvt_action_error_vs_exact_polynomial": action_error,
                        "common_wrapper_reconstruction_error": wrapper.reconstruction_error,
                        "common_signal_unitary_gate_count": signal_gates,
                        "common_signal_unitary_depth": int(compiled_signal.depth()),
                        "common_signal_unitary_cx_count": int(signal_counts.get("cx", 0)),
                        "common_signal_call_count": signal_calls,
                        "component_composed_gates_per_attempt": gates_per_attempt,
                        "component_composed_expected_gates_per_postselection": (
                            gates_per_attempt / postselection
                            if postselection > 0.0
                            else np.inf
                        ),
                        "raw_structured_qsvt_instruction_count": int(
                            sum(raw_qsvt_counts.values())
                        ),
                        "raw_structured_qsvt_depth": int(
                            bundle.qsvt_operator_circuit.depth()
                        ),
                        "resource_semantics": (
                            "actual_transpiled_common_signal_unitary_with_component_composed_"
                            "qsvt_cost_and_raw_structured_statevector_circuit"
                        ),
                        "ridge_and_qsvt_sparse_matrix_fingerprint": stable_array_fingerprint(
                            sparse_matrix
                        ),
                        "ridge_qsvt_identical_sparse_matrix": True,
                        "support_error_separate_from_qsvt_error": True,
                        "execution_mode": "local_statevector_simulation_not_hardware",
                        "status": "completed",
                        "failure_reason": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "status": "failed",
                    "failure_reason": f"qsvt_statevector_failure: {type(exc).__name__}: {exc}",
                    "ridge_qsvt_identical_sparse_matrix": False,
                    "support_error_separate_from_qsvt_error": True,
                }
            )
    return pd.DataFrame(rows)


def stage_qsvt(context: GeneralizationContext) -> dict[str, Any]:
    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    included_ids = set(_included_instances(context)["instance_id"].astype(str))
    expected_ids = list(context.config["qsvt"]["predeclared_instance_ids"])
    missing = set(expected_ids).difference(included_ids)
    if missing:
        raise RuntimeError(
            f"qsvt_statevector_failure: predeclared instances excluded: {sorted(missing)}"
        )
    designs = _build_qsvt_designs(context, registry)
    # The complete design registry is durably declared before any support-specific output.
    atomic_write_json(context.output_dir / "qsvt_instance_designs.json", designs)
    parts: list[pd.DataFrame] = []
    resumed_instances = 0
    for instance_id in expected_ids:
        part = context.part_path("qsvt", instance_id, ".csv")
        if context.resume and not context.force and part.is_file():
            frame = pd.read_csv(part)
            resumed_instances += 1
        else:
            frame = _execute_qsvt_design(context, design=designs[instance_id])
            atomic_write_csv(part, frame)
        parts.append(frame)
    results = pd.concat(parts, ignore_index=True).sort_values(
        ["instance_id", "support_id", "functional_id"],
        kind="stable",
        na_position="last",
    )
    atomic_write_csv(context.output_dir / "qsvt_validation_results.csv", results)
    failures = results[results["status"] != "completed"]
    if not failures.empty:
        raise RuntimeError(f"qsvt_statevector_failure: {len(failures)} result rows")
    cases = sorted(results["ieee_case"].unique().tolist())
    if cases != ["ieee14", "ieee30", "ieee57"]:
        raise RuntimeError("qsvt_statevector_failure: all required IEEE cases not represented")
    return {
        "instances": int(results["instance_id"].nunique()),
        "supports": int(results["support_id"].nunique()),
        "rows": len(results),
        "cases": cases,
        "maximum_qsvt_action_error": float(
            results["qsvt_action_error_vs_exact_polynomial"].max()
        ),
        "maximum_selected_output_qsvt_error": float(
            results["qsvt_error_on_sparse_matrix"].max()
        ),
        "failures": len(failures),
        "resumed_instance_parts": resumed_instances,
    }


def stage_finite_shot(context: GeneralizationContext) -> dict[str, Any]:
    results = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    completed = results[results["status"] == "completed"].drop_duplicates("support_id")
    selected_rows: list[pd.Series] = []
    for case_name in context.config["required_cases"]:
        case = completed[completed["ieee_case"] == case_name]
        first_instance = sorted(case["instance_id"].unique())[0]
        for selector in ("balanced_magnitude", "sensitivity_initial_mean"):
            matched = case[
                (case["instance_id"] == first_instance) & (case["selector"] == selector)
            ]
            if len(matched) != 1:
                raise RuntimeError("finite-shot projection subset is incomplete")
            selected_rows.append(matched.iloc[0])
    finite = context.config["finite_shot"]
    shots_per_seed = int(finite["attempted_shots_per_seed"])
    seed_count = int(finite["seed_count"])
    projected_attempts = len(selected_rows) * shots_per_seed * seed_count
    projected_gate_applications = int(
        sum(
            float(row["component_composed_gates_per_attempt"])
            * shots_per_seed
            * seed_count
            for row in selected_rows
        )
    )
    attempts_ceiling = int(finite["attempted_shot_ceiling"])
    gates_ceiling = int(finite["projected_gate_application_ceiling"])
    execute = (
        projected_attempts <= attempts_ceiling
        and projected_gate_applications <= gates_ceiling
    )
    if execute:
        status = "partially_executed"
        reason = (
            "The predeclared cost ceilings were satisfied, but this benchmark does not "
            "duplicate the independently verified finite-shot estimator; no counts were fabricated."
        )
    else:
        status = "skipped_under_predeclared_cost_ceiling"
        reason = (
            "finite_shot_cost_ceiling_exceeded: projected attempts "
            f"{projected_attempts} (ceiling {attempts_ceiling}); projected component-composed "
            f"gate applications {projected_gate_applications} (ceiling {gates_ceiling})."
        )
    rows = [
        {
            "instance_id": row["instance_id"],
            "ieee_case": row["ieee_case"],
            "support_id": row["support_id"],
            "selector": row["selector"],
            "attempted_shots_per_seed": shots_per_seed,
            "seed_count": seed_count,
            "projected_attempted_shots": shots_per_seed * seed_count,
            "projected_gate_applications": int(
                float(row["component_composed_gates_per_attempt"])
                * shots_per_seed
                * seed_count
            ),
            "actual_attempted_shots": np.nan,
            "actual_postselected_shots": np.nan,
            "actual_readout_accepted_shots": np.nan,
            "status": status,
            "failure_reason": reason,
        }
        for row in selected_rows
    ]
    atomic_write_csv(context.output_dir / "finite_shot_results.csv", pd.DataFrame(rows))
    markdown = "\n".join(
        [
            "# Finite-Shot Status",
            "",
            f"- Status: `{status}`",
            f"- Projected attempted shots: {projected_attempts}",
            f"- Attempted-shot ceiling: {attempts_ceiling}",
            f"- Projected component-composed gate applications: {projected_gate_applications}",
            f"- Gate-application ceiling: {gates_ceiling}",
            f"- Reason: {reason}",
            "- No finite-shot counts were fabricated.",
            "",
        ]
    )
    _atomic_write_text(context.output_dir / "finite_shot_status.md", markdown)
    return {
        "status": status,
        "projected_attempted_shots": projected_attempts,
        "attempted_shot_ceiling": attempts_ceiling,
        "projected_gate_applications": projected_gate_applications,
        "gate_application_ceiling": gates_ceiling,
        "executed_counts": False,
    }


def _selector_generalization_summary(context: GeneralizationContext) -> pd.DataFrame:
    support_registry = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout_support = pd.read_csv(context.output_dir / "heldout_instance_summary.csv")
    heldout_rows = pd.read_csv(context.output_dir / "heldout_results.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    stability = pd.read_csv(context.output_dir / "support_stability_summary.csv")
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    primary = context.config["primary_comparison"]
    k_budget = int(primary["k_budget"])
    slot_budget = int(primary["slot_budget"])
    tolerance = float(primary["tie_relative_tolerance"])
    epsilon = float(primary["tie_epsilon"])
    selectors = [*DETERMINISTIC_SELECTORS, RANDOM_SELECTOR]
    scopes = [*context.config["required_cases"], "overall"]
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for selector in selectors:
            registry = support_registry[support_registry["selector"] == selector]
            if scope != "overall":
                registry = registry[registry["ieee_case"] == scope]
            feasible_fraction = float((registry["status"] == "completed").mean())
            selected = heldout_support[
                (heldout_support["selector"] == selector)
                & (heldout_support["k_budget"] == k_budget)
                & (heldout_support["slot_budget"] == slot_budget)
                & (heldout_support["status"] == "completed")
            ]
            if scope != "overall":
                selected = selected[selected["ieee_case"] == scope]
            by_instance = (
                selected.groupby(["instance_id", "ieee_case"], sort=True)
                .agg(
                    median_normalized_error=("median_normalized_error", "median"),
                    p90_normalized_error=("p90_normalized_error", "median"),
                    worst_normalized_error=("worst_normalized_error", "max"),
                    failure_fraction=("failure_fraction", "mean"),
                )
                .reset_index()
            )
            baseline = heldout_support[
                (heldout_support["selector"] == "balanced_magnitude")
                & (heldout_support["k_budget"] == k_budget)
                & (heldout_support["slot_budget"] == slot_budget)
                & (heldout_support["status"] == "completed")
            ][["instance_id", "median_normalized_error"]].rename(
                columns={"median_normalized_error": "baseline_error"}
            )
            matched = by_instance.merge(baseline, on="instance_id", how="inner")
            outcomes = [
                classify_matched_errors(
                    candidate,
                    baseline_error,
                    relative_tolerance=tolerance,
                    epsilon=epsilon,
                )
                for candidate, baseline_error in zip(
                    matched["median_normalized_error"],
                    matched["baseline_error"],
                    strict=True,
                )
            ]
            outcome_counts = pd.Series(outcomes, dtype=str).value_counts()
            resource = resources[
                (resources["selector"] == selector)
                & (resources["k_budget"] == k_budget)
                & (resources["slot_budget"] == slot_budget)
                & (resources["status"] == "completed")
            ]
            if scope != "overall":
                resource = resource[resource["ieee_case"] == scope]
            certificate = heldout_rows[
                (heldout_rows["selector"] == selector)
                & (heldout_rows["k_budget"] == k_budget)
                & (heldout_rows["slot_budget"] == slot_budget)
                & (heldout_rows["status"] == "completed")
            ]
            if scope != "overall":
                certificate = certificate[certificate["ieee_case"] == scope]
            stability_row = stability[stability["selector"] == selector]
            qsvt_rows = qsvt[(qsvt["selector"] == selector)]
            if scope != "overall":
                qsvt_rows = qsvt_rows[qsvt_rows["ieee_case"] == scope]
            tightness = certificate["certificate_tightness"].dropna()
            rows.append(
                {
                    "summary_scope": scope,
                    "selector": selector,
                    "k_budget": k_budget,
                    "slot_budget": slot_budget,
                    "instances_evaluated": int(by_instance["instance_id"].nunique()),
                    "feasible_support_fraction_all_budgets": feasible_fraction,
                    "heldout_median_normalized_error": float(
                        by_instance["median_normalized_error"].median()
                    ),
                    "heldout_p90_normalized_error": float(
                        by_instance["p90_normalized_error"].quantile(0.9)
                    ),
                    "heldout_worst_normalized_error": float(
                        by_instance["worst_normalized_error"].max()
                    ),
                    "failure_fraction": float(by_instance["failure_fraction"].mean()),
                    "wins_vs_magnitude": int(outcome_counts.get("win", 0)),
                    "ties_vs_magnitude": int(outcome_counts.get("tie", 0)),
                    "losses_vs_magnitude": int(outcome_counts.get("loss", 0)),
                    "median_signal_unitary_gate_count": float(
                        resource["signal_unitary_gate_count"].median()
                    ),
                    "median_support_stability_jaccard": (
                        float(stability_row.iloc[0]["median_jaccard"])
                        if not stability_row.empty
                        else np.nan
                    ),
                    "certificate_coverage": float(
                        certificate["certificate_holds"].astype(bool).mean()
                    ),
                    "median_certificate_tightness": float(tightness.median()),
                    "qsvt_validation_status": (
                        "passed"
                        if not qsvt_rows.empty
                        and bool((qsvt_rows["status"] == "completed").all())
                        else "not_in_predeclared_subset"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _generalization_verdict(primary: Mapping[str, Any], qsvt: pd.DataFrame) -> str:
    outcomes = primary["primary_overall_win_tie_loss"]
    median_difference = float(
        primary["primary_median_paired_difference_sensitivity_minus_magnitude"]
    )
    all_cases_qsvt = set(qsvt["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    qsvt_passed = all_cases_qsvt and bool((qsvt["status"] == "completed").all())
    case_outcomes = primary["primary_case_win_tie_loss"]
    case_consistent = all(
        counts["win"] >= counts["loss"] for counts in case_outcomes.values()
    )
    if (
        outcomes["win"] > outcomes["loss"]
        and median_difference < 0.0
        and case_consistent
        and qsvt_passed
    ):
        return "Output-aware sparse selection generalized across multiple PSSE-derived workloads"
    if outcomes["win"] > outcomes["loss"] and median_difference < 0.0 and qsvt_passed:
        return "Generalization is case-dependent but statistically supported in part"
    return "Single-instance gains did not generalize reliably"


def stage_summary(context: GeneralizationContext) -> dict[str, Any]:
    summary = _selector_generalization_summary(context)
    atomic_write_csv(context.output_dir / "selector_generalization_summary.csv", summary)
    primary = json.loads(
        (context.output_dir / "generalization_primary_test.json").read_text(encoding="utf-8")
    )
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    verdict = _generalization_verdict(primary, qsvt)
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    stability = pd.read_csv(context.output_dir / "support_stability_summary.csv")
    certificate = pd.read_csv(context.output_dir / "certificate_case_summary.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    frozen = json.loads(
        (context.output_dir / "frozen_selector_configuration.json").read_text()
    )
    deterministic_resource_records = int(
        resources["selector"].isin(DETERMINISTIC_SELECTORS).sum()
    )
    lines = [
        "# Multi-Instance Output-Aware Sparse Selection Summary",
        "",
        "- Frozen configuration fingerprint: "
        f"`{frozen['configuration_fingerprint']}`",
        f"- Included evaluation instances: {len(instances)}",
        f"- Cases: {', '.join(sorted(instances['ieee_case'].unique()))}",
        "- Development block included in primary score: no",
        f"- Primary win/tie/loss: {primary['primary_overall_win_tie_loss']}",
        "- Primary median paired difference (sensitivity minus magnitude): "
        f"{primary['primary_median_paired_difference_sensitivity_minus_magnitude']:.12g}",
        "- Primary 95% paired-bootstrap interval: "
        f"{primary['primary_bootstrap_confidence_interval_95']}",
        f"- Certificate violations: {int((certificate['violations'] > 0).sum())}",
        f"- Deterministic resource records: {deterministic_resource_records}",
        f"- QSVT instances: {qsvt['instance_id'].nunique()}",
        f"- QSVT failures: {int((qsvt['status'] != 'completed').sum())}",
        f"- Stability minimum Jaccard: {stability['worst_jaccard'].min():.6g}",
        f"- Generalization verdict: **{verdict}**",
        "",
        "Generated measurements are controlled PYPOWER benchmark-model calculations, "
        "not field PMU/SCADA data. Statevector execution is not hardware execution.",
        "",
    ]
    _atomic_write_text(context.output_dir / "summary.md", "\n".join(lines))
    atomic_write_json(
        context.output_dir / "summary_metrics.json",
        {
            "verdict": verdict,
            "primary": primary,
            "instances": len(instances),
            "cases": sorted(instances["ieee_case"].unique().tolist()),
            "qsvt_instances": int(qsvt["instance_id"].nunique()),
        },
    )
    return {
        "selector_summary_rows": len(summary),
        "verdict": verdict,
    }


def _compare_protected_snapshot(context: GeneralizationContext) -> dict[str, Any]:
    path = context.output_dir / "protected_path_snapshot.json"
    initial = json.loads(path.read_text(encoding="utf-8"))
    current = _protected_file_snapshot(context.root, initial["protected_paths"])
    original = dict(initial["files"])
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


def _internal_verification_checks(context: GeneralizationContext) -> dict[str, Any]:
    frozen = json.loads(
        (context.output_dir / "frozen_selector_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    instances = pd.read_csv(context.output_dir / "instance_registry.csv")
    exclusions = pd.read_csv(context.output_dir / "instance_exclusion_registry.csv")
    functionals = pd.read_csv(context.output_dir / "functional_registry.csv")
    residuals = pd.read_csv(context.output_dir / "residual_registry.csv")
    supports = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout = pd.read_csv(context.output_dir / "heldout_results.csv")
    certificates = pd.read_csv(context.output_dir / "certificate_results.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    protected = _compare_protected_snapshot(context)

    split_overlap = 0
    for _instance_id, group in residuals.groupby("instance_id", sort=True):
        training = set(group[group["split"] == "training"]["residual_seed"])
        held_out = set(group[group["split"] == "held_out"]["residual_seed"])
        split_overlap += len(training & held_out)
    support_constraint_failures = 0
    completed_supports = supports[supports["status"] == "completed"]
    for record in completed_supports.itertuples(index=False):
        payload = _load_instance_payload(context, str(record.instance_id))
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        support = load_generalization_support(context, str(record.support_file))
        report = support_constraint_report(
            matrix,
            support,
            SupportConstraints(int(record.k_budget), int(record.slot_budget), True),
        )
        support_constraint_failures += int(not report["valid"])
    deterministic_resources = resources[
        resources["selector"].isin(DETERMINISTIC_SELECTORS)
    ]
    checks = {
        "frozen_configuration_fingerprint_valid": (
            frozen["configuration_fingerprint"] == configuration_fingerprint(frozen)
        ),
        "development_matrix_excluded": (
            context.config["development_matrix_fingerprint"]
            not in set(instances["matrix_fingerprint"])
        ),
        "minimum_instance_count": len(instances) >= 12,
        "required_cases_represented": set(instances["ieee_case"])
        == {"ieee14", "ieee30", "ieee57"},
        "instance_inclusion_outcome_independent": bool(
            (~instances["selector_outcomes_used_for_inclusion"].astype(bool)).all()
        ),
        "exclusions_have_reasons": bool(
            exclusions.empty or exclusions["failure_reason"].fillna("").str.len().gt(0).all()
        ),
        "three_functionals_per_instance": len(functionals) == 3 * len(instances),
        "residual_split_overlap_zero": split_overlap == 0,
        "residual_failures_zero": bool(
            (residuals["generation_status"] == "completed").all()
        ),
        "support_constraints_hold": support_constraint_failures == 0,
        "heldout_only_results": set(heldout["split"].dropna()) == {"held_out"},
        "certificate_formula_unchanged": bool(
            (
                certificates["certificate_formula_version"]
                == context.config["certificate"]["formula_version"]
            ).all()
        ),
        "certificate_actual_error_not_used": bool(
            (~certificates["certificate_used_actual_error_in_computation"].astype(bool)).all()
        ),
        "certificate_violations_zero": bool(
            certificates["certificate_holds"].astype(bool).all()
        ),
        "deterministic_resource_measurements_recorded": bool(
            deterministic_resources["signal_unitary_gate_count"].notna().all()
            and (deterministic_resources["signal_unitary_gate_count"] > 0).all()
        ),
        "resource_validation_failures_retained": bool(
            deterministic_resources.loc[
                deterministic_resources["status"] != "completed", "failure_reason"
            ]
            .fillna("")
            .str.len()
            .gt(0)
            .all()
        ),
        "resource_costs_actual_not_zero_filled": bool(
            (
                (deterministic_resources["signal_unitary_gate_count"] > 0)
                & (~deterministic_resources["missing_cost_is_zero"].astype(bool))
            ).all()
        ),
        "qsvt_required_cases_represented": set(qsvt["ieee_case"])
        == {"ieee14", "ieee30", "ieee57"},
        "qsvt_failures_zero": bool((qsvt["status"] == "completed").all()),
        "qsvt_no_per_support_refit": bool(
            (~qsvt["per_support_phase_refit"].astype(bool)).all()
        ),
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
            "instances": len(instances),
            "functionals": len(functionals),
            "residuals": len(residuals),
            "support_records": len(supports),
            "heldout_rows": len(heldout),
            "certificate_rows": len(certificates),
            "resource_rows": len(resources),
            "qsvt_rows": len(qsvt),
        },
    }


def stage_verify(context: GeneralizationContext) -> dict[str, Any]:
    report = _internal_verification_checks(context)
    external_path = context.output_dir / "verification_commands.json"
    external = (
        json.loads(external_path.read_text(encoding="utf-8"))
        if external_path.is_file()
        else {"status": "not_recorded_yet"}
    )
    lines = [
        "# Output-Aware Generalization Verification Report",
        "",
        f"- Generated: {now_iso()}",
        f"- Internal verification: {'PASS' if report['all_internal_checks_pass'] else 'FAIL'}",
        f"- External command record: `{external.get('status', 'recorded')}`",
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
    return {
        "all_internal_checks_pass": True,
        "protected_paths_unchanged": report["checks"]["protected_paths_unchanged"],
        **report["counts"],
    }


def refresh_manifest_and_checksums(context: GeneralizationContext) -> None:
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
    checksum_targets = [
        path
        for path in sorted(context.output_dir.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    lines: list[str] = []
    for path in checksum_targets:
        try:
            relative = path.relative_to(context.root).as_posix()
        except ValueError:
            relative = path.as_posix()
        lines.append(f"{_sha256_file(path)}  {relative}")
    _atomic_write_text(checksum_path, "\n".join(lines) + "\n")


STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "audit": ("implementation_audit.md", "protected_path_snapshot.json"),
    "freeze": ("frozen_selector_configuration.json", "study_configuration.json"),
    "instances": ("instance_registry.csv", "instance_exclusion_registry.csv"),
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
        "generalization_primary_test.json",
        "generalization_bootstrap.csv",
    ),
    "stability": ("support_stability.csv", "support_stability_summary.csv"),
    "certificates": ("certificate_results.csv", "certificate_case_summary.csv"),
    "resources": ("resource_registry.csv", "resource_case_summary.csv"),
    "pareto": (
        "pareto_candidates_error_nnz.csv",
        "pareto_frontier_error_nnz.csv",
        "pareto_candidates_error_slots.csv",
        "pareto_frontier_error_slots.csv",
        "pareto_candidates_error_gates.csv",
        "pareto_frontier_error_gates.csv",
    ),
    "qsvt": ("qsvt_instance_designs.json", "qsvt_validation_results.csv"),
    "finite-shot": ("finite_shot_status.md", "finite_shot_results.csv"),
    "summary": ("selector_generalization_summary.csv", "summary.md"),
    "verify": ("verification_report.md", "internal_verification.json"),
}


STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "freeze": stage_freeze,
    "instances": stage_instances,
    "functionals": stage_functionals,
    "residuals": stage_residuals,
    "supports": stage_supports,
    "heldout": stage_heldout,
    "primary-test": stage_primary_test,
    "stability": stage_stability,
    "certificates": stage_certificates,
    "resources": stage_resources,
    "pareto": stage_pareto,
    "qsvt": stage_qsvt,
    "finite-shot": stage_finite_shot,
    "summary": stage_summary,
    "verify": stage_verify,
}


def run_generalization_study(
    context: GeneralizationContext, *, stage: str = "all"
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
