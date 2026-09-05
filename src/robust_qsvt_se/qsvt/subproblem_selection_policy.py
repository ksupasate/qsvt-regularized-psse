from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    _subproblem_metadata,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system, select_subproblem
from robust_qsvt_se.utils.io import ensure_directory

SELECTION_POLICY_CLAIM = (
    "Subproblems are selected by numerical and metadata criteria, not by post "
    "hoc QSVT performance alone."
)

SELECTION_COLUMNS = [
    "candidate_id",
    "case",
    "model",
    "case_source",
    "selection_source",
    "row_indices",
    "col_indices",
    "matrix_shape",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "ridge_residual",
    "no_update_residual",
    "ridge_residual_reduction_ratio",
    "ridge_update_norm",
    "metadata_available",
    "physical_labels",
    "score",
    "selected",
    "rejection_reason",
]


@dataclass(frozen=True, slots=True)
class CandidateSubproblem:
    candidate_id: str
    selection_source: str
    rows: np.ndarray
    columns: np.ndarray


def run_qsvt_subproblem_selection_policy(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "candidate_modes": [
            "best_conditioned",
            "high_leverage",
            "metadata_mapped",
            "residual_supported",
            "random_seeded_pool",
            "worst_conditioned_control",
        ],
        "condition_threshold": 1.0e8,
        "alpha": 1.0e-4,
        "seed": 123,
        "output_dir": "outputs/qsvt_subproblem_selection_policy",
        "max_selected": 3,
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    system, matrix_source = _build_system(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        seed=int(resolved["seed"]),
    )
    candidates = generate_candidate_subproblems(
        system=system,
        matrix_source=matrix_source,
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        candidate_modes=[str(value) for value in resolved["candidate_modes"]],
        seed=int(resolved["seed"]),
    )
    rows = score_candidate_subproblems(
        system=system,
        matrix_source=matrix_source,
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        candidates=candidates,
        condition_threshold=float(resolved["condition_threshold"]),
        alpha=float(resolved["alpha"]),
        max_selected=int(resolved["max_selected"]),
    )
    artifacts = _write_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "candidate_rows": rows, "artifacts": artifacts}


def generate_candidate_subproblems(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case: str,
    model: str,
    submatrix_size: int,
    candidate_modes: list[str],
    seed: int,
) -> list[CandidateSubproblem]:
    candidates: list[CandidateSubproblem] = []
    for mode in candidate_modes:
        if mode == "random_seeded_pool":
            candidates.extend(
                _random_pool_candidates(
                    system=system,
                    size=int(submatrix_size),
                    seed=int(seed),
                    count=5,
                )
            )
            continue
        if mode == "residual_supported":
            rows, columns = _residual_supported_indices(system, int(submatrix_size))
        elif mode == "metadata_mapped":
            subproblem = select_subproblem(
                system=system,
                matrix_source=matrix_source,
                case=case,
                model=model,
                submatrix_size=int(submatrix_size),
                selection_mode="physically_mapped_if_available",
                seed=int(seed),
            )
            rows = np.asarray(subproblem.metadata["selected_measurement_indices"], dtype=np.int64)
            columns = np.asarray(subproblem.metadata["selected_state_indices"], dtype=np.int64)
        elif mode == "worst_conditioned_control":
            subproblem = select_subproblem(
                system=system,
                matrix_source=matrix_source,
                case=case,
                model=model,
                submatrix_size=int(submatrix_size),
                selection_mode="worst_conditioned",
                seed=int(seed),
            )
            rows = np.asarray(subproblem.metadata["selected_measurement_indices"], dtype=np.int64)
            columns = np.asarray(subproblem.metadata["selected_state_indices"], dtype=np.int64)
        else:
            subproblem = select_subproblem(
                system=system,
                matrix_source=matrix_source,
                case=case,
                model=model,
                submatrix_size=int(submatrix_size),
                selection_mode=mode,
                seed=int(seed),
            )
            rows = np.asarray(subproblem.metadata["selected_measurement_indices"], dtype=np.int64)
            columns = np.asarray(subproblem.metadata["selected_state_indices"], dtype=np.int64)
        candidates.append(
            CandidateSubproblem(
                candidate_id=f"{mode}_{len(candidates):02d}",
                selection_source=mode,
                rows=np.sort(rows),
                columns=np.sort(columns),
            )
        )
    return candidates


def score_candidate_subproblems(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case: str,
    model: str,
    case_source: str,
    candidates: list[CandidateSubproblem],
    condition_threshold: float,
    alpha: float,
    max_selected: int,
) -> list[dict[str, Any]]:
    scored = [
        _score_candidate(
            system=system,
            matrix_source=matrix_source,
            case=case,
            model=model,
            case_source=case_source,
            candidate=candidate,
            condition_threshold=float(condition_threshold),
            alpha=float(alpha),
        )
        for candidate in candidates
    ]
    eligible = [
        row
        for row in scored
        if not row["rejection_reason"] and row["selection_source"] != "worst_conditioned_control"
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["score"]),
            float(row["condition_number"]),
            str(row["candidate_id"]),
        )
    )
    selected_keys: set[tuple[str, str]] = set()
    selected_ids: set[str] = set()
    for row in eligible:
        key = (str(row["row_indices"]), str(row["col_indices"]))
        if key in selected_keys:
            row["rejection_reason"] = "duplicate_of_higher_scoring_selected_candidate"
            continue
        if len(selected_ids) >= int(max_selected):
            row["rejection_reason"] = "lower_score_than_selected_cutoff"
            continue
        selected_keys.add(key)
        selected_ids.add(str(row["candidate_id"]))
    for row in scored:
        row["selected"] = str(row["candidate_id"]) in selected_ids
        if row["selection_source"] == "worst_conditioned_control" and not row["rejection_reason"]:
            row["rejection_reason"] = "worst_conditioned_control_not_selected_by_policy"
    return scored


def build_selected_subproblem_from_policy_row(
    *,
    system: WeightedSystem,
    matrix_source: str,
    row: dict[str, Any],
) -> SelectedSubproblem:
    rows = _parse_indices(row["row_indices"])
    columns = _parse_indices(row["col_indices"])
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    metadata = _subproblem_metadata(
        system=system,
        matrix_source=matrix_source,
        case=str(row.get("case", "ieee14")),
        model=str(row.get("model", "ac_linearized")),
        selected_rows=rows,
        selected_columns=columns,
    )
    metadata["selection_mode"] = str(row.get("selection_source", "selection_policy"))
    metadata["candidate_id"] = str(row.get("candidate_id", "selected_subproblem"))
    return SelectedSubproblem(
        H_tilde=H_full[np.ix_(rows, columns)],
        r_tilde=r_full[rows],
        metadata=metadata,
    )


def _score_candidate(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case: str,
    model: str,
    case_source: str,
    candidate: CandidateSubproblem,
    condition_threshold: float,
    alpha: float,
) -> dict[str, Any]:
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    H = H_full[np.ix_(candidate.rows, candidate.columns)]
    r = r_full[candidate.rows]
    metadata = _subproblem_metadata(
        system=system,
        matrix_source=matrix_source,
        case=case,
        model=model,
        selected_rows=candidate.rows,
        selected_columns=candidate.columns,
    )
    singular_values = np.linalg.svd(H, compute_uv=False)
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = _strict_condition_number(singular_values)
    ridge = ridge_tikhonov_update(H, r, alpha=float(alpha))
    ridge_residual = float(np.linalg.norm(H @ ridge - r))
    no_update = float(np.linalg.norm(r))
    ridge_ratio = ridge_residual / max(no_update, 1.0e-15)
    ridge_norm = float(np.linalg.norm(ridge))
    metadata_available, physical_labels = _metadata_status(metadata)
    leverage = _leverage_score(H_full, candidate.rows, candidate.columns)
    score = _policy_score(
        condition=condition,
        condition_threshold=float(condition_threshold),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        no_update=no_update,
        ridge_ratio=ridge_ratio,
        ridge_norm=ridge_norm,
        metadata_available=metadata_available,
        physical_labels=physical_labels,
        leverage=leverage,
    )
    rejection = _rejection_reason(
        condition=condition,
        condition_threshold=float(condition_threshold),
        sigma_min=sigma_min,
        no_update=no_update,
        ridge_ratio=ridge_ratio,
        ridge_norm=ridge_norm,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "case": case,
        "model": model,
        "case_source": case_source,
        "selection_source": candidate.selection_source,
        "row_indices": _format_indices(candidate.rows),
        "col_indices": _format_indices(candidate.columns),
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "condition_number": condition,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "ridge_residual": ridge_residual,
        "no_update_residual": no_update,
        "ridge_residual_reduction_ratio": ridge_ratio,
        "ridge_update_norm": ridge_norm,
        "metadata_available": metadata_available,
        "physical_labels": physical_labels,
        "score": score,
        "selected": False,
        "rejection_reason": rejection,
    }


def _policy_score(
    *,
    condition: float,
    condition_threshold: float,
    sigma_min: float,
    sigma_max: float,
    no_update: float,
    ridge_ratio: float,
    ridge_norm: float,
    metadata_available: bool,
    physical_labels: str,
    leverage: float,
) -> float:
    score = 0.0
    if np.isfinite(condition):
        score += 1.0
    if np.isfinite(condition) and condition <= condition_threshold:
        score += 2.0
    if no_update > 1.0e-12:
        score += 1.0
    if sigma_min > 1.0e-12 and sigma_max > 0.0:
        score += 1.0
    if ridge_norm > 1.0e-12:
        score += 1.0
    if metadata_available:
        score += 1.0
    if physical_labels:
        score += 1.0
    score += min(max(float(leverage), 0.0), 1.0)
    score += 2.0 * max(0.0, min(1.0, 1.0 - float(ridge_ratio)))
    return float(score)


def _rejection_reason(
    *,
    condition: float,
    condition_threshold: float,
    sigma_min: float,
    no_update: float,
    ridge_ratio: float,
    ridge_norm: float,
) -> str:
    if not np.isfinite(condition) or condition > condition_threshold:
        return "condition_number_exceeds_threshold_or_rank_deficient"
    if sigma_min <= 1.0e-12:
        return "pathological_near_rank_deficient_block"
    if no_update <= 1.0e-12:
        return "zero_residual_support"
    if ridge_norm <= 1.0e-12:
        return "ridge_update_norm_near_zero"
    if ridge_ratio >= 0.95:
        return "weak_ridge_residual_reduction_potential"
    return ""


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    candidate_path = output_dir / "candidate_subproblem_scores.csv"
    selected_path = output_dir / "selected_subproblems.csv"
    rejected_path = output_dir / "rejected_subproblems.csv"
    policy_path = output_dir / "selection_policy.md"
    interpretation_path = output_dir / "selection_policy_interpretation.md"
    frame = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    frame.to_csv(candidate_path, index=False)
    frame[frame["selected"] == True].to_csv(selected_path, index=False)  # noqa: E712
    frame[frame["selected"] != True].to_csv(rejected_path, index=False)  # noqa: E712
    policy_path.write_text(_policy_markdown(resolved), encoding="utf-8")
    interpretation_path.write_text(_interpretation_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "candidate_subproblem_scores": str(candidate_path),
            "selected_subproblems": str(selected_path),
            "rejected_subproblems": str(rejected_path),
            "selection_policy": str(policy_path),
            "selection_policy_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=SELECTION_POLICY_CLAIM,
    )
    return {
        "manifest": manifest,
        "candidate_subproblem_scores": candidate_path,
        "selected_subproblems": selected_path,
        "rejected_subproblems": rejected_path,
        "selection_policy": policy_path,
        "selection_policy_interpretation": interpretation_path,
    }


def _policy_markdown(resolved: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Criteria-Based QSVT Subproblem Selection Policy",
            "",
            SELECTION_POLICY_CLAIM,
            "",
            "The score uses finite conditioning, residual support, singular-value "
            "support, Ridge update norm, metadata availability, physical labels, "
            "leverage, and Ridge-only residual-reduction potential.",
            "",
            f"- Condition threshold: {float(resolved['condition_threshold']):.17g}",
            f"- Candidate modes: {', '.join(str(value) for value in resolved['candidate_modes'])}",
            "- QSVT residuals are not used in the selection score.",
            "- A worst-conditioned control is retained as a rejected/failure case.",
            "",
        ]
    )


def _interpretation_markdown(frame: pd.DataFrame) -> str:
    selected = int((frame["selected"] == True).sum())  # noqa: E712
    rejected = int((frame["selected"] != True).sum())  # noqa: E712
    controls = frame[frame["selection_source"] == "worst_conditioned_control"]
    control_statement = "yes" if not controls.empty else "no"
    return "\n".join(
        [
            "# Subproblem Selection Policy Interpretation",
            "",
            SELECTION_POLICY_CLAIM,
            "",
            f"- Candidates generated: {len(frame)}",
            f"- Selected subproblems: {selected}",
            f"- Rejected subproblems: {rejected}",
            f"- Worst-conditioned control included: {control_statement}",
            "- Rejection reasons are retained to show failure behavior honestly.",
            "",
        ]
    )


def _random_pool_candidates(
    *,
    system: WeightedSystem,
    size: int,
    seed: int,
    count: int,
) -> list[CandidateSubproblem]:
    H = np.asarray(system.H_tilde)
    rng = np.random.default_rng(int(seed))
    candidates = []
    for index in range(int(count)):
        rows = np.sort(rng.choice(H.shape[0], size=int(size), replace=False))
        columns = np.sort(rng.choice(H.shape[1], size=int(size), replace=False))
        candidates.append(
            CandidateSubproblem(
                candidate_id=f"random_seeded_pool_{index:02d}",
                selection_source="random_seeded_pool",
                rows=rows.astype(np.int64),
                columns=columns.astype(np.int64),
            )
        )
    return candidates


def _residual_supported_indices(system: WeightedSystem, size: int) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    rows = _top_indices(np.abs(r), int(size))
    column_scores = np.linalg.norm(H[rows, :], axis=0) * (
        np.abs(H.T @ r) / max(float(np.linalg.norm(H.T @ r)), 1.0e-15)
    )
    columns = _top_indices(column_scores, int(size))
    return rows, columns


def _metadata_status(metadata: dict[str, Any]) -> tuple[bool, str]:
    records = list(metadata.get("state_index_mapping", []))
    labels = []
    for record in records:
        state_type = str(record.get("state_type", "unknown"))
        label = str(record.get("label", ""))
        if state_type in {"angle", "voltage"}:
            labels.append(label)
    return bool(records), ";".join(label for label in labels if label)


def _leverage_score(H_full: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> float:
    row_norms = np.linalg.norm(H_full, axis=1)
    col_norms = np.linalg.norm(H_full, axis=0)
    row_score = float(np.sum(row_norms[rows]) / max(float(np.sum(row_norms)), 1.0e-15))
    col_score = float(np.sum(col_norms[columns]) / max(float(np.sum(col_norms)), 1.0e-15))
    return 0.5 * (row_score + col_score)


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[: int(count)]).astype(np.int64)


def _strict_condition_number(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.size == 0:
        return float("inf")
    sigma_min = float(np.min(values))
    if sigma_min <= 1.0e-14:
        return float("inf")
    return float(np.max(values) / sigma_min)


def _format_indices(values: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in values)


def _parse_indices(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray([int(part) for part in value.split() if part], dtype=np.int64)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.int64)
    raise ValueError(f"cannot parse index list from {value!r}")
