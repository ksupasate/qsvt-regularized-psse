"""Workstream C - physically implementable measurement-row sparsification.

A measurement-row decision retains or removes an ENTIRE measurement equation and its corresponding
weighted-Jacobian row (residual entry, standard-deviation entry, and row metadata together); no
partial row is ever kept.  Whole-row selection is exactly the restriction of arbitrary entry
selection to whole-row masks, so both are evaluated through the identical Ridge path
``ridge(H * mask, r, alpha)`` and compared at fair nonzero / row-count budgets.

Reuses the frozen structural-group design (24 instances = 12 benchmark-derived structural groups x 2
realizations over IEEE-14/30/57), per-instance physical functionals, per-seed controlled residual +
truth, the SVD Ridge operator, the exact selected-output gradient, and the risk selectors.  Truth is
used only for controlled evaluation, never for support construction.  Structural groups remain the
statistical unit; no result claims a physically optimal support or QSVT superiority over Ridge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.physical_alignment.risk_selectors import evaluate_risk
from robust_qsvt_se.physical_alignment.structures import (
    build_instance_functionals,
    instance_residual_and_truth,
    load_instance,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import _ridge_filter_operator
from robust_qsvt_se.qsvt.ridge_output_certificate import ridge_selected_output_gradient
from robust_qsvt_se.reviewer_blocking.common import (
    atomic_write_csv,
    atomic_write_json,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.tqe_extensions.common import CLAIM_BOUNDARY, load_yaml_config

STUDY_ID = "tqe_measurement_row_sparsification_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_measurement_row_sparsification")
DEFAULT_CONFIG_PATH = Path("configs/tqe_measurement_row_sparsification.yaml")

ENTRY_SELECTORS = ("entry_global_magnitude", "entry_sensitivity_mean")
ROW_SELECTORS = (
    "row_magnitude",
    "row_leverage",
    "row_sensitivity_mean",
    "row_sensitivity_worst",
    "row_noise_risk",
    "row_posterior_risk",
    "random_row",
)


# --------------------------------------------------------------------------- data


@dataclass(slots=True)
class Structure:
    instance_id: str
    group_id: str
    case: str
    realization: int
    matrix: np.ndarray
    alpha: float
    row_meta: list[dict[str, Any]]
    functionals: list[np.ndarray]
    functional_ids: list[str]
    training_seeds: list[int]
    heldout_seeds: list[int]
    instance: Any  # StructuralInstance (for the frozen residual/truth pathway)

    def seed_data(self, seeds: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Per-seed ``(seed, residual[selected_rows], truth[selected_columns])`` (frozen "
        "pathway)."""

        out = []
        for seed in seeds:
            residual, truth, _ref = instance_residual_and_truth(self.instance, int(seed))
            out.append((int(seed), np.asarray(residual, np.float64), np.asarray(truth, np.float64)))
        return out


def load_structures(source_root: str | Path, config: dict[str, Any]) -> list[Structure]:
    root = Path(source_root)
    n_train = int(config["training_seed_count"])
    n_heldout = int(config["heldout_seed_count"])
    classification = str(config.get("functional_classification", "physical"))
    limit = config.get("max_structures")  # test hook; None => all frozen instances
    structures: list[Structure] = []
    for path in sorted((root / "instances").glob("*.json")):
        if limit is not None and len(structures) >= int(limit):
            break
        instance_id = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("inclusion_status") != "included":
            continue
        inst = load_instance(root, instance_id)
        split = json.loads(
            (root / "residual_splits" / f"{instance_id}.json").read_text(encoding="utf-8")
        )
        funcs = [
            f
            for f in build_instance_functionals(inst)
            if f.status == "available"
            and f.vector is not None
            and f.classification == classification
        ]
        if not funcs:
            continue
        structures.append(
            Structure(
                instance_id=instance_id,
                group_id=inst.structural_group_id,
                case=inst.ieee_case,
                realization=inst.realization_order,
                matrix=np.asarray(inst.matrix, dtype=np.float64),
                alpha=float(inst.alpha),
                row_meta=[dict(m) for m in inst.measurement_metadata],
                functionals=[np.asarray(f.vector, dtype=np.float64) for f in funcs],
                functional_ids=[f.functional_id for f in funcs],
                training_seeds=[int(s) for s in split["training_seed_ids"][:n_train]],
                heldout_seeds=[int(s) for s in split["held_out_seed_ids"][:n_heldout]],
                instance=inst,
            )
        )
    return structures


# --------------------------------------------------------------------------- Ridge


def ridge_solution(matrix: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    return _ridge_filter_operator(matrix, alpha) @ residual


# --------------------------------------------------------------------------- scores


def entry_magnitude_score(matrix: np.ndarray) -> np.ndarray:
    return np.abs(matrix)


def _task_gradients(matrix, tasks, alpha) -> list[np.ndarray]:
    return [
        ridge_selected_output_gradient(matrix, residual, functional, alpha)
        for (residual, functional) in tasks
    ]


def entry_sensitivity_score(matrix, tasks, alpha) -> np.ndarray:
    """Per-entry task-aware sensitivity, L1-normalized within each task then averaged (task "
    "5.4/7.4)."""

    acc = np.zeros_like(matrix, dtype=np.float64)
    for gradient in _task_gradients(matrix, tasks, alpha):
        contribution = np.abs(matrix * gradient)
        total = float(contribution.sum())
        if total > 0.0:
            acc += contribution / total
    return acc / max(len(tasks), 1)


def row_magnitude_score(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix, axis=1)


def row_leverage_score(matrix: np.ndarray, alpha: float) -> np.ndarray:
    gram = matrix.T @ matrix + float(alpha) * np.eye(matrix.shape[1])
    resolvent = np.linalg.inv(gram)
    return np.einsum("ij,jk,ik->i", matrix, resolvent, matrix)


def row_sensitivity_scores(matrix, tasks, alpha, *, aggregation: str) -> np.ndarray:
    """Row-level task-aware score: normalize the entry sensitivity within each task, sum over the
    row, then aggregate across tasks (``mean`` or ``worst_case`` = min-over-tasks robustness)."""

    per_task_rows = []
    for gradient in _task_gradients(matrix, tasks, alpha):
        contribution = np.abs(matrix * gradient)
        total = float(contribution.sum())
        row_scores = contribution.sum(axis=1) / total if total > 0.0 else contribution.sum(axis=1)
        per_task_rows.append(row_scores)
    stacked = np.vstack(per_task_rows) if per_task_rows else np.zeros((1, matrix.shape[0]))
    if aggregation == "mean":
        return stacked.mean(axis=0)
    if aggregation == "worst_case":
        return stacked.min(axis=0)  # a row important even in its least-favorable task
    raise ValueError(f"unknown aggregation {aggregation}")


# --------------------------------------------------------------------------- supports


def _row_mask(matrix: np.ndarray, rows: set[int]) -> np.ndarray:
    """8x8 boolean keeping every original nonzero of the selected rows (whole-row retention)."""

    mask = np.zeros_like(matrix, dtype=bool)
    pattern = matrix != 0.0
    for i in rows:
        mask[i, :] = pattern[i, :]
    return mask


def _entry_mask_topk(score: np.ndarray, matrix: np.ndarray, k_nnz: int) -> np.ndarray:
    """Top-``k_nnz`` entries by score among original nonzeros (deterministic tie-break by index)."""

    pattern = matrix != 0.0
    flat_idx = np.argwhere(pattern)
    scores = np.array([score[i, j] for i, j in flat_idx], dtype=np.float64)
    order = np.lexsort((flat_idx[:, 1], flat_idx[:, 0], -scores))  # -score primary; then i,j
    keep = order[: int(k_nnz)]
    mask = np.zeros_like(matrix, dtype=bool)
    for idx in keep:
        i, j = flat_idx[idx]
        mask[i, j] = True
    return mask


def _rows_by_score(row_score: np.ndarray, count: int) -> set[int]:
    order = np.lexsort((np.arange(row_score.size), -row_score))
    return set(int(i) for i in order[: int(count)])


def _rows_greedy_nnz(row_score: np.ndarray, row_nnz: np.ndarray, budget_nnz: int) -> set[int]:
    order = np.lexsort((np.arange(row_score.size), -row_score))
    chosen: set[int] = set()
    used = 0
    for i in order:
        cost = int(row_nnz[i])
        if used + cost <= budget_nnz:
            chosen.add(int(i))
            used += cost
    return chosen


def _greedy_risk_rows(
    matrix,
    alpha,
    functionals,
    *,
    risk_kind: str,
    budget_rows: int | None,
    budget_nnz: int | None,
    row_nnz: np.ndarray,
) -> set[int]:
    """Forward-greedy whole-row selection minimizing the mean output risk over functionals."""

    n_rows = matrix.shape[0]
    chosen: set[int] = set()
    used_nnz = 0
    while True:
        if budget_rows is not None and len(chosen) >= budget_rows:
            break
        best_row, best_obj = None, np.inf
        for i in range(n_rows):
            if i in chosen:
                continue
            if budget_nnz is not None and used_nnz + int(row_nnz[i]) > budget_nnz:
                continue
            trial = chosen | {i}
            masked = np.where(_row_mask(matrix, trial), matrix, 0.0)
            try:
                obj = evaluate_risk(
                    masked, alpha, functionals, risk_kind=risk_kind, aggregation="mean"
                ).objective
            except Exception:
                continue
            if obj < best_obj or (np.isclose(obj, best_obj) and (best_row is None or i < best_row)):
                best_obj, best_row = obj, i
        if best_row is None:
            break
        chosen.add(best_row)
        used_nnz += int(row_nnz[best_row])
    return chosen


def _random_rows(n_rows: int, count: int, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(n_rows, size=min(count, n_rows), replace=False))


# --------------------------------------------------- exact-loss one-swap row refinement


def _stack_tasks(tasks) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.stack([r for r, _ell in tasks]) if tasks else np.zeros((0, 0))
    functionals = np.stack([ell for _r, ell in tasks]) if tasks else np.zeros((0, 0))
    return residuals, functionals


def _masked_loss(matrix, mask, alpha, residuals, functionals, y_full, floor) -> float:
    """Vectorized mean normalized selected-output error of the masked block vs full-support "
    "Ridge."""

    sparse_op = _ridge_filter_operator(np.where(mask, matrix, 0.0), alpha)
    y_sparse = np.einsum("tn,nm,tm->t", functionals, sparse_op, residuals)
    return float(np.mean(np.abs(y_sparse - y_full) / np.maximum(np.abs(y_full), floor)))


def refine_rows_one_swap(matrix, rows: set[int], tasks, alpha, floor, *, max_iters=8) -> set[int]:
    """Local exact-loss one-swap refinement over whole rows (no global-optimality claim)."""

    n_rows = matrix.shape[0]
    residuals, functionals = _stack_tasks(tasks)
    if residuals.size == 0:
        return set(rows)
    full_op = _ridge_filter_operator(matrix, alpha)
    y_full = np.einsum("tn,nm,tm->t", functionals, full_op, residuals)
    current = set(rows)
    best = _masked_loss(
        matrix, _row_mask(matrix, current), alpha, residuals, functionals, y_full, floor
    )
    for _ in range(max_iters):
        improved = False
        for out_row in sorted(current):
            for in_row in range(n_rows):
                if in_row in current:
                    continue
                trial = (current - {out_row}) | {in_row}
                loss = _masked_loss(
                    matrix, _row_mask(matrix, trial), alpha, residuals, functionals, y_full, floor
                )
                if loss < best - 1e-12:
                    best, current, improved = loss, trial, True
        if not improved:
            break
    return current


# --------------------------------------------------------------------------- feasibility


def row_set_feasibility(
    rows: set[int], row_meta: list[dict[str, Any]], constraints: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    min_rows = int(constraints.get("min_rows", 1))
    if len(rows) < min_rows:
        reasons.append(f"below_min_rows_{min_rows}")
    if constraints.get("require_type_coverage"):
        all_types = {str(m.get("measurement_type")) for m in row_meta}
        kept_types = {str(row_meta[i].get("measurement_type")) for i in rows}
        missing = all_types - kept_types
        if missing:
            reasons.append("missing_measurement_types_" + "|".join(sorted(missing)))
    if constraints.get("require_bus_coverage"):
        all_buses = {
            m.get("measurement_bus") for m in row_meta if m.get("measurement_bus") is not None
        }
        kept_buses = {row_meta[i].get("measurement_bus") for i in rows}
        if all_buses and not (all_buses <= kept_buses):
            reasons.append("incomplete_bus_coverage")
    return (len(reasons) == 0, reasons)


# --------------------------------------------------------------------------- support dispatch


def _row_nnz(matrix: np.ndarray) -> np.ndarray:
    return (matrix != 0.0).sum(axis=1).astype(int)


def build_support(
    structure: Structure,
    selector: str,
    protocol: str,
    budget: int,
    train_tasks: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build one support (8x8 mask) for a selector at a (protocol, budget); retain infeasible "
    "ones."""

    matrix = structure.matrix
    alpha = structure.alpha
    n_rows = matrix.shape[0]
    row_nnz = _row_nnz(matrix)
    avg_row_nnz = float(row_nnz.mean())
    floor = float(config.get("support_floor", 1e-6))
    is_row = selector in ROW_SELECTORS
    is_entry = selector in ENTRY_SELECTORS

    # Resolve the target for this protocol.
    if protocol == "row_count":
        target_rows = int(budget)
        target_nnz = round(budget * avg_row_nnz)
    else:  # nnz
        target_rows = None
        target_nnz = int(budget)

    rows: set[int] | None = None
    mask: np.ndarray

    if selector == "full_support":
        mask = matrix != 0.0
        rows = set(range(n_rows))
    elif is_entry:
        score = (
            entry_magnitude_score(matrix)
            if selector == "entry_global_magnitude"
            else entry_sensitivity_score(matrix, train_tasks, alpha)
        )
        mask = _entry_mask_topk(score, matrix, target_nnz)
        rows = set(int(i) for i in np.where(mask.any(axis=1))[0])
    elif selector == "random_row":
        count = (
            target_rows
            if target_rows is not None
            else max(1, int(np.floor(target_nnz / max(avg_row_nnz, 1e-9))))
        )
        rows = _random_rows(
            n_rows, count, int(config.get("random_seed", 0)) + hash(structure.instance_id) % 100000
        )
        mask = _row_mask(matrix, rows)
    elif selector in {"row_noise_risk", "row_posterior_risk"}:
        risk_kind = (
            "noise_propagation" if selector == "row_noise_risk" else "posterior_variance_reference"
        )
        rows = _greedy_risk_rows(
            matrix,
            alpha,
            structure.functionals,
            risk_kind=risk_kind,
            budget_rows=target_rows,
            budget_nnz=(None if protocol == "row_count" else target_nnz),
            row_nnz=row_nnz,
        )
        mask = _row_mask(matrix, rows)
    else:  # score-based row selectors
        if selector == "row_magnitude":
            score = row_magnitude_score(matrix)
        elif selector == "row_leverage":
            score = row_leverage_score(matrix, alpha)
        elif selector == "row_sensitivity_mean":
            score = row_sensitivity_scores(matrix, train_tasks, alpha, aggregation="mean")
        elif selector == "row_sensitivity_worst":
            score = row_sensitivity_scores(matrix, train_tasks, alpha, aggregation="worst_case")
        else:
            raise ValueError(f"unknown selector {selector}")
        if target_rows is not None:
            rows = _rows_by_score(score, target_rows)
        else:
            rows = _rows_greedy_nnz(score, row_nnz, target_nnz)
        if config.get("row_swap_refinement") and rows:
            rows = refine_rows_one_swap(matrix, rows, train_tasks, alpha, floor)
        mask = _row_mask(matrix, rows)

    feasible, reasons = row_set_feasibility(
        rows if rows is not None else set(np.where(mask.any(axis=1))[0]),
        structure.row_meta,
        config.get("coverage_constraints", {}),
    )
    retained_rows = int(np.count_nonzero(mask.any(axis=1)))
    retained_nnz = int(mask.sum())
    return {
        "mask": mask,
        "rows": sorted(rows) if rows is not None else None,
        "retained_rows": retained_rows,
        "retained_nnz": retained_nnz,
        "is_row_level": is_row or selector in {"full_support", "random_row"},
        "feasible": feasible,
        "reasons": reasons,
        "max_row_degree": int(row_nnz.max()),
        "avg_row_nnz": avg_row_nnz,
    }


# --------------------------------------------------------------------------- evaluation


def evaluate_support(
    structure: Structure,
    mask: np.ndarray,
    seed_data: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    split: str,
    physical_floor: float,
    support_floor: float,
) -> list[dict[str, Any]]:
    matrix = structure.matrix
    alpha = structure.alpha
    full_op = _ridge_filter_operator(matrix, alpha)
    sparse_op = _ridge_filter_operator(np.where(mask, matrix, 0.0), alpha)
    kept_rows = np.where(mask.any(axis=1))[0]
    gram = np.where(mask, matrix, 0.0)
    reg = gram.T @ gram + alpha * np.eye(matrix.shape[1])
    reg_cond = float(np.linalg.cond(reg))
    raw_rank = int(np.linalg.matrix_rank(np.where(mask, matrix, 0.0)))
    rows: list[dict[str, Any]] = []
    for seed, residual, truth in seed_data:
        x_full = full_op @ residual
        x_sparse = sparse_op @ residual
        resid_vec = np.where(mask, matrix, 0.0) @ x_sparse - residual
        weighted_residual = (
            float(np.linalg.norm(resid_vec[kept_rows])) if kept_rows.size else float("nan")
        )
        update_rmse_vs_full = float(
            np.linalg.norm(x_sparse - x_full) / max(np.linalg.norm(x_full), 1e-30)
        )
        update_rmse_vs_true = float(
            np.linalg.norm(x_sparse - truth) / max(np.linalg.norm(truth), 1e-30)
        )
        for fid, ell in zip(structure.functional_ids, structure.functionals, strict=True):
            y_true = float(ell @ truth)
            y_full = float(ell @ x_full)
            y_sparse = float(ell @ x_sparse)
            rows.append(
                {
                    "instance_id": structure.instance_id,
                    "group_id": structure.group_id,
                    "case": structure.case,
                    "realization": structure.realization,
                    "split": split,
                    "seed": int(seed),
                    "functional_id": fid,
                    "y_true": y_true,
                    "y_full_ridge": y_full,
                    "y_sparse": y_sparse,
                    "E_support_norm": abs(y_sparse - y_full) / max(abs(y_full), support_floor),
                    "E_physical_norm": abs(y_sparse - y_true) / max(abs(y_true), physical_floor),
                    "near_zero_y_true": bool(abs(y_true) < physical_floor),
                    "update_rmse_vs_full": update_rmse_vs_full,
                    "update_rmse_vs_true": update_rmse_vs_true,
                    "weighted_residual": weighted_residual,
                    "reg_condition_number": reg_cond,
                    "raw_rank": raw_rank,
                }
            )
    return rows


# --------------------------------------------------------------------------- statistics

ALL_SELECTORS = ("full_support", *ENTRY_SELECTORS, *ROW_SELECTORS)


def _paired_stats(diffs: np.ndarray, tie_tol: float = 1e-12) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank + sign test on paired effects (row - entry)."""

    from scipy.stats import binomtest, wilcoxon

    diffs = np.asarray(diffs, dtype=np.float64)
    wins = int(np.sum(diffs < -tie_tol))  # row better (lower E_physical)
    losses = int(np.sum(diffs > tie_tol))
    ties = int(np.sum(np.abs(diffs) <= tie_tol))
    informative = wins + losses
    wilcoxon_p = float("nan")
    nonzero = diffs[np.abs(diffs) > tie_tol]
    if nonzero.size >= 1:
        try:
            wilcoxon_p = float(
                wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = float("nan")
    sign_p = float(binomtest(wins, informative, 0.5).pvalue) if informative else float("nan")
    return {
        "n_groups": int(diffs.size),
        "wins_row_better": wins,
        "losses_row_worse": losses,
        "ties": ties,
        "mean_effect": float(np.mean(diffs)),
        "median_effect": float(np.median(diffs)),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
    }


def structural_group_statistics(
    per_group: pd.DataFrame,
    contrasts: list[dict[str, str]],
    budgets: list[int],
    *,
    protocol: str = "nnz",
) -> pd.DataFrame:
    """Per-contrast, per-budget paired statistics with structural groups as the unit.

    Selectors are paired at the same *target* budget (the fair-comparison quantity) within the given
    ``protocol``; ``per_group`` holds one held-out median E_physical per
    (group_id, case, selector, protocol, budget).
    """

    per_group = per_group[per_group["protocol"] == protocol]
    out: list[dict[str, Any]] = []
    for contrast in contrasts:
        name, row_sel, entry_sel = contrast["name"], contrast["row"], contrast["entry"]
        for budget in budgets:
            sub = per_group[per_group["budget"] == budget]
            groups = sorted(sub["group_id"].unique())
            diffs, cases = [], []
            for g in groups:
                gsub = sub[sub["group_id"] == g]
                r = _family_value(gsub, row_sel, ROW_SELECTORS)
                e = _family_value(gsub, entry_sel, ENTRY_SELECTORS)
                if r is None or e is None:
                    continue
                diffs.append(r - e)
                cases.append(gsub["case"].iloc[0])
            if not diffs:
                continue
            diffs = np.asarray(diffs)
            cases = np.asarray(cases)
            stats = _paired_stats(diffs)
            per_case = {
                f"mean_effect_{c}": float(np.mean(diffs[cases == c])) for c in sorted(set(cases))
            }
            loco = {
                f"loco_mean_excl_{c}": float(np.mean(diffs[cases != c])) for c in sorted(set(cases))
            }
            out.append(
                {
                    "contrast": name,
                    "budget_nnz": int(budget),
                    "row_selector": row_sel,
                    "entry_selector": entry_sel,
                    **stats,
                    **per_case,
                    **loco,
                    "interpretation": (
                        "negative mean/median effect = whole-row selection lower held-out physical "
                        "error "
                        "than entry selection at matched nonzero budget"
                    ),
                }
            )
    return pd.DataFrame(out)


def _family_value(gsub: pd.DataFrame, selector: str, family: tuple[str, ...]) -> float | None:
    """Held-out median E_physical for a named selector, or the family best (min) if a sentinel."""

    if selector in {"__best_row__", "__best_entry__"}:
        pool = gsub[gsub["selector"].isin(family)]
        return float(pool["heldout_median_E_physical"].min()) if not pool.empty else None
    rec = gsub[gsub["selector"] == selector]
    return float(rec["heldout_median_E_physical"].iloc[0]) if not rec.empty else None


# --------------------------------------------------------------------------- orchestrator


def _aggregate(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median E_physical (near-zero-truth excluded) + median E_support + means, per key group."""

    out = []
    for values, block in rows.groupby(keys, dropna=False):
        vals = values if isinstance(values, tuple) else (values,)
        rec = dict(zip(keys, vals, strict=True))
        physical = block[~block["near_zero_y_true"].astype(bool)]["E_physical_norm"]
        rec["median_E_physical"] = float(physical.median()) if not physical.empty else float("nan")
        rec["median_E_support"] = float(block["E_support_norm"].median())
        rec["mean_E_physical"] = float(physical.mean()) if not physical.empty else float("nan")
        rec["mean_update_rmse_vs_true"] = float(block["update_rmse_vs_true"].mean())
        rec["mean_weighted_residual"] = float(block["weighted_residual"].mean())
        rec["mean_reg_condition_number"] = float(block["reg_condition_number"].mean())
        rec["n_eval_rows"] = len(block)
        out.append(rec)
    return pd.DataFrame(out)


def run_row_sparsification(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"config study_id mismatch: {config.get('study_id')!r}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_root = config["source_structural_root"]
    physical_floor = float(config.get("physical_floor", 1e-6))
    support_floor = float(config.get("support_floor", 1e-6))

    structures = load_structures(source_root, config)
    protocol_a = [int(q) for q in config["budgets"]["protocol_a_row_counts"]]
    protocol_b = [int(k) for k in config["budgets"]["protocol_b_nnz"]]

    row_registry, selector_scores, support_registry = [], [], []
    raw_rows, infeasible = [], []

    for si, structure in enumerate(structures):
        for i, meta in enumerate(structure.row_meta):
            row_registry.append(
                {
                    "instance_id": structure.instance_id,
                    "group_id": structure.group_id,
                    "case": structure.case,
                    "row_index": i,
                    "measurement_label": meta.get("measurement_label"),
                    "measurement_type": meta.get("measurement_type"),
                    "row_nnz": int((structure.matrix[i] != 0.0).sum()),
                    "row_l2_norm": float(np.linalg.norm(structure.matrix[i])),
                }
            )
        train_tasks = [
            (r, ell)
            for (_s, r, _t) in structure.seed_data(structure.training_seeds)
            for ell in structure.functionals
        ]
        train_data = structure.seed_data(structure.training_seeds)
        heldout_data = structure.seed_data(structure.heldout_seeds)
        # record row-level scores for the task-aware selectors (provenance / test anchor)
        for agg in ("mean", "worst_case"):
            scores = row_sensitivity_scores(
                structure.matrix, train_tasks, structure.alpha, aggregation=agg
            )
            for i, s in enumerate(scores):
                selector_scores.append(
                    {
                        "instance_id": structure.instance_id,
                        "selector": f"row_sensitivity_{agg}",
                        "row_index": i,
                        "score": float(s),
                    }
                )

        for protocol, budgets in (("row_count", protocol_a), ("nnz", protocol_b)):
            for budget in budgets:
                for selector in ALL_SELECTORS:
                    support = build_support(
                        structure, selector, protocol, budget, train_tasks, config
                    )
                    reg = {
                        "instance_id": structure.instance_id,
                        "group_id": structure.group_id,
                        "case": structure.case,
                        "selector": selector,
                        "protocol": protocol,
                        "budget": int(budget),
                        "retained_rows": support["retained_rows"],
                        "retained_nnz": support["retained_nnz"],
                        "rows": json.dumps(support["rows"]),
                        "is_row_level": support["is_row_level"],
                        "feasible": support["feasible"],
                        "reasons": "|".join(support["reasons"]),
                    }
                    support_registry.append(reg)
                    if not support["feasible"]:
                        infeasible.append(reg)
                        continue
                    budget_nnz = support["retained_nnz"]
                    for split, seed_data in (("training", train_data), ("heldout", heldout_data)):
                        for r in evaluate_support(
                            structure,
                            support["mask"],
                            seed_data,
                            split=split,
                            physical_floor=physical_floor,
                            support_floor=support_floor,
                        ):
                            r.update(
                                {
                                    "selector": selector,
                                    "protocol": protocol,
                                    "budget": int(budget),
                                    "budget_nnz": int(budget_nnz),
                                    "retained_rows": support["retained_rows"],
                                    "is_row_level": support["is_row_level"],
                                }
                            )
                            raw_rows.append(r)
        if progress:
            print(f"[row_sparsification] structure {si + 1}/{len(structures)}", flush=True)

    raw = pd.DataFrame(raw_rows)
    atomic_write_csv(destination / "row_registry.csv", pd.DataFrame(row_registry))
    atomic_write_csv(destination / "selector_scores.csv", pd.DataFrame(selector_scores))
    atomic_write_csv(destination / "support_registry.csv", pd.DataFrame(support_registry))
    atomic_write_csv(destination / "raw_evaluation_rows.csv", raw)
    atomic_write_csv(
        destination / "infeasibility_registry.csv",
        pd.DataFrame(infeasible)
        if infeasible
        else pd.DataFrame(columns=["instance_id", "selector", "protocol", "budget", "reasons"]),
    )

    structure_summary = _aggregate(
        raw,
        [
            "instance_id",
            "group_id",
            "case",
            "selector",
            "protocol",
            "budget",
            "budget_nnz",
            "split",
            "is_row_level",
        ],
    )
    atomic_write_csv(destination / "structure_summary.csv", structure_summary)
    case_summary = _aggregate(raw, ["case", "selector", "protocol", "split", "budget_nnz"])
    atomic_write_csv(destination / "case_summary.csv", case_summary)

    # Per-group held-out E_physical (average the two realizations) keyed by nearest nnz budget.
    per_group = _per_group_heldout(structure_summary)
    contrasts = config.get(
        "contrasts",
        [
            {"name": "best_row_vs_best_entry", "row": "__best_row__", "entry": "__best_entry__"},
            {
                "name": "row_sensitivity_mean_vs_entry_sensitivity_mean",
                "row": "row_sensitivity_mean",
                "entry": "entry_sensitivity_mean",
            },
            {
                "name": "row_magnitude_vs_entry_magnitude",
                "row": "row_magnitude",
                "entry": "entry_global_magnitude",
            },
        ],
    )
    # Primary fair comparison = Protocol B (matched nonzero budget).
    stat_budgets = sorted(per_group[per_group["protocol"] == "nnz"]["budget"].unique().tolist())
    statistical_summary = structural_group_statistics(
        per_group, contrasts, stat_budgets, protocol="nnz"
    )
    atomic_write_csv(destination / "statistical_summary.csv", statistical_summary)

    resource = _resource_comparison(structure_summary)
    atomic_write_csv(destination / "resource_comparison.csv", resource)

    claim = _claim_support(structures, raw, statistical_summary, infeasible)
    atomic_write_json(destination / "claim_support.json", claim)
    atomic_write_json(
        destination / "run_manifest.json",
        provenance_block(config_path, config)
        | {"study_id": STUDY_ID, "structures": len(structures)},
    )
    _write_resolved_config(destination, config)
    _write_readme(destination, structures, raw, statistical_summary)
    write_manifest_and_checksums(
        destination, study_id=STUDY_ID, extra={"structures": len(structures), "raw_rows": len(raw)}
    )
    return {
        "structures": len(structures),
        "raw_rows": len(raw),
        "supports": len(support_registry),
        "infeasible": len(infeasible),
        "contrasts": statistical_summary["contrast"].nunique()
        if not statistical_summary.empty
        else 0,
    }


def _per_group_heldout(structure_summary: pd.DataFrame) -> pd.DataFrame:
    """One held-out value per (group, selector, protocol, target budget), averaging the two
    realizations so the structural group is the statistical unit (task 7.7)."""

    held = structure_summary[structure_summary["split"] == "heldout"].copy()
    out = []
    for keys, block in held.groupby(
        ["group_id", "case", "selector", "protocol", "budget"], dropna=False
    ):
        g, case, selector, protocol, budget = keys
        out.append(
            {
                "group_id": g,
                "case": case,
                "selector": selector,
                "protocol": protocol,
                "budget": int(budget),
                "heldout_median_E_physical": float(block["median_E_physical"].mean()),
                "heldout_median_E_support": float(block["median_E_support"].mean()),
                "median_retained_nnz": float(block["budget_nnz"].mean()),
            }
        )
    return pd.DataFrame(out)


def _resource_comparison(structure_summary: pd.DataFrame) -> pd.DataFrame:
    held = structure_summary[structure_summary["split"] == "heldout"]
    out = []
    for keys, block in held.groupby(["selector", "is_row_level", "budget_nnz"], dropna=False):
        selector, is_row, budget_nnz = keys
        out.append(
            {
                "selector": selector,
                "is_row_level": bool(is_row),
                "budget_nnz": int(budget_nnz),
                "median_retained_nnz": float(block["budget_nnz"].median()),
                "median_E_physical": float(block["median_E_physical"].median()),
                "median_E_support": float(block["median_E_support"].median()),
                "n_structures": int(block["instance_id"].nunique()),
            }
        )
    return pd.DataFrame(out).sort_values(["selector", "budget_nnz"])


def _write_resolved_config(destination: Path, config: dict[str, Any]) -> None:
    import yaml

    (destination / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )


def _claim_support(structures, raw, statistical_summary, infeasible) -> dict[str, Any]:
    best = statistical_summary[statistical_summary["contrast"] == "best_row_vs_best_entry"]
    row_better_any = bool((best["median_effect"] < 0).any()) if not best.empty else False
    return {
        "study_id": STUDY_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "structural_groups": int(
            pd.DataFrame([{"g": s.group_id} for s in structures])["g"].nunique()
        ),
        "instances": len(structures),
        "cases": sorted({s.case for s in structures}),
        "whole_row_retention_enforced": True,
        "truth_used_only_for_evaluation": True,
        "physical_and_support_metrics_separated": True,
        "structural_group_is_statistical_unit": True,
        "infeasible_configurations_retained": len(infeasible),
        "row_selection_lower_physical_error_at_some_budget": row_better_any,
        "allowed_claim": (
            "Whole-measurement-row selection provides a physically interpretable "
            "accuracy-resource tradeoff evaluated against arbitrary entry selection at matched "
            "nonzero budgets over distinct benchmark-derived structural groups; the direction and "
            "significance of any advantage are reported directly from the paired structural-group "
            "statistics without claiming a physically optimal support."
        ),
        "forbidden_claims": [
            "physically optimal support",
            "QSVT beats matched Ridge",
            "independent power systems",
            "quantum advantage",
        ],
    }


def _write_readme(destination: Path, structures, raw, statistical_summary) -> None:
    lines = [
        "# Workstream C - Measurement-Row-Level Sparsification",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- structures: {len(structures)} instances = "
        f"{len({s.group_id for s in structures})} structural groups x 2 realizations "
        f"({sorted({s.case for s in structures})})",
        f"- raw evaluation rows: {len(raw)}",
        "",
        "## Method",
        "Whole-row selection is the restriction of arbitrary entry selection to whole-row masks; "
        "both "
        "are evaluated through the identical Ridge path `ridge(H*mask, r, alpha)` and compared at "
        "matched nonzero (Protocol B) and row-count (Protocol A) budgets. Scores are built on "
        "training "
        "residuals only; the benchmark reference enters only the benchmark-error evaluation. Structural groups are "
        "the "
        "statistical unit (two realizations averaged within a group).",
        "",
        "## Files",
        "- `row_registry.csv`, `selector_scores.csv`, `support_registry.csv`",
        "- `raw_evaluation_rows.csv`, `structure_summary.csv`, `case_summary.csv`",
        "- `statistical_summary.csv` (paired Wilcoxon + sign test, per-case, LOCO, win/tie/loss)",
        "- `resource_comparison.csv` (accuracy-resource frontier), `infeasibility_registry.csv`",
        "- `claim_support.json`, `run_manifest.json`, `config_resolved.yaml`, `checksums.sha256`",
        "",
        "## Reproduce",
        "```",
        "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 \\",
        "  .venv/bin/python scripts/run_tqe_measurement_row_sparsification.py",
        "```",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
