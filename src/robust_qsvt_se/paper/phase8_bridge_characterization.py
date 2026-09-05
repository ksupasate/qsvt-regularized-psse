"""Phase 8: selected-submatrix / full-system bridge characterization.

Extends the three-row bridge audit into a characterized boundary result: for
generated IEEE/PYPOWER weighted PSSE systems, block sizes ``b`` in
{4, 8, 16, 32, 64}, and three block-selection rules, it compares the first
selected coordinate of the full-system Ridge update with the corresponding
selected-block Ridge update at the same ``alpha``:

    Delta_l = | l^T dx_full - l_B^T dx_B |,   l_B = e_1,
    rel     = Delta_l / max(|l^T dx_full|, eps).

Selection rules (all numerical; none claims physical optimality):

* ``largest_row_col_norms`` -- the executed provenance rule,
* ``column_leverage``       -- deterministic SVD leverage-score ranking,
* ``seeded_random``         -- seeded uniform control draw.

Each row also records deleted-coupling diagnostics (out-of-block energy of the
selected rows/columns, functional-column leakage, block Frobenius and residual
energy fractions) plus block conditioning, so the discrepancy can be read as a
function of how much coupling the surrogate discards.  The three original
bridge rows are reproduced verbatim as ``provenance_bridge_row`` entries.

This is a purely classical audit of the surrogate boundary.  It does not make
the selected-submatrix circuits full-system selected-output evaluations, and it
claims no speedup and no quantum execution beyond the existing workloads.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_block_bridge import build_bridge_rows
from robust_qsvt_se.paper.selected_observable_qsvt_common import array_checksum
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    assert_safe,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.utils.io import ensure_directory

PHASE8_BRIDGE_DIR = Path("outputs/phase8_bridge_characterization")
DEFAULT_TABLE_PATH = Path("manuscript/tables/phase8_bridge_characterization.tex")

CASES = ("ieee14", "ieee30", "ieee57", "ieee118")
BLOCK_SIZES = (4, 8, 16, 32, 64)
SELECTION_RULES = ("largest_row_col_norms", "column_leverage", "seeded_random")
RANK_CUTOFF_RELATIVE = 1.0e-10
RELATIVE_EPS = float(np.finfo(np.float64).tiny)

FULL_COLUMNS = [
    "case_name",
    "full_shape",
    "block_size",
    "selection_rule",
    "status",
    "skipped_with_reason",
    "selected_rows",
    "selected_cols",
    "lambda_alpha_over_block_beta2",
    "alpha",
    "sigma_min",
    "sigma_max",
    "kappa_effective",
    "numerical_rank",
    "full_selected_functional",
    "block_selected_functional",
    "absolute_discrepancy_delta_l",
    "relative_discrepancy_vs_full",
    "selected_coordinate_vector_relative_discrepancy",
    "out_of_block_coupling_fraction",
    "selected_rows_out_of_block_energy_fraction",
    "selected_cols_out_of_block_energy_fraction",
    "functional_column_leakage",
    "block_frobenius_fraction",
    "residual_energy_fraction",
    "block_checksum",
    "full_matrix_checksum",
    "magnitude_class",
    "interpretation",
]

DIAGNOSTIC_COLUMNS = [
    "out_of_block_coupling_fraction",
    "selected_rows_out_of_block_energy_fraction",
    "selected_cols_out_of_block_energy_fraction",
    "functional_column_leakage",
    "block_frobenius_fraction",
    "residual_energy_fraction",
    "kappa_effective",
]

INTERPRETATION = (
    "selected-submatrix surrogate; not the selected functional of the full-system Ridge update"
)


def _leverage_block(
    H: np.ndarray, r: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic SVD leverage-score selection (numerical rule, rank-truncated)."""

    U, singular_values, Vt = np.linalg.svd(np.asarray(H, dtype=np.float64), full_matrices=False)
    rank = int(np.sum(singular_values > RANK_CUTOFF_RELATIVE * singular_values[0]))
    col_scores = np.sum(Vt[:rank, :] ** 2, axis=0)
    row_scores = np.sum(U[:, :rank] ** 2, axis=1)
    cols = _top_indices(col_scores, size)
    rows = _top_indices(row_scores, size)
    return H[np.ix_(rows, cols)], np.asarray(r, dtype=np.float64)[rows], rows, cols


def _random_block(
    H: np.ndarray, r: np.ndarray, size: int, *, seed_sequence: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(list(seed_sequence))
    rows = np.sort(rng.choice(H.shape[0], size=size, replace=False)).astype(np.int64)
    cols = np.sort(rng.choice(H.shape[1], size=size, replace=False)).astype(np.int64)
    return H[np.ix_(rows, cols)], np.asarray(r, dtype=np.float64)[rows], rows, cols


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:count]).astype(np.int64)


def _coupling_diagnostics(
    H: np.ndarray, r: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> dict[str, float]:
    row_mask = np.zeros(H.shape[0], dtype=bool)
    row_mask[rows] = True
    col_mask = np.zeros(H.shape[1], dtype=bool)
    col_mask[cols] = True

    block_sq = float(np.sum(H[np.ix_(rows, cols)] ** 2))
    rows_all_sq = float(np.sum(H[rows, :] ** 2))
    cols_all_sq = float(np.sum(H[:, cols] ** 2))
    rows_out_sq = float(np.sum(H[np.ix_(rows, ~col_mask)] ** 2))
    cols_out_sq = float(np.sum(H[np.ix_(~row_mask, cols)] ** 2))
    functional_col = int(cols[0])
    col_total_sq = float(np.sum(H[:, functional_col] ** 2))
    col_out_sq = float(np.sum(H[~row_mask, functional_col] ** 2))

    def _fraction(part: float, whole: float) -> float:
        return part / whole if whole > 0.0 else float("nan")

    return {
        "out_of_block_coupling_fraction": _fraction(
            rows_out_sq + cols_out_sq, rows_all_sq + cols_all_sq
        ),
        "selected_rows_out_of_block_energy_fraction": _fraction(rows_out_sq, rows_all_sq),
        "selected_cols_out_of_block_energy_fraction": _fraction(cols_out_sq, cols_all_sq),
        "functional_column_leakage": _fraction(col_out_sq, col_total_sq),
        "block_frobenius_fraction": _fraction(block_sq, float(np.sum(H**2))),
        "residual_energy_fraction": _fraction(
            float(np.sum(np.asarray(r, dtype=np.float64)[rows] ** 2)),
            float(np.sum(np.asarray(r, dtype=np.float64) ** 2)),
        ),
    }


def _magnitude_class(relative: float) -> str:
    if not np.isfinite(relative):
        return "undefined"
    if relative < 0.05:
        return "below_0.05"
    if relative < 0.5:
        return "0.05_to_0.5"
    return "above_0.5"


def characterize_bridge(
    *, seed: int = 123, cases: tuple[str, ...] = CASES
) -> tuple[pd.DataFrame, float]:
    """Build the full characterization frame; returns ``(frame, anchor_lambda)``."""

    anchor_system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    anchor_block, _, _, _ = select_deterministic_block(
        np.asarray(anchor_system.H_tilde),
        np.asarray(anchor_system.r_tilde),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    anchor_singular = np.linalg.svd(anchor_block, compute_uv=False)
    anchor_lambda = 4.0 * float(anchor_singular[-1]) ** 2 / float(anchor_singular[0]) ** 2

    rows: list[dict[str, Any]] = []
    for case_index, case_name in enumerate(cases):
        system, _ = build_engineering_system(
            {
                "case_name": case_name,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        full = np.asarray(system.H_tilde, dtype=np.float64)
        residual = np.asarray(system.r_tilde, dtype=np.float64)
        full_checksum = array_checksum(full)
        full_shape = f"{full.shape[0]}x{full.shape[1]}"
        for block_size in BLOCK_SIZES:
            for rule in SELECTION_RULES:
                base = {
                    "case_name": case_name,
                    "full_shape": full_shape,
                    "block_size": int(block_size),
                    "selection_rule": rule,
                    "full_matrix_checksum": full_checksum,
                    "lambda_alpha_over_block_beta2": anchor_lambda,
                    "interpretation": INTERPRETATION,
                }
                if block_size > min(full.shape):
                    rows.append(
                        {
                            **base,
                            "status": "skipped",
                            "skipped_with_reason": (
                                f"block size {block_size} exceeds min dimension "
                                f"{min(full.shape)} of the {full_shape} weighted Jacobian"
                            ),
                        }
                    )
                    continue
                if rule == "largest_row_col_norms":
                    block, block_residual, sel_rows, sel_cols = select_deterministic_block(
                        full,
                        residual,
                        row_count=block_size,
                        col_count=block_size,
                        policy="largest_row_col_norms",
                    )
                elif rule == "column_leverage":
                    block, block_residual, sel_rows, sel_cols = _leverage_block(
                        full, residual, block_size
                    )
                else:
                    block, block_residual, sel_rows, sel_cols = _random_block(
                        full,
                        residual,
                        block_size,
                        seed_sequence=(int(seed), case_index, int(block_size)),
                    )
                beta = float(np.linalg.norm(block, 2))
                if beta <= 1.0e-12:
                    rows.append(
                        {
                            **base,
                            "selected_rows": " ".join(str(int(v)) for v in sel_rows),
                            "selected_cols": " ".join(str(int(v)) for v in sel_cols),
                            "status": "skipped",
                            "skipped_with_reason": (
                                "selected block is numerically zero (sigma_max <= 1e-12); "
                                "alpha = lambda*beta^2 is undefined at fixed lambda"
                            ),
                        }
                    )
                    continue
                rows.append(
                    {
                        **base,
                        **_bridge_metrics(
                            full,
                            residual,
                            block,
                            block_residual,
                            sel_rows,
                            sel_cols,
                            lam=anchor_lambda,
                        ),
                        "status": "computed",
                        "skipped_with_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=FULL_COLUMNS)
    return frame, anchor_lambda


def _bridge_metrics(
    full: np.ndarray,
    residual: np.ndarray,
    block: np.ndarray,
    block_residual: np.ndarray,
    sel_rows: np.ndarray,
    sel_cols: np.ndarray,
    *,
    lam: float,
) -> dict[str, Any]:
    singular = np.linalg.svd(block, compute_uv=False)
    beta = float(singular[0])
    alpha = float(lam) * beta**2
    positive = singular[singular > RANK_CUTOFF_RELATIVE * beta]
    rank = int(positive.size)
    kappa = float(positive.max() / positive.min()) if rank > 0 else float("inf")

    full_update = ridge_svd_solution(full, residual, alpha=alpha)
    block_update = ridge_svd_solution(block, block_residual, alpha=alpha)
    full_value = float(full_update[int(sel_cols[0])])
    block_value = float(block_update[0])
    absolute = abs(full_value - block_value)
    relative = absolute / max(abs(full_value), RELATIVE_EPS)
    vector_relative = float(
        np.linalg.norm(full_update[sel_cols] - block_update)
        / max(np.linalg.norm(full_update[sel_cols]), RELATIVE_EPS)
    )
    return {
        "selected_rows": " ".join(str(int(v)) for v in sel_rows),
        "selected_cols": " ".join(str(int(v)) for v in sel_cols),
        "alpha": alpha,
        "sigma_min": float(singular[-1]),
        "sigma_max": beta,
        "kappa_effective": kappa,
        "numerical_rank": rank,
        "full_selected_functional": full_value,
        "block_selected_functional": block_value,
        "absolute_discrepancy_delta_l": absolute,
        "relative_discrepancy_vs_full": relative,
        "selected_coordinate_vector_relative_discrepancy": vector_relative,
        **_coupling_diagnostics(full, residual, sel_rows, sel_cols),
        "block_checksum": array_checksum(block),
        "magnitude_class": _magnitude_class(relative),
    }


def _provenance_rows(seed: int) -> pd.DataFrame:
    """The three original bridge rows, reproduced verbatim for continuity."""

    rows = build_bridge_rows(seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "case_name": row["case_name"],
                "full_shape": row["full_shape"],
                "block_size": int(str(row["block_shape"]).split("x")[0]),
                "selection_rule": "provenance_bridge_row",
                "status": "computed",
                "skipped_with_reason": "",
                "selected_rows": row["selected_rows"],
                "selected_cols": row["selected_cols"],
                "lambda_alpha_over_block_beta2": row["lambda_alpha_over_block_beta2"],
                "alpha": row["alpha"],
                "full_selected_functional": row["full_selected_functional"],
                "block_selected_functional": row["block_selected_functional"],
                "absolute_discrepancy_delta_l": row["absolute_discrepancy_delta_l"],
                "relative_discrepancy_vs_full": row["relative_discrepancy_vs_full"],
                "selected_coordinate_vector_relative_discrepancy": row[
                    "selected_coordinate_vector_relative_discrepancy"
                ],
                "block_checksum": row["block_checksum"],
                "full_matrix_checksum": row["full_matrix_checksum"],
                "magnitude_class": _magnitude_class(float(row["relative_discrepancy_vs_full"])),
                "interpretation": row["interpretation"],
            }
        )
    return pd.DataFrame(records, columns=FULL_COLUMNS)


def _rule_summary(computed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case_name, rule), block in computed.groupby(["case_name", "selection_rule"], sort=True):
        rel = block["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "case_name": case_name,
                "selection_rule": rule,
                "num_blocks": len(block),
                "median_relative_discrepancy": float(np.median(rel)),
                "min_relative_discrepancy": float(np.min(rel)),
                "max_relative_discrepancy": float(np.max(rel)),
                "median_out_of_block_coupling_fraction": float(
                    block["out_of_block_coupling_fraction"].median()
                ),
                "median_functional_column_leakage": float(
                    block["functional_column_leakage"].median()
                ),
            }
        )
    for rule, block in computed.groupby("selection_rule", sort=True):
        rel = block["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "case_name": "__all_cases__",
                "selection_rule": rule,
                "num_blocks": len(block),
                "median_relative_discrepancy": float(np.median(rel)),
                "min_relative_discrepancy": float(np.min(rel)),
                "max_relative_discrepancy": float(np.max(rel)),
                "median_out_of_block_coupling_fraction": float(
                    block["out_of_block_coupling_fraction"].median()
                ),
                "median_functional_column_leakage": float(
                    block["functional_column_leakage"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    rx = pd.Series(x[finite]).rank().to_numpy(dtype=np.float64)
    ry = pd.Series(y[finite]).rank().to_numpy(dtype=np.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    return float(np.sum(rx * ry) / denominator) if denominator > 0.0 else float("nan")


def _diagnostic_correlations(computed: pd.DataFrame) -> dict[str, float]:
    rel = computed["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)
    return {
        column: _spearman(computed[column].to_numpy(dtype=np.float64), rel)
        for column in DIAGNOSTIC_COLUMNS
    }


def _sci(value: float) -> str:
    if not np.isfinite(value) or value == 0.0:
        return "$0$" if value == 0.0 else "--"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


_MAGNITUDE_TEX = {
    "below_0.05": "$<0.05$",
    "0.05_to_0.5": "0.05--0.5",
    "above_0.5": "$>0.5$",
    "undefined": "--",
}


def _table_tex(computed: pd.DataFrame, rule_summary: pd.DataFrame) -> str:
    norm_rows = computed[computed["selection_rule"] == "largest_row_col_norms"].sort_values(
        ["case_name", "block_size"]
    )
    lines = [
        "% Source: outputs/phase8_bridge_characterization/bridge_characterization_full.csv",
        "% Regenerate: .venv/bin/python scripts/run_phase8_bridge_characterization.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Bridge characterization: first-coordinate relative discrepancy between the "
        r"selected-block and full-system Ridge updates at the same $\alpha$ "
        r"($\lambda=\alpha/\beta_B^2$ fixed at the executed anchor value), for the executed "
        r"norm-based selection rule. ``Leak'' is the functional-column leakage (energy "
        r"fraction of the functional's Jacobian column outside the retained rows); "
        r"``coupling'' is the out-of-block energy fraction of the selected rows/columns. "
        r"Medians over block sizes for all three rules appear below the rule; the "
        r"seeded-random control gives the no-structure baseline. All rows are surrogate "
        r"boundary audits, not full-system selected-output evaluations.}",
        r"\label{tab:phase8_bridge_characterization}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Case & $b$ & $\ell^T\Delta x_{\rm full}$ & $\ell_B^T\Delta x_B$ & rel.\ $\Delta_\ell$ "
        r"& coupling & leak & class \\",
        r"\hline",
    ]
    for _, row in norm_rows.iterrows():
        magnitude_label = _MAGNITUDE_TEX.get(str(row["magnitude_class"]), "--")
        lines.append(
            f"{row['case_name']} & {int(row['block_size'])} & "
            f"{_sci(float(row['full_selected_functional']))} & "
            f"{_sci(float(row['block_selected_functional']))} & "
            f"{float(row['relative_discrepancy_vs_full']):.3f} & "
            f"{float(row['out_of_block_coupling_fraction']):.2f} & "
            f"{float(row['functional_column_leakage']):.2f} & "
            f"{magnitude_label} \\\\"
        )
    lines += [
        r"\hline",
        r"\multicolumn{8}{@{}l@{}}{\footnotesize Median rel.\ $\Delta_\ell$ over all "
        r"computed case/size cells, per rule:} \\",
    ]
    overall = rule_summary[rule_summary["case_name"] == "__all_cases__"]
    for _, row in overall.iterrows():
        rule_label = str(row["selection_rule"]).replace("_", r"\_")
        lines.append(
            rf"\multicolumn{{8}}{{@{{}}l@{{}}}}{{\footnotesize \quad {rule_label}: "
            f"{float(row['median_relative_discrepancy']):.3f} "
            f"(min {float(row['min_relative_discrepancy']):.3f}, "
            f"max {float(row['max_relative_discrepancy']):.3f}, "
            f"{int(row['num_blocks'])} blocks)}} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _readme(
    computed: pd.DataFrame,
    rule_summary: pd.DataFrame,
    correlations: dict[str, float],
    anchor_lambda: float,
) -> str:
    provenance = computed[computed["selection_rule"] == "provenance_bridge_row"]
    ieee30_norm = computed[
        (computed["case_name"] == "ieee30")
        & (computed["selection_rule"] == "largest_row_col_norms")
        & (computed["block_size"] == 16)
    ]
    lines = [
        "# Phase 8: Bridge-Discrepancy Characterization",
        "",
        "Classical audit of the selected-submatrix surrogate boundary across IEEE cases, "
        "block sizes, and selection rules at the executed normalized regularization "
        f"lambda = {anchor_lambda:.6f} (alpha = lambda * beta_B^2 per block). The three "
        "original bridge rows are reproduced verbatim as `provenance_bridge_row` entries.",
        "",
        "## Rule-level medians (relative first-coordinate discrepancy)",
        "",
        "| Case | Rule | blocks | median | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rule_summary.iterrows():
        lines.append(
            f"| {row['case_name']} | {row['selection_rule']} | {int(row['num_blocks'])} | "
            f"{row['median_relative_discrepancy']:.3f} | {row['min_relative_discrepancy']:.3f} "
            f"| {row['max_relative_discrepancy']:.3f} |"
        )
    lines += [
        "",
        "## Spearman correlation of the relative discrepancy with deleted-coupling "
        "diagnostics (computed rows, all rules)",
        "",
        "| Diagnostic | Spearman rho |",
        "| --- | --- |",
    ]
    for name, value in correlations.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += [
        "",
        "## Mechanism reading",
        "",
        "The selected-submatrix surrogate discards out-of-block couplings. The discrepancy "
        "increases when the selected functional depends strongly on variables or equations "
        "outside the retained block, so the bridge result should be interpreted as a boundary "
        "on the selected-submatrix surrogate, not as evidence of full-system selected-output "
        "accuracy. The IEEE-30 16x16 provenance row "
        f"(relative discrepancy {float(provenance.iloc[-1]['relative_discrepancy_vs_full']):.3f}"
        ") sits in the same regime as the corresponding characterized cell "
        f"({float(ieee30_norm.iloc[0]['relative_discrepancy_vs_full']):.3f} at the anchor "
        "lambda) and its functional-column leakage "
        f"{float(ieee30_norm.iloc[0]['functional_column_leakage']):.2f} shows most of that "
        "column's energy lies outside the retained rows.",
        "",
        "No speedup, no quantum execution, and no full-system selected-output claim is made; "
        "this is a boundary audit of the surrogate construction.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase8_bridge_characterization(config: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(config or {})
    output_dir = ensure_directory(Path(options.get("output_dir", PHASE8_BRIDGE_DIR)))
    table_path = Path(options.get("table_path", DEFAULT_TABLE_PATH))
    seed = int(options.get("seed", 123))
    cases = tuple(options.get("cases", CASES))

    sweep, anchor_lambda = characterize_bridge(seed=seed, cases=cases)
    provenance = _provenance_rows(seed)
    full_frame = pd.concat([sweep, provenance], ignore_index=True)[FULL_COLUMNS]
    computed = full_frame[full_frame["status"] == "computed"].copy()
    sweep_computed = computed[computed["selection_rule"] != "provenance_bridge_row"].copy()
    rule_summary = _rule_summary(sweep_computed)
    correlations = _diagnostic_correlations(sweep_computed)

    full_csv = output_dir / "bridge_characterization_full.csv"
    by_cell_csv = output_dir / "bridge_characterization_by_case_size_rule.csv"
    summary_csv = output_dir / "bridge_characterization_summary.csv"
    readme_md = output_dir / "README.md"

    full_frame.to_csv(full_csv, index=False)
    by_cell = sweep_computed[
        [
            "case_name",
            "block_size",
            "selection_rule",
            "alpha",
            "full_selected_functional",
            "block_selected_functional",
            "relative_discrepancy_vs_full",
            "selected_coordinate_vector_relative_discrepancy",
            "out_of_block_coupling_fraction",
            "functional_column_leakage",
            "kappa_effective",
            "magnitude_class",
        ]
    ]
    by_cell.to_csv(by_cell_csv, index=False)
    rule_summary.to_csv(summary_csv, index=False)
    readme_md.write_text(
        _readme(computed, rule_summary, correlations, anchor_lambda), encoding="utf-8"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_table_tex(sweep_computed, rule_summary), encoding="utf-8")

    artifacts = {
        "bridge_characterization_full_csv": full_csv,
        "bridge_characterization_by_case_size_rule_csv": by_cell_csv,
        "bridge_characterization_summary_csv": summary_csv,
        "readme_md": readme_md,
        "manuscript_table_tex": table_path,
    }
    checksum_path = output_dir / "checksums.sha256"
    local_artifacts = [
        path
        for _, path in sorted(artifacts.items(), key=lambda item: item[1].name)
        if path.parent.resolve() == output_dir.resolve()
    ]
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in local_artifacts),
        encoding="utf-8",
    )
    artifacts["checksums_sha256"] = checksum_path
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="phase8_bridge_characterization",
        script_name="scripts/run_phase8_bridge_characterization.py",
        command=str(options.get("command", "run_phase8_bridge_characterization")),
        description=(
            "Classical characterization of the selected-submatrix / full-system Ridge bridge "
            "discrepancy across IEEE cases, block sizes 4-64, and three numerical selection "
            "rules at the executed anchor lambda, with deleted-coupling diagnostics and the "
            "three original bridge rows reproduced verbatim."
        ),
        artifacts=artifacts,
        inputs_used=[f"build_engineering_system:{case}:weighted_jacobian" for case in cases],
        random_seeds={
            "system_seed": seed,
            "random_rule_seed_sequence": "[seed, case_index, block_size]",
        },
        warnings=[],
        failures=[],
        interpretation_boundary=(
            "A classical audit of the surrogate boundary: it quantifies when the "
            "selected-block first-coordinate functional tracks the full-system Ridge "
            "functional and when it does not. It does not make the executed circuits "
            "full-system selected-output evaluations and claims no quantum execution beyond "
            "the existing selected-submatrix workloads."
        ),
        extra={
            "anchor_lambda": anchor_lambda,
            "cases": list(cases),
            "block_sizes": list(BLOCK_SIZES),
            "selection_rules": list(SELECTION_RULES),
            "diagnostic_spearman_correlations": correlations,
            "manifest_name": "bridge_characterization_manifest.json",
        },
        manifest_name="bridge_characterization_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "full_frame": full_frame,
        "rule_summary": rule_summary,
        "correlations": correlations,
        "anchor_lambda": anchor_lambda,
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 8: bridge-discrepancy characterization")
    parser.add_argument("--output-dir", default=str(PHASE8_BRIDGE_DIR))
    parser.add_argument("--table-path", default=str(DEFAULT_TABLE_PATH))
    args = parser.parse_args(argv)
    run = run_phase8_bridge_characterization(
        {
            "output_dir": args.output_dir,
            "table_path": args.table_path,
            "command": "scripts/run_phase8_bridge_characterization.py " + " ".join(argv or []),
        }
    )
    print(f"Bridge characterization complete: {run['artifacts']['manifest']}")
    print(run["rule_summary"].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
