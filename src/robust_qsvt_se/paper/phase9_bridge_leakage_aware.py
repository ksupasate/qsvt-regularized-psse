"""Phase 9: leakage-aware selected-submatrix / full-system bridge audit.

Adds a fourth, deterministic *leakage-aware* block-selection rule to the Phase 8
bridge characterization and tests the hypothesis that the selected-submatrix
surrogate discrepancy tracks discarded functional-column coupling: when the
block is chosen to keep most of the target functional's Jacobian column inside
the retained rows, the first-coordinate discrepancy between the selected-block
Ridge update and the full-system Ridge update should shrink.

Rules compared (all numerical, all deterministic except the seeded-random
control), reusing the Phase 8 metric machinery verbatim:

* ``largest_row_col_norms`` -- the executed provenance rule,
* ``column_leverage``       -- SVD leverage-score ranking,
* ``seeded_random``         -- seeded uniform control,
* ``leakage_aware``         -- NEW: target column = global max column-norm; retain
  the rows carrying most of that column's energy (minimizing functional-column
  leakage) and the remaining columns best supported by those rows.

The leakage-aware rule uses only the weighted Jacobian, its row/column norms, the
target functional index, and within-block energy -- never the solved full-system
Ridge update or the post-solve selected value.  This remains a purely classical
audit of the surrogate boundary: it does not make the selected-submatrix circuits
full-system selected-output evaluations and claims no quantum execution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase8_bridge_characterization import (
    _bridge_metrics,
    _leverage_block,
    _random_block,
    _spearman,
    _top_indices,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import array_checksum
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    assert_safe,
    get_pyplot,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

PHASE9_BRIDGE_DIR = Path("outputs/phase9_bridge_leakage_aware")
CASES = ("ieee14", "ieee30", "ieee57", "ieee118")
BLOCK_SIZES = (4, 8, 16, 32, 64)
SELECTION_RULES = ("largest_row_col_norms", "column_leverage", "seeded_random", "leakage_aware")

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
    "functional_column_leakage",
    "out_of_block_coupling_fraction",
    "selected_rows_out_of_block_energy_fraction",
    "selected_cols_out_of_block_energy_fraction",
    "kappa_effective",
]

INTERPRETATION = (
    "selected-submatrix surrogate; not the selected functional of the full-system Ridge update"
)


def leakage_aware_block(
    H: np.ndarray, r: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic leakage-aware block selection (pre-solve; no post-solve inputs).

    Target functional = the global maximum column-norm column (placed first, so the
    in-block functional ``e_1`` recovers it).  Retain the ``size`` rows carrying the
    most energy of that target column (minimizing functional-column leakage) and the
    remaining ``size - 1`` columns best supported by the retained rows.
    """

    H = np.asarray(H, dtype=np.float64)
    column_norms = np.linalg.norm(H, axis=0)
    target_col = int(np.argmax(column_norms))

    row_scores = np.abs(H[:, target_col])
    rows = _top_indices(row_scores, size)

    within_row_energy = np.sum(H[rows, :] ** 2, axis=0)
    within_row_energy[target_col] = -np.inf  # exclude the target from the "others" ranking
    others = _top_indices(within_row_energy, size - 1)
    cols = np.concatenate(([target_col], others[others != target_col]))[:size].astype(np.int64)

    return H[np.ix_(rows, cols)], np.asarray(r, dtype=np.float64)[rows], rows, cols


def _anchor_lambda(seed: int) -> float:
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    block, _r, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde),
        np.asarray(system.r_tilde),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    singular = np.linalg.svd(block, compute_uv=False)
    return 4.0 * float(singular[-1]) ** 2 / float(singular[0]) ** 2


def _select_block(
    rule: str,
    full: np.ndarray,
    residual: np.ndarray,
    size: int,
    *,
    seed: int,
    case_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rule == "largest_row_col_norms":
        return select_deterministic_block(
            full, residual, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
    if rule == "column_leverage":
        return _leverage_block(full, residual, size)
    if rule == "leakage_aware":
        return leakage_aware_block(full, residual, size)
    return _random_block(full, residual, size, seed_sequence=(int(seed), case_index, int(size)))


def characterize_leakage_aware_bridge(
    *, seed: int = 123, cases: tuple[str, ...] = CASES
) -> tuple[pd.DataFrame, float]:
    """Full four-rule bridge frame; returns ``(frame, anchor_lambda)``."""

    anchor_lambda = _anchor_lambda(seed)
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
                block, block_residual, sel_rows, sel_cols = _select_block(
                    rule, full, residual, block_size, seed=seed, case_index=case_index
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
                                "selected block is numerically zero (sigma_max <= 1e-12)"
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


def _rule_group_summary(computed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule, block in computed.groupby("selection_rule", sort=True):
        rel = block["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)
        leak = block["functional_column_leakage"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "selection_rule": rule,
                "num_blocks": len(block),
                "median_relative_discrepancy": float(np.nanmedian(rel)),
                "min_relative_discrepancy": float(np.nanmin(rel)),
                "max_relative_discrepancy": float(np.nanmax(rel)),
                "median_functional_column_leakage": float(np.nanmedian(leak)),
                "median_out_of_block_coupling_fraction": float(
                    np.nanmedian(block["out_of_block_coupling_fraction"].to_numpy(dtype=np.float64))
                ),
                "fraction_below_0.05": float(np.mean(rel < 0.05)),
            }
        )
    return pd.DataFrame(rows).sort_values("median_relative_discrepancy").reset_index(drop=True)


def _diagnostic_correlations(computed: pd.DataFrame) -> dict[str, float]:
    rel = computed["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)
    return {
        column: _spearman(computed[column].to_numpy(dtype=np.float64), rel)
        for column in DIAGNOSTIC_COLUMNS
    }


def _scatter_figure(computed: pd.DataFrame, output_dir: Path) -> Path:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    rules = list(SELECTION_RULES)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(rules)))
    for rule, color in zip(rules, colors, strict=True):
        block = computed[computed["selection_rule"] == rule]
        ax.scatter(
            block["functional_column_leakage"].to_numpy(dtype=np.float64),
            block["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64),
            s=28,
            color=color,
            alpha=0.8,
            edgecolors="none",
            label=rule,
        )
    ax.set_xlabel("functional-column leakage (target column energy outside retained rows)")
    ax.set_ylabel("relative first-coordinate discrepancy vs full-system Ridge")
    ax.set_yscale("symlog", linthresh=1.0e-2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Bridge discrepancy vs functional-column leakage (all rules)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "discrepancy_vs_leakage_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase9_bridge_leakage_aware(config: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(config or {})
    output_dir = ensure_directory(Path(options.get("output_dir", PHASE9_BRIDGE_DIR)))
    seed = int(options.get("seed", 123))
    cases = tuple(options.get("cases", CASES))

    full_frame, anchor_lambda = characterize_leakage_aware_bridge(seed=seed, cases=cases)
    computed = full_frame[full_frame["status"] == "computed"].copy()
    rule_summary = _rule_group_summary(computed)
    correlations = _diagnostic_correlations(computed)

    full_csv = output_dir / "bridge_leakage_aware_full.csv"
    summary_csv = output_dir / "bridge_leakage_aware_rule_summary.csv"
    by_cell_csv = output_dir / "bridge_leakage_aware_by_case_size_rule.csv"
    readme_md = output_dir / "README.md"

    full_frame.to_csv(full_csv, index=False)
    rule_summary.to_csv(summary_csv, index=False)
    computed[
        [
            "case_name",
            "block_size",
            "selection_rule",
            "full_selected_functional",
            "block_selected_functional",
            "relative_discrepancy_vs_full",
            "selected_coordinate_vector_relative_discrepancy",
            "functional_column_leakage",
            "out_of_block_coupling_fraction",
            "kappa_effective",
            "magnitude_class",
        ]
    ].to_csv(by_cell_csv, index=False)
    scatter_path = _scatter_figure(computed, output_dir)
    readme_md.write_text(
        _readme(computed, rule_summary, correlations, anchor_lambda), encoding="utf-8"
    )

    artifacts = {
        "bridge_leakage_aware_full_csv": full_csv,
        "bridge_leakage_aware_rule_summary_csv": summary_csv,
        "bridge_leakage_aware_by_case_size_rule_csv": by_cell_csv,
        "discrepancy_vs_leakage_scatter_png": scatter_path,
        "readme_md": readme_md,
    }
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for _, path in sorted(artifacts.items(), key=lambda item: item[1].name)
        ),
        encoding="utf-8",
    )
    artifacts["checksums_sha256"] = checksum_path

    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="phase9_bridge_leakage_aware",
        script_name="scripts/run_phase9_bridge_leakage_aware.py",
        command=str(options.get("command", "run_phase9_bridge_leakage_aware")),
        description=(
            "Leakage-aware selected-submatrix / full-system Ridge bridge audit: adds a "
            "deterministic leakage-aware block-selection rule (target column retained, "
            "functional-column leakage minimized) to the three existing rules and tests "
            "whether lower functional-column leakage reduces the first-coordinate surrogate "
            "discrepancy, across IEEE cases and block sizes 4-64."
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
            "A classical audit of the surrogate boundary. Lower leakage improves surrogate "
            "fidelity only when the selected functional is mostly supported by retained rows "
            "and columns; it does not prove equivalence to the full PSSE Ridge update. It "
            "helps define when a selected-submatrix workload is a meaningful surrogate, and "
            "claims no quantum execution beyond the existing selected-submatrix workloads."
        ),
        extra={
            "anchor_lambda": anchor_lambda,
            "cases": list(cases),
            "block_sizes": list(BLOCK_SIZES),
            "selection_rules": list(SELECTION_RULES),
            "diagnostic_spearman_correlations": correlations,
            "leakage_aware_rule_definition": (
                "target column = argmax full-system column norm; rows = top-size |H[:,target]|; "
                "other columns = top within-retained-row energy; deterministic, pre-solve"
            ),
        },
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


def _readme(
    computed: pd.DataFrame,
    rule_summary: pd.DataFrame,
    correlations: dict[str, float],
    anchor_lambda: float,
) -> str:
    leak_rho = correlations.get("functional_column_leakage", float("nan"))
    leakage_row = rule_summary[rule_summary["selection_rule"] == "leakage_aware"]
    norm_row = rule_summary[rule_summary["selection_rule"] == "largest_row_col_norms"]
    lines = [
        "# Phase 9: Leakage-Aware Bridge Audit",
        "",
        "Adds a deterministic leakage-aware block-selection rule to the selected-submatrix / "
        "full-system Ridge bridge audit and tests whether reducing discarded "
        "functional-column coupling reduces the first-coordinate surrogate discrepancy. The "
        f"normalized regularization is fixed at the executed anchor lambda = {anchor_lambda:.6f} "
        "(alpha = lambda * beta_B^2 per block). All rows remain surrogate boundary audits.",
        "",
        "## Median relative discrepancy and functional-column leakage by rule",
        "",
        "| Rule | blocks | median rel. discrepancy | median leakage | median coupling | "
        "frac < 0.05 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rule_summary.iterrows():
        lines.append(
            f"| `{row['selection_rule']}` | {int(row['num_blocks'])} | "
            f"{row['median_relative_discrepancy']:.3f} | "
            f"{row['median_functional_column_leakage']:.3f} | "
            f"{row['median_out_of_block_coupling_fraction']:.3f} | "
            f"{row['fraction_below_0.05']:.2f} |"
        )
    lines += [
        "",
        "## Spearman correlation of relative discrepancy with deleted-coupling diagnostics "
        "(all computed rows)",
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
        f"The relative first-coordinate discrepancy correlates with functional-column leakage "
        f"(Spearman rho = {leak_rho:.3f}): the surrogate tracks the full-system functional only "
        "when most of the target functional's Jacobian column is retained.",
    ]
    if not leakage_row.empty and not norm_row.empty:
        la_leak = float(leakage_row["median_functional_column_leakage"].iloc[0])
        la_disc = float(leakage_row["median_relative_discrepancy"].iloc[0])
        nr_leak = float(norm_row["median_functional_column_leakage"].iloc[0])
        nr_disc = float(norm_row["median_relative_discrepancy"].iloc[0])
        lines.append(
            f"The leakage-aware rule attains median functional-column leakage {la_leak:.3f} and "
            f"median relative discrepancy {la_disc:.3f}, versus the executed norm rule's leakage "
            f"{nr_leak:.3f} and discrepancy {nr_disc:.3f}."
        )
    lines += [
        "",
        "## Boundary statements",
        "",
        "- Selected-submatrix circuits remain surrogate boundary tests.",
        "- Lower leakage improves surrogate fidelity only when the selected functional is "
        "mostly supported by the retained rows and columns.",
        "- This does not prove equivalence to the full PSSE Ridge update.",
        "- This helps define when a selected-submatrix workload is a meaningful surrogate.",
        "",
        "No speedup, no quantum execution, and no full-system selected-output claim is made; "
        "this is a boundary audit of the surrogate construction.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 9: leakage-aware bridge audit")
    parser.add_argument("--output-dir", default=str(PHASE9_BRIDGE_DIR))
    args = parser.parse_args(argv)
    run = run_phase9_bridge_leakage_aware(
        {
            "output_dir": args.output_dir,
            "command": "scripts/run_phase9_bridge_leakage_aware.py " + " ".join(argv or []),
        }
    )
    print(run["rule_summary"].to_string(index=False))
    print(f"Bridge leakage-aware audit complete: {run['artifacts']['manifest']}")


if __name__ == "__main__":  # pragma: no cover
    main()
