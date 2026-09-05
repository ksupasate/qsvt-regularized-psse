"""Phase 9: lambda-accuracy-realizability tradeoff for the bounded Ridge QSVT target.

Sweeps the normalized regularization ``lambda = alpha/beta^2`` over a fixed grid
for four selected-submatrix blocks and records, per ``(block, lambda)``, both

* the *estimator* effect: matched Ridge/Tikhonov selected functional, update norm,
  and the regularization bias relative to a benchmark ``lambda`` (the sharpest
  grid point), and
* the *QSVT realizability* effect: minimum polynomial degree meeting a configured
  fit-error target, phase-synthesis pass/fail and boundedness at the reference
  degree, the statevector SVT update error where synthesis succeeds, and both the
  residual-specific and uniform-input postselection probabilities.

The scientific message is a *tested toolchain boundary*, not a general QSVT lower
bound: larger ``lambda`` makes the bounded target smoother and easier to
synthesize but changes the Ridge estimator; smaller ``lambda`` preserves a
sharper inverse-like filter but is harder for the tested polynomial-fitting and
phase-synthesis pipeline.  Ridge/Tikhonov is the matched reference at the same
``alpha`` throughout; no speedup and no QSVT-over-Ridge numerical superiority is
claimed, and the controlled high-``kappa`` block is a controlled stress row, not
broad high-``kappa`` PSSE feasibility evidence.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import fit_codesigned_bounded_polynomial
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    _state_labels_for_cols,
    run_demo_for_block,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    assert_safe,
    controlled_svd_matrix,
    get_pyplot,
    uniform_postselection_probability,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/phase9_lambda_tradeoff")
REFERENCE_DEGREE = 31
TARGET_FIT_ERROR = 5.0e-3
MIN_DEGREE_LADDER = (7, 11, 15, 19, 23, 27, 31, 35, 39, 45, 51, 61)
LAMBDA_GRID = (1.0e-1, 6.9e-2, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 1.0e-4)
BENCHMARK_LAMBDA = min(LAMBDA_GRID)  # sharpest inverse-like filter in the grid
PASS_RELATIVE_TOLERANCE = 0.05
CONTROLLED_RESIDUAL_SEED = 0
BOUND_TOLERANCE = 2.0e-3

FULL_COLUMNS = [
    "block_id",
    "case",
    "block_size",
    "block_kind",
    "selection_rule",
    "residual_provenance",
    "kappa_block",
    "sigma_min",
    "sigma_max",
    "beta",
    "lambda_normalized",
    "alpha_physical",
    "bound_C",
    "ridge_selected_functional",
    "ridge_update_norm",
    "residual_norm",
    "benchmark_lambda",
    "relative_functional_bias_vs_benchmark",
    "relative_update_difference_vs_benchmark",
    "reference_degree",
    "target_fit_error",
    "min_feasible_degree_fit",
    "modeled_signal_unitary_calls_for_target_precision",
    "polynomial_fit_max_abs_error_ref_degree",
    "boundedness_ok_ref_degree",
    "bounded_max_abs_ref_degree",
    "phase_synthesis_status_ref_degree",
    "phase_count_ref_degree",
    "statevector_svt_update_relative_error",
    "actual_residual_postselection_probability",
    "uniform_input_proxy_postselection_probability",
    "demo_status_label",
    "notes",
]

SUMMARY_COLUMNS = [
    "block_id",
    "block_kind",
    "kappa_block",
    "lambda_normalized",
    "min_feasible_degree_fit",
    "phase_synthesis_status_ref_degree",
    "statevector_svt_update_relative_error",
    "actual_residual_postselection_probability",
    "uniform_input_proxy_postselection_probability",
    "relative_functional_bias_vs_benchmark",
    "demo_status_label",
]


def _domain_min(singular_values: np.ndarray, beta: float, domain_low_factor: float = 0.9) -> float:
    normalized_min = float(np.min(singular_values)) / float(beta) * domain_low_factor
    return float(np.clip(normalized_min, 1e-4, 0.999))


def _min_feasible_degree_for_fit(
    beta: float, alpha: float, domain_min: float, target_error: float
) -> tuple[int, dict[int, float]]:
    """Smallest odd degree meeting the fit-error target with a bounded polynomial.

    Fit-only (no phase synthesis): the polynomial-approximation lever. Returns the
    minimum degree (0 if none in the ladder) and the per-degree fit-error trace.
    """

    trace: dict[int, float] = {}
    best = 0
    for degree in MIN_DEGREE_LADDER:
        try:
            target = fit_codesigned_bounded_polynomial(
                beta=float(beta),
                alpha=float(alpha),
                domain_min=float(domain_min),
                domain_max=1.0,
                degree=int(degree),
                margin=1.05,
            )
            fit_error = float(target.fit_max_abs_error)
            coefficients = np.asarray(target.coefficients, dtype=np.float64)
        except Exception:
            trace[int(degree)] = float("nan")
            continue
        try:
            validate_qsvt_polynomial(coefficients, parity="odd", bound_tolerance=BOUND_TOLERANCE)
            bounded = True
        except Exception:
            bounded = False
        trace[int(degree)] = fit_error
        if best == 0 and bounded and fit_error <= float(target_error):
            best = int(degree)
    return best, trace


def _block_specs(seed: int) -> list[dict[str, Any]]:
    """The four tradeoff blocks; each yields ``H_block, r_block, singular_values, meta``."""

    specs: list[dict[str, Any]] = []

    def ieee_block(case: str, size: int, block_id: str) -> dict[str, Any] | None:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        H_full = np.asarray(system.H_tilde, dtype=np.float64)
        r_full = np.asarray(system.r_tilde, dtype=np.float64)
        if size > min(H_full.shape):
            return None
        H_block, r_block, rows, cols = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        return {
            "block_id": block_id,
            "case": case,
            "block_size": size,
            "block_kind": "ieee_derived_weighted_jacobian",
            "selection_rule": "largest_row_col_norms (deterministic, pre-solve)",
            "residual_provenance": (
                f"PYPOWER-generated {case} AC weighted-Jacobian residual, seed {seed}"
            ),
            "H_block": H_block,
            "r_block": r_block,
            "selected_rows": rows,
            "selected_cols": cols,
            "matrix_source": matrix_source,
            "column_labels": _state_labels_for_cols(system.metadata, cols),
        }

    for case, size, block_id, required in (
        ("ieee14", 4, "ieee14_4x4_anchor", True),
        ("ieee14", 8, "ieee14_8x8_lambda_matched", True),
        ("ieee30", 16, "ieee30_16x16_raw", False),
    ):
        try:
            spec = ieee_block(case, size, block_id)
        except Exception:
            spec = None
        if spec is not None:
            specs.append(spec)
        elif required:
            raise RuntimeError(f"required block {block_id} could not be constructed")

    # Controlled high-kappa stress block: synthetic log-spaced SVD matrix and a
    # deterministic synthetic residual (there is no physical PSSE residual here).
    matrix, _singular = controlled_svd_matrix(8, 1.0e4, seed=101)
    rng = np.random.default_rng(CONTROLLED_RESIDUAL_SEED)
    synthetic_residual = rng.standard_normal(8)
    specs.append(
        {
            "block_id": "controlled_8x8_kappa_1e4",
            "case": "controlled_svd",
            "block_size": 8,
            "block_kind": "controlled_svd_stress",
            "selection_rule": "controlled_svd_matrix(log-spaced sigma, random U,V; seed 101)",
            "residual_provenance": (
                f"synthetic standard-normal residual (seed {CONTROLLED_RESIDUAL_SEED}); "
                "controlled stress row, not a physical PSSE residual"
            ),
            "H_block": np.asarray(matrix, dtype=np.float64),
            "r_block": synthetic_residual,
            "selected_rows": np.arange(8),
            "selected_cols": np.arange(8),
            "matrix_source": "controlled_svd",
            "column_labels": [],
        }
    )
    return specs


def _rows_for_block(spec: dict[str, Any]) -> list[dict[str, Any]]:
    H_block = np.asarray(spec["H_block"], dtype=np.float64)
    r_block = np.asarray(spec["r_block"], dtype=np.float64)
    singular = np.linalg.svd(H_block, compute_uv=False)
    beta = float(singular.max())
    sigma_min = float(singular.min())
    kappa = float(beta / sigma_min) if sigma_min > 0 else math.inf
    residual_norm = float(np.linalg.norm(r_block))

    rows: list[dict[str, Any]] = []
    for lam in LAMBDA_GRID:
        alpha = float(lam) * beta**2
        ridge_update = ridge_svd_solution(H_block, r_block, alpha=alpha)
        functional = float(ridge_update[0])
        update_norm = float(np.linalg.norm(ridge_update))

        domain_min = _domain_min(singular, beta)
        min_degree, _trace = _min_feasible_degree_for_fit(beta, alpha, domain_min, TARGET_FIT_ERROR)

        result = run_demo_for_block(
            case=str(spec["case"]),
            matrix_source=str(spec["matrix_source"]),
            H_block=H_block,
            r_block=r_block,
            selected_rows=np.asarray(spec["selected_rows"]),
            selected_cols=np.asarray(spec["selected_cols"]),
            column_labels=list(spec["column_labels"]),
            alpha=alpha,
            degree=REFERENCE_DEGREE,
            angle_solver="iterative",
            margin=1.05,
            domain_low_factor=0.9,
            pass_relative_tolerance=PASS_RELATIVE_TOLERANCE,
            phase_cache_dir=OUTPUT_DIR / "phase_cache",
        )
        common = result.row_common
        bound_c = float(common["bound_C"])
        synthesized = common["phase_synthesis_status"] == "completed"
        uniform_proxy = uniform_postselection_probability(
            singular, beta=beta, alpha=alpha, bound_C=bound_c
        )
        rows.append(
            {
                "block_id": spec["block_id"],
                "case": spec["case"],
                "block_size": int(spec["block_size"]),
                "block_kind": spec["block_kind"],
                "selection_rule": spec["selection_rule"],
                "residual_provenance": spec["residual_provenance"],
                "kappa_block": kappa,
                "sigma_min": sigma_min,
                "sigma_max": beta,
                "beta": beta,
                "lambda_normalized": float(lam),
                "alpha_physical": alpha,
                "bound_C": bound_c,
                "ridge_selected_functional": functional,
                "ridge_update_norm": update_norm,
                "residual_norm": residual_norm,
                "benchmark_lambda": BENCHMARK_LAMBDA,
                "relative_functional_bias_vs_benchmark": math.nan,  # filled post-hoc
                "relative_update_difference_vs_benchmark": math.nan,  # filled post-hoc
                "reference_degree": REFERENCE_DEGREE,
                "target_fit_error": TARGET_FIT_ERROR,
                "min_feasible_degree_fit": min_degree if min_degree > 0 else math.nan,
                "modeled_signal_unitary_calls_for_target_precision": (
                    min_degree if min_degree > 0 else math.nan
                ),
                "polynomial_fit_max_abs_error_ref_degree": float(
                    common["polynomial_fit_max_abs_error"]
                ),
                "boundedness_ok_ref_degree": bool(common["boundedness_ok"]),
                "bounded_max_abs_ref_degree": float(common["bounded_max_abs"]),
                "phase_synthesis_status_ref_degree": common["phase_synthesis_status"],
                "phase_count_ref_degree": int(common["phase_count"]),
                "statevector_svt_update_relative_error": (
                    float(common["update_relative_error_vs_ridge"]) if synthesized else math.nan
                ),
                "actual_residual_postselection_probability": (
                    float(common["postselection_probability"]) if synthesized else math.nan
                ),
                "uniform_input_proxy_postselection_probability": float(uniform_proxy),
                "demo_status_label": result.status_label,
                "notes": "",
            }
        )

    # Regularization bias relative to the benchmark (sharpest) lambda.
    benchmark = next(
        row for row in rows if math.isclose(row["lambda_normalized"], BENCHMARK_LAMBDA)
    )
    bench_func = benchmark["ridge_selected_functional"]
    bench_update = ridge_svd_solution(H_block, r_block, alpha=BENCHMARK_LAMBDA * beta**2)
    bench_update_norm = float(np.linalg.norm(bench_update))
    for row in rows:
        alpha = row["alpha_physical"]
        update = ridge_svd_solution(H_block, r_block, alpha=alpha)
        row["relative_functional_bias_vs_benchmark"] = (
            abs(row["ridge_selected_functional"] - bench_func) / abs(bench_func)
            if abs(bench_func) > 1.0e-15
            else math.nan
        )
        row["relative_update_difference_vs_benchmark"] = (
            float(np.linalg.norm(update - bench_update)) / bench_update_norm
            if bench_update_norm > 1.0e-15
            else math.nan
        )
    return rows


def run_phase9_lambda_tradeoff(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "command": "run_phase9_lambda_tradeoff",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    ensure_directory(output_dir / "phase_cache")

    specs = _block_specs(int(resolved["seed"]))
    all_rows: list[dict[str, Any]] = []
    included: list[str] = []
    for spec in specs:
        all_rows.extend(_rows_for_block(spec))
        included.append(spec["block_id"])
    unavailable = [
        block_id
        for block_id in (
            "ieee14_4x4_anchor",
            "ieee14_8x8_lambda_matched",
            "ieee30_16x16_raw",
            "controlled_8x8_kappa_1e4",
        )
        if block_id not in included
    ]

    frame = pd.DataFrame(all_rows, columns=FULL_COLUMNS)
    summary = frame[SUMMARY_COLUMNS].copy()

    full_csv = output_dir / "lambda_tradeoff_full.csv"
    summary_csv = output_dir / "lambda_tradeoff_summary.csv"
    frame.to_csv(full_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    figures = _write_figures(frame, output_dir)
    readme_md = output_dir / "README.md"
    readme_md.write_text(_readme(frame, included, unavailable), encoding="utf-8")

    artifacts: dict[str, Path] = {
        "lambda_tradeoff_full_csv": full_csv,
        "lambda_tradeoff_summary_csv": summary_csv,
        "readme_md": readme_md,
        **figures,
    }
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="phase9_lambda_tradeoff",
        script_name="scripts/run_phase9_lambda_tradeoff.py",
        command=str(resolved["command"]),
        description=(
            "Normalized-regularization (lambda = alpha/beta^2) accuracy-realizability tradeoff "
            "for the bounded Ridge QSVT target across four selected-submatrix blocks. Records "
            "the Ridge estimator effect (functional, update norm, regularization bias) and the "
            "QSVT realizability effect (minimum fit-feasible degree, phase-synthesis pass/fail, "
            "statevector SVT update error, residual and uniform-input postselection "
            "probabilities) per (block, lambda)."
        ),
        artifacts=artifacts,
        inputs_used=[
            "build_engineering_system:ieee14:weighted_jacobian",
            "build_engineering_system:ieee30:weighted_jacobian",
            "controlled_svd_matrix:8x8:kappa_1e4:seed101",
        ],
        random_seeds={
            "system_seed": int(resolved["seed"]),
            "controlled_matrix_seed": 101,
            "controlled_residual_seed": CONTROLLED_RESIDUAL_SEED,
        },
        warnings=[f"blocks_unavailable:{unavailable}"] if unavailable else [],
        failures=[],
        interpretation_boundary=(
            "A tested toolchain boundary of the polynomial-fitting and phase-synthesis "
            "pipeline, not a general QSVT lower bound. Ridge/Tikhonov is the matched reference "
            "at the same alpha; no speedup and no QSVT-over-Ridge numerical superiority is "
            "claimed. The controlled high-kappa block is a controlled stress row, not broad "
            "high-kappa PSSE feasibility evidence."
        ),
        extra={
            "lambda_grid": list(LAMBDA_GRID),
            "benchmark_lambda": BENCHMARK_LAMBDA,
            "reference_degree": REFERENCE_DEGREE,
            "target_fit_error": TARGET_FIT_ERROR,
            "min_degree_ladder": list(MIN_DEGREE_LADDER),
            "blocks_included": included,
            "blocks_unavailable": unavailable,
        },
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "results": frame, "summary": summary, "artifacts": artifacts}


def _write_figures(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    plt = get_pyplot()
    blocks = list(dict.fromkeys(frame["block_id"]))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(blocks)))

    def _line(ax, ycol: str, *, ylabel: str, logy: bool = False) -> None:
        for block, color in zip(blocks, colors, strict=True):
            block_rows = frame[frame["block_id"] == block].sort_values("lambda_normalized")
            x = block_rows["lambda_normalized"].to_numpy(dtype=np.float64)
            y = block_rows[ycol].to_numpy(dtype=np.float64)
            ax.plot(x, y, marker="o", ms=4, color=color, label=block)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"normalized regularization $\lambda=\alpha/\beta^2$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _line(
        ax,
        "relative_functional_bias_vs_benchmark",
        ylabel=f"Ridge functional bias vs $\\lambda={BENCHMARK_LAMBDA:g}$",
        logy=True,
    )
    ax.set_title("Estimator regularization bias vs $\\lambda$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    bias_path = output_dir / "estimator_bias_vs_lambda.png"
    fig.savefig(bias_path, dpi=150)
    plt.close(fig)
    paths["figure_estimator_bias_png"] = bias_path

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _line(ax, "min_feasible_degree_fit", ylabel="min fit-feasible odd degree")
    ax.set_title(f"Minimum feasible degree vs $\\lambda$ (fit target {TARGET_FIT_ERROR:g})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    degree_path = output_dir / "min_degree_vs_lambda.png"
    fig.savefig(degree_path, dpi=150)
    plt.close(fig)
    paths["figure_min_degree_png"] = degree_path

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _line(
        ax,
        "uniform_input_proxy_postselection_probability",
        ylabel="uniform-input postselection probability",
    )
    ax.set_title("Postselection probability (uniform proxy) vs $\\lambda$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    psucc_path = output_dir / "postselection_vs_lambda.png"
    fig.savefig(psucc_path, dpi=150)
    plt.close(fig)
    paths["figure_postselection_png"] = psucc_path

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _line(axes[0], "min_feasible_degree_fit", ylabel="min fit-feasible odd degree")
    axes[0].set_title("Realizability: min degree vs $\\lambda$")
    _line(
        axes[1],
        "relative_functional_bias_vs_benchmark",
        ylabel=f"Ridge functional bias vs $\\lambda={BENCHMARK_LAMBDA:g}$",
        logy=True,
    )
    axes[1].set_title("Estimator: bias vs $\\lambda$")
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Lambda accuracy-realizability tradeoff (smaller $\\lambda$: sharper filter, "
        "harder synthesis)"
    )
    fig.tight_layout()
    combined_path = output_dir / "lambda_tradeoff_combined.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    paths["figure_combined_png"] = combined_path

    return paths


def _readme(frame: pd.DataFrame, included: list[str], unavailable: list[str]) -> str:
    lines = [
        "# Phase 9: Lambda Accuracy-Realizability Tradeoff",
        "",
        "How the normalized regularization `lambda = alpha/beta^2` trades off the "
        "Ridge/Tikhonov estimator against QSVT realizability, for four selected-submatrix "
        "blocks. For each block and each lambda, `alpha = lambda * beta^2`; the matched "
        "Ridge reference is evaluated at the same alpha.",
        "",
        f"- lambda grid: {', '.join(f'{v:g}' for v in LAMBDA_GRID)}",
        f"- benchmark lambda for regularization bias: {BENCHMARK_LAMBDA:g} (sharpest grid point)",
        f"- reference synthesis degree: {REFERENCE_DEGREE}; fit-feasibility target: "
        f"{TARGET_FIT_ERROR:g}",
        f"- blocks included: {', '.join(included)}",
    ]
    if unavailable:
        lines.append(f"- blocks unavailable: {', '.join(unavailable)}")
    lines += [
        "",
        "## Key messages",
        "",
        "- Larger `lambda` improves QSVT realizability (smoother bounded target, lower minimum "
        "feasible degree, phase synthesis succeeds at the reference degree) but changes the "
        "Ridge/Tikhonov estimator (larger regularization bias vs the sharpest grid point).",
        "- Smaller `lambda` preserves a sharper inverse-like filter (smaller estimator bias) "
        "but is harder for the tested polynomial-fitting and phase-synthesis pipeline (higher "
        "minimum feasible degree; boundedness or phase synthesis fails at the reference "
        "degree).",
        "- This is a tested toolchain boundary, not a general QSVT lower bound.",
        "- No quantum speed-up is claimed.",
        "- No QSVT-over-Ridge numerical superiority is claimed.",
        "- Successful high-`kappa` controlled rows are controlled stress evidence, not broad "
        "high-`kappa` PSSE feasibility.",
        "",
        "## Per-block realizability boundary at the reference degree",
        "",
        "| Block | kappa | smallest feasible lambda (pass) | largest infeasible lambda | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for block in included:
        block_rows = frame[frame["block_id"] == block]
        passed = block_rows[block_rows["demo_status_label"] == "pass"]["lambda_normalized"]
        failed = block_rows[block_rows["demo_status_label"] != "pass"]["lambda_normalized"]
        smallest_pass = f"{passed.min():g}" if not passed.empty else "none in grid"
        largest_fail = f"{failed.max():g}" if not failed.empty else "none in grid"
        kappa = float(block_rows["kappa_block"].iloc[0])
        lines.append(
            f"| `{block}` | {kappa:.1f} | {smallest_pass} | {largest_fail} | "
            "smaller lambda is harder |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `estimator_bias_vs_lambda.png`: Ridge functional bias grows with lambda.",
        "- `min_degree_vs_lambda.png`: minimum fit-feasible degree grows as lambda shrinks.",
        "- `postselection_vs_lambda.png`: uniform-input postselection probability vs lambda.",
        "- `lambda_tradeoff_combined.png`: realizability and estimator panels side by side.",
        "",
        "Ridge/Tikhonov is the matched reference at the same alpha; the QSVT-target filter is "
        "the same regularized spectral filter expressed as a QSVT workload. This package does "
        "not claim a quantum speed-up, QSVT-over-Ridge numerical superiority, validation "
        "against real PMU/SCADA field measurements, or full IEEE-scale execution.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 9: lambda accuracy-realizability tradeoff")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_phase9_lambda_tradeoff(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "command": "scripts/run_phase9_lambda_tradeoff.py " + " ".join(argv or []),
        }
    )
    summary = run["summary"]
    print(summary.to_string(index=False, max_colwidth=28))
    print(f"Results: {run['output_dir']}/lambda_tradeoff_full.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
