"""Experiment A: isolated signed-overlap shot statistics for the 4x4 demo.

Replaces the single finite-shot Hadamard-test readout of the passing 4x4
selected-observable QSVT demonstration with a *statistically characterized*
readout over many shot seeds and shot counts. This answers the reviewer concern
(W4) that the reported readout is one finite-shot realization and might be a
favourable draw.

The Hadamard-test overlap estimator (``estimate_overlap``) directly prepares the
classically computed postselected output vector.  With ``qiskit-aer`` present
each seed samples that isolated compiled overlap circuit; otherwise ancilla
outcomes are drawn from its exact probability.  This is not an integrated
residual-loading--QSVT--postselection--readout execution.

Scope: this characterizes the demonstrated 4x4 selected observable only. It does
not solve full-vector readout and does not imply IEEE-scale quantum execution.
Ridge/Tikhonov is the matched reference; the QSVT-target filter is the same
regularized filter at the same alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.circuit_signed_readout import estimate_overlap
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    _state_labels_for_cols,
    run_demo_for_block,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    EXPERIMENTS_CLAIM_BOUNDARY,
    READOUT_DIR,
    assert_safe,
    get_pyplot,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

DEFAULT_CASE = "ieee14"
DEFAULT_SEED = 123
DEFAULT_DEGREE = 31
DEFAULT_ALPHA_MULT = 4.0
DEMO_PHASE_CACHE = Path("outputs/selected_observable_qsvt_demo/phase_cache")

FULL_SHOTS = (1_000, 10_000, 100_000, 1_000_000)
DEFAULT_SHOTS = (1_000, 10_000, 100_000)
QUICK_SHOTS = (1_000, 10_000, 100_000)

SEED_RESULT_COLUMNS = [
    "demo_id",
    "matrix_source",
    "matrix_shape",
    "observable_label",
    "seed",
    "shots",
    "sampling_mode",
    "output_state_access_model",
    "integrated_qsvt_readout",
    "exact_ridge_functional",
    "exact_svt_functional",
    "exact_qsvt_statevector_functional",
    "finite_shot_estimate",
    "absolute_error_vs_ridge",
    "relative_error_vs_ridge",
    "absolute_error_vs_exact_svt",
    "relative_error_vs_exact_svt",
    "overlap_standard_error",
    "success_probability",
    "postselection_probability",
    "degree",
    "alpha",
    "beta",
    "C",
    "physical_rescaling_factor",
    "notes",
]

SUMMARY_COLUMNS = [
    "observable_label",
    "shots",
    "mean_estimate",
    "std_estimate",
    "median_estimate",
    "mean_absolute_error_vs_ridge",
    "std_absolute_error_vs_ridge",
    "mean_relative_error_vs_ridge",
    "std_relative_error_vs_ridge",
    "p05_relative_error_vs_ridge",
    "p50_relative_error_vs_ridge",
    "p95_relative_error_vs_ridge",
    "mean_overlap_standard_error",
    "num_seeds",
    "num_failures",
]

POOLED_LABEL = "__all_signed_pooled__"


def _build_headline(case: str, case_source: str, seed: int, degree: int, alpha_mult: float) -> Any:
    """Reproduce the passing 4x4 demo headline block (no files written)."""

    system, matrix_source = build_engineering_system(
        {
            "case_name": case,
            "case_source": case_source,
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    H_block, r_block, rows, cols = select_deterministic_block(
        H_full, r_full, row_count=4, col_count=4, policy="largest_row_col_norms"
    )
    sigma_min = float(np.linalg.svd(H_block, compute_uv=False).min())
    alpha = float(alpha_mult) * sigma_min**2
    column_labels = _state_labels_for_cols(system.metadata, cols)
    result = run_demo_for_block(
        case=case,
        matrix_source=matrix_source,
        H_block=H_block,
        r_block=r_block,
        selected_rows=rows,
        selected_cols=cols,
        column_labels=column_labels,
        alpha=alpha,
        degree=int(degree),
        angle_solver="iterative",
        margin=1.05,
        domain_low_factor=0.9,
        pass_relative_tolerance=0.05,
        phase_cache_dir=DEMO_PHASE_CACHE,
    )
    return result, matrix_source


def _exact_svt_update(result: Any) -> np.ndarray:
    """Ideal dense-SVT update ``dx_svt = (C/beta)(U p(S) V^T) r_B`` (no synthesis error)."""

    meta = result.pipeline_metadata
    beta = float(meta["beta"])
    recovery = float(meta["physical_recovery_factor_C_over_beta"])  # = C/beta
    coefficients = np.asarray(meta["polynomial_coefficients_bounded"], dtype=np.float64)
    A = result.H_block.T / beta
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    transform = U @ np.diag(Polynomial(coefficients)(S)) @ Vt
    return recovery * (transform @ result.r_block)


def run_readout_statistics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    seeds = list(range(int(resolved["num_seeds"])))
    shots_grid = [int(s) for s in resolved["shots_grid"]]

    result, matrix_source = _build_headline(
        resolved["case"],
        resolved["case_source"],
        int(resolved["seed"]),
        int(resolved["degree"]),
        float(resolved["alpha_mult"]),
    )
    warnings: list[str] = []
    failures: list[dict[str, Any]] = []
    if result.status_label != "pass":
        warnings.append(
            f"headline 4x4 block status is '{result.status_label}', not 'pass'; readout "
            "statistics are still reported for the demonstrated observable"
        )

    meta = result.pipeline_metadata
    beta = float(meta["beta"])
    bound_c = float(meta["bound_C"])
    alpha = float(meta["alpha"])
    degree = int(meta["degree"])
    recovery = float(meta["physical_recovery_factor_C_over_beta"])
    p_success = float(result.row_common["postselection_probability"])
    matrix_shape = str(result.row_common["block_shape"])
    demo_id = f"{resolved['case']}_{matrix_shape}_selected_observable_qsvt_demo"

    dx_ridge = result.ridge_update
    dx_svt = _exact_svt_update(result)
    psi = np.asarray(result.output_state, dtype=np.complex128)  # complex postselected |psi_out>
    physical_recovery_scale = float(result.physical_recovery_scale)

    signed = [obs for obs in result.observables if obs.sign_aware_required]
    rows: list[dict[str, Any]] = []
    for obs in signed:
        ell = np.asarray(obs.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        scale = physical_recovery_scale * ell_norm
        exact_ridge = float(ell @ dx_ridge)
        exact_svt = float(ell @ dx_svt)
        for shots in shots_grid:
            for seed in seeds:
                estimate = estimate_overlap(ell, psi, shots=int(shots), seed=int(seed))
                sampling_mode = (
                    "finite_shot_sampling_proxy_from_exact_overlap"
                    if estimate.backend == "exact_p0_binomial"
                    else "aer_circuit_shot_sampling"
                )
                finite = scale * estimate.overlap_estimate
                exact_qsvt = scale * estimate.overlap_exact
                abs_ridge = abs(finite - exact_ridge)
                abs_svt = abs(finite - exact_svt)
                rows.append(
                    {
                        "demo_id": demo_id,
                        "matrix_source": matrix_source,
                        "matrix_shape": matrix_shape,
                        "observable_label": obs.observable_id,
                        "seed": int(seed),
                        "shots": int(shots),
                        "sampling_mode": sampling_mode,
                        "output_state_access_model": (
                            "direct_StatePreparation_of_classically_computed_postselected_output"
                        ),
                        "integrated_qsvt_readout": False,
                        "exact_ridge_functional": exact_ridge,
                        "exact_svt_functional": exact_svt,
                        "exact_qsvt_statevector_functional": exact_qsvt,
                        "finite_shot_estimate": finite,
                        "absolute_error_vs_ridge": abs_ridge,
                        "relative_error_vs_ridge": abs_ridge / abs(exact_ridge)
                        if abs(exact_ridge) > 1.0e-15
                        else float("nan"),
                        "absolute_error_vs_exact_svt": abs_svt,
                        "relative_error_vs_exact_svt": abs_svt / abs(exact_svt)
                        if abs(exact_svt) > 1.0e-15
                        else float("nan"),
                        "overlap_standard_error": float(estimate.standard_error),
                        "success_probability": p_success,
                        "postselection_probability": p_success,
                        "degree": degree,
                        "alpha": alpha,
                        "beta": beta,
                        "C": bound_c,
                        "physical_rescaling_factor": recovery,
                        "notes": obs.physical_meaning,
                    }
                )

    seed_frame = pd.DataFrame(rows, columns=SEED_RESULT_COLUMNS)
    summary_frame = _summarize(seed_frame)

    artifacts = _write_outputs(
        output_dir=output_dir,
        seed_frame=seed_frame,
        summary_frame=summary_frame,
        result=result,
        p_success=p_success,
    )
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="A_readout_statistics",
        script_name="scripts/run_tqe_revision_readout_statistics.py",
        command=resolved["command"],
        description=(
            "Seed-resolved isolated Hadamard-overlap shot statistics for the passing "
            "4x4 selected-submatrix QSVT demonstration. The classically computed "
            "postselected output is loaded directly; this is not integrated QSVT readout."
        ),
        artifacts=artifacts,
        inputs_used=[f"build_engineering_system:{resolved['case']}:weighted_jacobian"],
        random_seeds={
            "shot_seeds": seeds,
            "demo_system_seed": int(resolved["seed"]),
        },
        warnings=warnings,
        failures=failures,
        interpretation_boundary=(
            "This characterizes isolated signed-overlap shot noise under assumed preparation "
            "of the classically computed postselected output state. It does not integrate the "
            "residual loader, QSVT circuit, postselection, and readout; does not solve "
            "full-vector readout; and does not imply IEEE-scale quantum execution. The "
            "shot-noise floor sits above the (small, systematic) QSVT-vs-Ridge approximation "
            "error; the QSVT-target filter equals Ridge at the same alpha."
        ),
        extra={
            "headline_status": result.status_label,
            "num_shot_seeds": len(seeds),
            "shots_grid": shots_grid,
            "signed_observables": [obs.observable_id for obs in signed],
            "sampling_backends_seen": sorted(seed_frame["sampling_mode"].unique().tolist())
            if not seed_frame.empty
            else [],
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "seed_results": seed_frame,
        "summary": summary_frame,
        "artifacts": artifacts,
        "headline_status": result.status_label,
    }


def _summarize(seed_frame: pd.DataFrame) -> pd.DataFrame:
    if seed_frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    groups: list[dict[str, Any]] = []

    def _stats(label: str, shots: int, block: pd.DataFrame) -> dict[str, Any]:
        estimates = block["finite_shot_estimate"].to_numpy(dtype=np.float64)
        abs_err = block["absolute_error_vs_ridge"].to_numpy(dtype=np.float64)
        rel_err = block["relative_error_vs_ridge"].to_numpy(dtype=np.float64)
        finite_ok = np.isfinite(estimates)
        rel_ok = rel_err[np.isfinite(rel_err)]
        # Raw estimate statistics are only meaningful per observable; for the pooled row
        # (mixing observables of different magnitude) they are left NaN.
        pooled = label == POOLED_LABEL
        return {
            "observable_label": label,
            "shots": int(shots),
            "mean_estimate": float("nan")
            if pooled
            else (float(np.mean(estimates[finite_ok])) if finite_ok.any() else float("nan")),
            "std_estimate": float("nan")
            if pooled
            else (float(np.std(estimates[finite_ok])) if finite_ok.any() else float("nan")),
            "median_estimate": float("nan")
            if pooled
            else (float(np.median(estimates[finite_ok])) if finite_ok.any() else float("nan")),
            "mean_absolute_error_vs_ridge": float(np.mean(abs_err[np.isfinite(abs_err)]))
            if np.isfinite(abs_err).any()
            else float("nan"),
            "std_absolute_error_vs_ridge": float(np.std(abs_err[np.isfinite(abs_err)]))
            if np.isfinite(abs_err).any()
            else float("nan"),
            "mean_relative_error_vs_ridge": float(np.mean(rel_ok)) if rel_ok.size else float("nan"),
            "std_relative_error_vs_ridge": float(np.std(rel_ok)) if rel_ok.size else float("nan"),
            "p05_relative_error_vs_ridge": float(np.percentile(rel_ok, 5))
            if rel_ok.size
            else float("nan"),
            "p50_relative_error_vs_ridge": float(np.percentile(rel_ok, 50))
            if rel_ok.size
            else float("nan"),
            "p95_relative_error_vs_ridge": float(np.percentile(rel_ok, 95))
            if rel_ok.size
            else float("nan"),
            "mean_overlap_standard_error": float(block["overlap_standard_error"].mean()),
            "num_seeds": int(block["seed"].nunique()),
            "num_failures": int((~finite_ok).sum()),
        }

    for (label, shots), block in seed_frame.groupby(["observable_label", "shots"], sort=True):
        groups.append(_stats(str(label), int(shots), block))
    for shots, block in seed_frame.groupby("shots", sort=True):
        groups.append(_stats(POOLED_LABEL, int(shots), block))
    return pd.DataFrame(groups, columns=SUMMARY_COLUMNS)


def _write_outputs(
    *,
    output_dir: Path,
    seed_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    result: Any,
    p_success: float,
) -> dict[str, Path]:
    seed_csv = output_dir / "readout_seed_results.csv"
    summary_csv = output_dir / "readout_shot_scaling_summary.csv"
    seed_frame.to_csv(seed_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)

    pooled = summary_frame[summary_frame["observable_label"] == POOLED_LABEL].sort_values("shots")
    pdf_path = output_dir / "readout_error_vs_shots.pdf"
    png_path = output_dir / "readout_error_vs_shots.png"
    fig_tex = output_dir / "readout_error_vs_shots.tex"
    table_tex = output_dir / "readout_statistics_table.tex"
    _plot_error_vs_shots(pooled, pdf_path, png_path)
    fig_tex.write_text(_figure_tex(pooled), encoding="utf-8")
    table_tex.write_text(_summary_table_tex(summary_frame), encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(_readme(seed_frame, summary_frame, result, p_success), encoding="utf-8")

    return {
        "readout_seed_results_csv": seed_csv,
        "readout_shot_scaling_summary_csv": summary_csv,
        "readout_error_vs_shots_pdf": pdf_path,
        "readout_error_vs_shots_png": png_path,
        "readout_error_vs_shots_tex": fig_tex,
        "readout_statistics_table_tex": table_tex,
        "readme_md": readme,
    }


def _plot_error_vs_shots(pooled: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if not pooled.empty:
        shots = pooled["shots"].to_numpy(dtype=np.float64)
        mean_rel = pooled["mean_relative_error_vs_ridge"].to_numpy(dtype=np.float64)
        std_rel = pooled["std_relative_error_vs_ridge"].to_numpy(dtype=np.float64)
        ax.errorbar(
            shots,
            mean_rel,
            yerr=std_rel,
            marker="o",
            capsize=4,
            color="#1f77b4",
            label="mean relative error vs Ridge (pooled)",
        )
        # 1/sqrt(N) reference trend scaled to the first observed mean error.
        reference = mean_rel[0] * np.sqrt(shots[0] / shots)
        ax.plot(
            shots,
            reference,
            linestyle="--",
            color="#555555",
            label=r"$1/\sqrt{N_{\mathrm{shots}}}$ reference",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots $N_{\\mathrm{shots}}$")
    ax.set_ylabel("relative error vs Ridge reference")
    ax.set_title("4x4 isolated signed-overlap shot-noise scaling")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def _figure_tex(pooled: pd.DataFrame) -> str:
    coords = " ".join(
        f"({int(row['shots'])},{row['mean_relative_error_vs_ridge']:.6e})"
        for _, row in pooled.iterrows()
    )
    ref = ""
    if not pooled.empty:
        first = pooled.iloc[0]
        ref_pairs = [
            (
                int(row["shots"]),
                float(first["mean_relative_error_vs_ridge"])
                * (float(first["shots"]) / float(row["shots"])) ** 0.5,
            )
            for _, row in pooled.iterrows()
        ]
        ref = " ".join(f"({s},{v:.6e})" for s, v in ref_pairs)
    lines = [
        "% Readout shot-noise scaling for the 4x4 selected observable (pooled signed).",
        "% Requires pgfplots. Data: pooled mean relative error vs Ridge and 1/sqrt(N) ref.",
        r"\begin{tikzpicture}",
        r"\begin{loglogaxis}[width=0.8\linewidth, xlabel={Shots $N_{\mathrm{shots}}$},",
        r"  ylabel={Relative error vs Ridge}, legend pos=south west, grid=both]",
        rf"\addplot+[mark=*] coordinates {{{coords}}};",
        r"\addlegendentry{mean rel.\ error (pooled)}",
        rf"\addplot+[dashed,no marks] coordinates {{{ref}}};",
        r"\addlegendentry{$1/\sqrt{N_{\mathrm{shots}}}$}",
        r"\end{loglogaxis}",
        r"\end{tikzpicture}",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _summary_table_tex(summary: pd.DataFrame) -> str:
    pooled = summary[summary["observable_label"] == POOLED_LABEL].sort_values("shots")
    lines = [
        "% Readout shot-scaling summary (pooled signed observables) for the 4x4 demo.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"Shots & mean rel.\ err & std rel.\ err & p05 & p95 & seeds \\",
        r"\midrule",
    ]
    for _, row in pooled.iterrows():
        lines.append(
            f"{int(row['shots'])} & {row['mean_relative_error_vs_ridge']:.3e} & "
            f"{row['std_relative_error_vs_ridge']:.3e} & "
            f"{row['p05_relative_error_vs_ridge']:.3e} & "
            f"{row['p95_relative_error_vs_ridge']:.3e} & {int(row['num_seeds'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Seed-resolved shot-noise statistics of an isolated Hadamard-overlap "
        r"circuit for the 4x4 selected-submatrix output (pooled over signed functionals and shot "
        r"seeds). The relative error is measured against the matched Ridge/Tikhonov "
        r"reference; the QSVT-target filter equals Ridge at the same $\alpha$. This "
        r"postselected output vector is computed classically and loaded directly; this is not "
        r"integrated residual-loading--QSVT--postselection--readout execution.}",
        r"\label{tab:readout_shot_statistics}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _readme(seed_frame: pd.DataFrame, summary: pd.DataFrame, result: Any, p_success: float) -> str:
    pooled = summary[summary["observable_label"] == POOLED_LABEL].sort_values("shots")
    empty = seed_frame.empty
    backends = sorted(seed_frame["sampling_mode"].unique().tolist()) if not empty else []
    n_seeds = int(seed_frame["seed"].nunique()) if not empty else 0
    shot_list = sorted(seed_frame["shots"].unique().tolist()) if not empty else []
    lines = [
        "# Experiment A: Readout Statistics (4x4 selected observable)",
        "",
        EXPERIMENTS_CLAIM_BOUNDARY,
        "",
        "Seed-resolved isolated Hadamard-overlap shot experiment for the passing 4x4 "
        "selected-submatrix QSVT demonstration. The postselected output vector is computed "
        "classically and loaded directly with `StatePreparation(output_state)`. Each row of "
        "`readout_seed_results.csv` is one "
        "`(observable, shot seed, shot count)` estimate; `readout_shot_scaling_summary.csv` "
        "aggregates seeds per shot count.",
        "",
        f"- headline block status: `{result.status_label}`, postselection probability "
        f"p_success = {p_success:.4f}",
        f"- sampling backends seen: {backends} "
        "(`aer_circuit_shot_sampling` = genuine compiled-circuit shots; "
        "`finite_shot_sampling_proxy_from_exact_overlap` = Bernoulli draws from the exact "
        "circuit probability when Aer is absent)",
        f"- shot seeds per cell: {n_seeds}, shot grid: {shot_list}",
        "",
        "## Pooled shot-noise scaling (relative error vs Ridge)",
        "",
        "| Shots | mean rel err | std rel err | p05 | p50 | p95 | overlap SE | seeds | fails |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in pooled.iterrows():
        mre, sre = row["mean_relative_error_vs_ridge"], row["std_relative_error_vs_ridge"]
        p05, p50, p95 = (
            row["p05_relative_error_vs_ridge"],
            row["p50_relative_error_vs_ridge"],
            row["p95_relative_error_vs_ridge"],
        )
        lines.append(
            f"| {int(row['shots'])} | {mre:.3e} | {sre:.3e} | {p05:.3e} | {p50:.3e} | {p95:.3e} | "
            f"{row['mean_overlap_standard_error']:.3e} | {int(row['num_seeds'])} | "
            f"{int(row['num_failures'])} |"
        )
    lines += [
        "",
        "The mean relative error falls with more shots and tracks the `1/sqrt(N_shots)` "
        "reference until it approaches the small, systematic QSVT-vs-Ridge approximation floor "
        "of the passing block (update relative error "
        f"{float(result.row_common['update_relative_error_vs_ridge']):.2e}).",
        "",
        "## Interpretation boundary",
        "",
        "This characterizes signed-overlap shot noise under assumed output-state preparation. "
        "It does **not** compose residual loading, QSVT, postselection, and readout into one "
        "shot-executed circuit. Integrated finite-shot readout remains future work. It does "
        "**not** solve full-vector readout and does **not** imply IEEE-scale quantum "
        "execution. The "
        "readout estimates the *same* signed functional of the *same* regularized update that "
        "Ridge/Tikhonov computes at the same alpha; no speedup or advantage is implied.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(READOUT_DIR),
        "case": DEFAULT_CASE,
        "case_source": "pypower",
        "seed": DEFAULT_SEED,
        "degree": DEFAULT_DEGREE,
        "alpha_mult": DEFAULT_ALPHA_MULT,
        "num_seeds": 30,
        "shots_grid": list(DEFAULT_SHOTS),
        "command": "run_tqe_revision_readout_statistics",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    resolved["num_seeds"] = int(resolved["num_seeds"])
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A: readout statistics for the 4x4 demo"
    )
    parser.add_argument("--output-dir", default=str(READOUT_DIR))
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--quick", action="store_true", help="fast smoke run (12 seeds)")
    parser.add_argument("--full", action="store_true", help="full run (30 seeds, incl. 1e6 shots)")
    parser.add_argument("--num-seeds", type=int, default=None, help="override the shot-seed count")
    args = parser.parse_args(argv)

    if args.full:
        num_seeds, shots = 30, list(FULL_SHOTS)
    elif args.quick:
        num_seeds, shots = 12, list(QUICK_SHOTS)
    else:
        num_seeds, shots = 30, list(DEFAULT_SHOTS)
    if args.num_seeds is not None:
        num_seeds = int(args.num_seeds)

    run = run_readout_statistics(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "case_source": args.case_source,
            "seed": args.seed,
            "degree": args.degree,
            "num_seeds": num_seeds,
            "shots_grid": shots,
            "command": "scripts/run_tqe_revision_readout_statistics.py " + " ".join(argv or []),
        }
    )
    print(f"Readout statistics complete: {run['artifacts']['readout_seed_results_csv']}")
    print(f"Headline status: {run['headline_status']}, rows: {len(run['seed_results'])}")


if __name__ == "__main__":  # pragma: no cover
    main()
