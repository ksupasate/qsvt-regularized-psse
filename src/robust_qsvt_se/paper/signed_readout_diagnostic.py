"""Task B: selected signed-readout diagnostic for QSVT output interpretation.

Basis sampling of a QSVT output state yields energy-style (squared-amplitude)
observables, but signed PSSE update components - voltage-angle and
voltage-magnitude corrections - need the *sign* of a linear functional of the
update. This module models a sign-aware readout for a *selected* signed linear
functional

    q_l = l^T . dx_alpha,    dx_alpha = (H~^T H~ + alpha I)^{-1} H~^T r~,

using a Hadamard-test-style proxy: with mu = <l_hat | dx_hat> in [-1, 1] the
ancilla measures 0 with probability p = (1 + mu)/2, and the shot estimate is
q_hat = ||l|| ||dx|| (2 p_hat - 1).

This is a simulator-level *diagnostic* of a sign-aware measurement model. It is
NOT full-vector tomography, NOT a synthesized circuit, NOT hardware execution,
and NOT a quantum-speedup claim. dx_alpha is the matched-alpha Ridge update, so
no QSVT-over-Ridge superiority is implied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import (
    ridge_svd_update,
    select_deterministic_block,
)
from robust_qsvt_se.paper.tqe_revision_support_common import (
    REVISION_OUTPUT_ROOT,
    find_forbidden,
    write_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory
from robust_qsvt_se.utils.seed import make_rng

SIGNED_READOUT_DIR = REVISION_OUTPUT_ROOT / "signed_readout"

READOUT_MODEL = "Hadamard-test-style sign-aware inner-product proxy"

DEFAULT_SHOTS = (100, 1_000, 10_000, 100_000)

SUMMARY_COLUMNS = [
    "case_or_block",
    "source_type",
    "case_source",
    "block_shape",
    "alpha",
    "observable_type",
    "observable_label",
    "true_signed_value",
    "estimated_signed_value",
    "absolute_error",
    "relative_error",
    "shots",
    "seed",
    "readout_model",
    "full_vector_readout",
    "hardware_execution",
]

# Relative error is only reported when the exact value is safely above this floor.
_RELATIVE_ERROR_FLOOR = 1.0e-9


@dataclass(frozen=True, slots=True)
class SignedObservable:
    observable_type: str
    label: str
    vector: np.ndarray


def coordinate_observable(dimension: int, index: int) -> SignedObservable:
    vector = np.zeros(dimension, dtype=np.float64)
    vector[index] = 1.0
    return SignedObservable("coordinate", f"e_{index}", vector)


def signed_pair_observable(dimension: int, plus: int, minus: int) -> SignedObservable:
    vector = np.zeros(dimension, dtype=np.float64)
    vector[plus] = 1.0
    vector[minus] = -1.0
    return SignedObservable("pair", f"e_{plus}-e_{minus}", vector)


def area_observable(dimension: int, count: int) -> SignedObservable:
    count = int(np.clip(count, 1, dimension))
    vector = np.zeros(dimension, dtype=np.float64)
    vector[:count] = 1.0
    return SignedObservable("area", f"sum(e_0..e_{count - 1})", vector)


def top_k_signed_direction(update: np.ndarray, k: int) -> SignedObservable:
    """Signed indicator on the top-``k`` |update| coordinates (selected direction)."""

    update = np.asarray(update, dtype=np.float64)
    k = int(np.clip(k, 1, update.size))
    support = np.sort(np.argsort(np.abs(update))[::-1][:k])
    vector = np.zeros(update.size, dtype=np.float64)
    signs = np.sign(update[support])
    signs[signs == 0.0] = 1.0
    vector[support] = signs
    label = "topk_sign[" + ",".join(str(int(i)) for i in support) + "]"
    return SignedObservable("top-k", label, vector)


def hadamard_test_estimate(
    observable: np.ndarray,
    update: np.ndarray,
    *,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return ``(true_value, estimated_value, mu)`` for a sign-aware shot estimate.

    ``mu`` is the normalized inner product in [-1, 1]; the Hadamard-test ancilla
    measures the |0> outcome with probability ``p = (1 + mu) / 2``. The estimator
    ``q_hat = ||l|| ||dx|| (2 p_hat - 1)`` is unbiased: E[q_hat] = q.
    """

    ell = np.asarray(observable, dtype=np.float64)
    dx = np.asarray(update, dtype=np.float64)
    true_value = float(ell @ dx)
    ell_norm = float(np.linalg.norm(ell))
    dx_norm = float(np.linalg.norm(dx))
    if ell_norm <= 0.0 or dx_norm <= 0.0:
        return true_value, 0.0, 0.0
    mu = float(np.clip(true_value / (ell_norm * dx_norm), -1.0, 1.0))
    p = 0.5 * (1.0 + mu)
    successes = int(rng.binomial(int(shots), p))
    p_hat = successes / int(shots)
    mu_hat = 2.0 * p_hat - 1.0
    estimated = ell_norm * dx_norm * mu_hat
    return true_value, float(estimated), mu


def build_observables(update: np.ndarray) -> list[SignedObservable]:
    dimension = int(update.size)
    order = np.argsort(np.abs(update))[::-1]
    top1 = int(order[0])
    top2 = int(order[1]) if dimension > 1 else top1
    observables = [
        coordinate_observable(dimension, 0),
        coordinate_observable(dimension, top1),
    ]
    if dimension > 1:
        observables.append(signed_pair_observable(dimension, top1, top2))
    observables.append(area_observable(dimension, (dimension + 1) // 2))
    observables.append(top_k_signed_direction(update, min(2, dimension)))
    # De-duplicate observables that collapse to the same label on tiny blocks.
    unique: dict[str, SignedObservable] = {}
    for observable in observables:
        unique.setdefault(observable.label, observable)
    return list(unique.values())


def run_signed_readout_diagnostic(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []

    for block_size in resolved["block_sizes"]:
        block = _prepare_block(resolved, block_size=block_size)
        block_records.append(block["record"])
        update = block["update"]
        observables = build_observables(update)
        for obs_index, observable in enumerate(observables):
            for shot_index, shots in enumerate(resolved["shots"]):
                seed = _row_seed(resolved["base_seed"], block_size, obs_index, shot_index)
                rng = make_rng(seed)
                true_value, estimated, _mu = hadamard_test_estimate(
                    observable.vector,
                    update,
                    shots=int(shots),
                    rng=rng,
                )
                rows.append(
                    _summary_row(
                        block=block,
                        observable=observable,
                        true_value=true_value,
                        estimated=estimated,
                        shots=int(shots),
                        seed=seed,
                        alpha=float(resolved["alpha"]),
                    )
                )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    error_curve = _error_vs_shots(summary)

    summary_csv = output_dir / "signed_readout_summary.csv"
    summary_md = output_dir / "signed_readout_summary.md"
    error_csv = output_dir / "signed_readout_error_vs_shots.csv"
    summary.to_csv(summary_csv, index=False)
    error_curve.to_csv(error_csv, index=False)
    summary_md.write_text(_summary_markdown(summary, error_curve, block_records), encoding="utf-8")

    artifacts = {
        "signed_readout_summary_csv": summary_csv,
        "signed_readout_summary_md": summary_md,
        "signed_readout_error_vs_shots": error_csv,
    }
    if resolved["save_plots"]:
        plot = _plot_error_vs_shots(output_dir, error_curve)
        if plot is not None:
            artifacts["signed_readout_error_vs_shots_png"] = plot

    manifest = write_manifest(
        output_dir=output_dir,
        artifact_name="signed_readout_diagnostic",
        description=(
            "Hadamard-test-style sign-aware readout for selected signed observables of the "
            "matched-alpha Ridge/QSVT-target update. Selected signed observables only; not "
            "full-vector readout, simulator-level only (no hardware run), not a speedup claim."
        ),
        artifacts=artifacts,
        extra={
            "readout_model": READOUT_MODEL,
            "full_vector_readout": False,
            "hardware_execution": False,
            "speedup_claim": False,
            "selected_signed_observables_only": True,
            "shots_grid": list(resolved["shots"]),
            "block_sizes": list(resolved["block_sizes"]),
            "alpha": float(resolved["alpha"]),
            "blocks": block_records,
        },
        manifest_name="signed_readout_manifest.json",
    )
    artifacts["manifest"] = manifest
    _self_check_safe(summary_md)
    return {
        "output_dir": output_dir,
        "summary": summary,
        "error_curve": error_curve,
        "blocks": block_records,
        "artifacts": artifacts,
    }


def _prepare_block(resolved: dict[str, Any], *, block_size: int) -> dict[str, Any]:
    source_type = "toy fallback, not manuscript main evidence"
    case_source = str(resolved["case_source"])
    block_label = f"{resolved['case_name']}_{block_size}x{block_size}"
    try:
        system, matrix_source = build_engineering_system(
            {
                "case_name": resolved["case_name"],
                "case_source": case_source,
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["base_seed"]),
            }
        )
        H_block, r_block, _rows, _cols = select_deterministic_block(
            system.H_tilde,
            system.r_tilde,
            row_count=int(block_size),
            col_count=int(block_size),
            policy="largest_row_col_norms",
        )
        source_type = "IEEE-derived"
    except Exception:
        H_block, r_block, matrix_source = _toy_block(block_size, seed=int(resolved["base_seed"]))
        block_label = f"toy_{block_size}x{block_size}"
        case_source = "toy"

    update = ridge_svd_update(H_block, r_block, alpha=float(resolved["alpha"]))
    record = {
        "case_or_block": block_label,
        "source_type": source_type,
        "case_source": case_source,
        "matrix_source": str(matrix_source),
        "block_shape": f"{H_block.shape[0]}x{H_block.shape[1]}",
        "update_norm": float(np.linalg.norm(update)),
    }
    return {"record": record, "update": update}


def _toy_block(block_size: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    rng = make_rng(seed + 7919)
    matrix = rng.normal(size=(block_size, block_size))
    # Inject a controlled small singular value so the block is ill-conditioned.
    u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    spectrum = np.linspace(1.0, 1.0e-2, block_size)
    matrix = (u * spectrum) @ vt
    residual = rng.normal(size=block_size)
    return matrix, residual, "toy_illconditioned_block"


def _summary_row(
    *,
    block: dict[str, Any],
    observable: SignedObservable,
    true_value: float,
    estimated: float,
    shots: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    record = block["record"]
    absolute_error = abs(estimated - true_value)
    relative_error = (
        absolute_error / abs(true_value) if abs(true_value) > _RELATIVE_ERROR_FLOOR else ""
    )
    return {
        "case_or_block": record["case_or_block"],
        "source_type": record["source_type"],
        "case_source": record["case_source"],
        "block_shape": record["block_shape"],
        "alpha": float(alpha),
        "observable_type": observable.observable_type,
        "observable_label": observable.label,
        "true_signed_value": float(true_value),
        "estimated_signed_value": float(estimated),
        "absolute_error": float(absolute_error),
        "relative_error": relative_error,
        "shots": int(shots),
        "seed": int(seed),
        "readout_model": READOUT_MODEL,
        "full_vector_readout": False,
        "hardware_execution": False,
    }


def _error_vs_shots(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=["shots", "mean_absolute_error", "median_absolute_error", "max_absolute_error"]
        )
    grouped = (
        summary.groupby("shots")["absolute_error"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_absolute_error",
                "median": "median_absolute_error",
                "max": "max_absolute_error",
                "count": "num_observables",
            }
        )
        .sort_values("shots")
    )
    grouped["one_over_sqrt_shots_reference"] = 1.0 / np.sqrt(grouped["shots"].astype(float))
    return grouped


def _summary_markdown(
    summary: pd.DataFrame,
    error_curve: pd.DataFrame,
    block_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Selected Signed-Readout Diagnostic",
        "",
        "Hadamard-test-style sign-aware estimate of selected signed linear functionals "
        "`q_l = l^T dx_alpha` of the matched-alpha Ridge/QSVT-target update.",
        "",
        "**Scope (claim boundary):** selected signed observables only; **not** full-vector "
        "readout, simulator-level only (no hardware run), **not** a quantum-speedup claim. "
        "The update `dx_alpha` is the matched-alpha Ridge/Tikhonov update, so no "
        "QSVT-over-Ridge superiority is implied.",
        "",
        "## Source Blocks",
        "",
        "| Block | Source type | Case source | Shape | ||dx_alpha|| |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in block_records:
        lines.append(
            f"| `{record['case_or_block']}` | {record['source_type']} | "
            f"{record['case_source']} | {record['block_shape']} | "
            f"{record['update_norm']:.4e} |"
        )
    lines += [
        "",
        "## Mean Absolute Error vs Shots",
        "",
        "| Shots | Mean abs error | Median abs error | Max abs error | 1/sqrt(shots) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in error_curve.iterrows():
        lines.append(
            f"| {int(row['shots'])} | {row['mean_absolute_error']:.4e} | "
            f"{row['median_absolute_error']:.4e} | {row['max_absolute_error']:.4e} | "
            f"{row['one_over_sqrt_shots_reference']:.4e} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The estimator is unbiased; absolute error decreases on the order of "
        "`1/sqrt(shots)`, matching the Monte-Carlo readout cost of a sign-aware "
        "measurement model.",
        "- Each estimate recovers a single selected signed functional, not the full "
        "signed update vector. Recovering every signed component would require a "
        "separate functional per coordinate and remains out of scope.",
        "- This diagnostic models a measurement protocol; it does not synthesize a "
        "circuit and does not run on quantum hardware.",
        "",
    ]
    return "\n".join(lines)


def _plot_error_vs_shots(output_dir: Path, error_curve: pd.DataFrame) -> Path | None:
    if error_curve.empty:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    shots = error_curve["shots"].astype(float)
    ax.loglog(shots, error_curve["mean_absolute_error"], marker="o", label="mean abs error")
    ax.loglog(shots, error_curve["max_absolute_error"], marker="s", label="max abs error")
    reference = error_curve["mean_absolute_error"].iloc[0] * np.sqrt(shots.iloc[0]) / np.sqrt(shots)
    ax.loglog(shots, reference, linestyle="--", color="gray", label="1/sqrt(shots) reference")
    ax.set_xlabel("Shots")
    ax.set_ylabel("Absolute error of selected signed functional")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "signed_readout_error_vs_shots.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _self_check_safe(markdown_path: Path) -> None:
    violations = find_forbidden(markdown_path.read_text(encoding="utf-8"))
    if violations:
        raise RuntimeError(f"signed-readout summary contains forbidden wording: {violations}")


def _row_seed(base_seed: int, block_size: int, obs_index: int, shot_index: int) -> int:
    return int(base_seed) * 1_000_000 + int(block_size) * 1_000 + obs_index * 10 + shot_index


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(SIGNED_READOUT_DIR),
        "case_name": "ieee14",
        "case_source": "pypower",
        "block_sizes": [4, 8],
        "alpha": 1.0e-4,
        "shots": list(DEFAULT_SHOTS),
        "base_seed": 7,
        "save_plots": True,
    }
    if config:
        resolved.update(config)
    resolved["block_sizes"] = [int(size) for size in resolved["block_sizes"]]
    resolved["shots"] = [int(shots) for shots in resolved["shots"]]
    resolved["alpha"] = float(resolved["alpha"])
    if resolved["alpha"] <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the selected signed-readout diagnostic")
    parser.add_argument("--output-dir", default=str(SIGNED_READOUT_DIR))
    parser.add_argument("--case-name", default="ieee14")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run = run_signed_readout_diagnostic(
        {
            "output_dir": args.output_dir,
            "case_name": args.case_name,
            "case_source": args.case_source,
            "block_sizes": args.block_sizes,
            "alpha": args.alpha,
            "shots": args.shots,
            "base_seed": args.base_seed,
            "save_plots": not args.no_plots,
        }
    )
    print(f"Signed-readout diagnostic complete: {run['artifacts']['signed_readout_summary_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
