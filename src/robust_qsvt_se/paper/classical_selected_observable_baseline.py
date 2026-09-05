"""Task 5: classical selected-observable baselines.

Computes ``y_l = l^T (H~^T H~ + alpha I)^{-1} H~^T r~`` classically by three
methods and reports their cost, so the manuscript can state the conservative
baseline plainly: for the tested matrices, classical selected-observable
computation remains the practical baseline, and the QSVT pathway is an
implementation study under explicit access/preparation/readout assumptions.

Methods:

* ``dense_direct``      - dense normal-equation solve (small blocks),
* ``sparse_factorized`` - sparse LU factorization of ``H~^T H~ + alpha I`` reused
  across all queried observables (one full update, many functionals),
* ``adjoint_functional``- solve ``(H~^T H~ + alpha I) w = l`` then ``y = w^T H~^T r~``
  (one adjoint solve per functional; the factorization is shared).

The adjoint route is the natural classical analogue of a selected-observable
query: it never forms the full update vector for the functional value.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    CLASSICAL_BASELINE_DIR,
    assert_safe,
    write_demo_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118", "ieee300")
DEFAULT_ALPHA = 1.0e-2

BASELINE_COLUMNS = [
    "case",
    "matrix_kind",
    "matrix_shape",
    "nnz",
    "observable_id",
    "observable_type",
    "alpha",
    "method",
    "runtime_seconds",
    "runtime_q1_seconds",
    "runtime_q3_seconds",
    "runtime_iqr_seconds",
    "runtime_min_seconds",
    "runtime_max_seconds",
    "preprocessing_median_seconds",
    "query_median_seconds",
    "timing_repeats",
    "solver_type",
    "timing_scope",
    "residual_norm",
    "selected_functional_value",
    "abs_difference_from_dense_reference",
    "factorization_reused",
    "num_observables_queried",
]


def _predetermined_observables(dimension: int) -> list[dict[str, Any]]:
    half = max(1, (dimension + 1) // 2)
    single = np.zeros(dimension)
    single[0] = 1.0
    diff = np.zeros(dimension)
    diff[0], diff[1 % dimension] = 1.0, -1.0
    area = np.zeros(dimension)
    area[:half] = 1.0
    return [
        {
            "observable_id": "state_correction_0",
            "observable_type": "single_state_correction",
            "vector": single,
        },
        {
            "observable_id": "state_difference_0_1",
            "observable_type": "branch_angle_difference_like",
            "vector": diff,
        },
        {
            "observable_id": "area_aggregate_first_half",
            "observable_type": "area_aggregate_functional",
            "vector": area,
        },
    ]


def _dense_direct(H: np.ndarray, r: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    gram = H.T @ H + alpha * np.eye(H.shape[1])
    update = np.linalg.solve(gram, H.T @ r)
    return update, time.perf_counter() - start


def _sparse_normal_matrix(H: sp.spmatrix, alpha: float) -> sp.csc_matrix:
    gram = (H.T @ H).tocsc()
    gram = gram + alpha * sp.identity(H.shape[1], format="csc")
    return gram.tocsc()


def run_classical_selected_observable_baseline(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    alpha = float(resolved["alpha"])
    rows: list[dict[str, Any]] = []
    timing_samples: list[dict[str, Any]] = []

    # Small block baseline (dense direct vs adjoint), loaded from the demo if present.
    block = _load_or_build_block(resolved)
    rows.extend(
        _baseline_for_matrix(
            block["H"],
            block["r"],
            alpha,
            case=block["label"],
            dense=True,
            timing_repeats=int(resolved["timing_repeats"]),
            warmup_repeats=int(resolved["warmup_repeats"]),
            timing_samples=timing_samples,
        )
    )

    for case in resolved["cases"]:
        try:
            system, _ = build_engineering_system(
                {
                    "case_name": case,
                    "case_source": resolved["case_source"],
                    "matrix_source": "weighted_jacobian",
                    "seed": int(resolved["seed"]),
                }
            )
        except Exception:
            continue
        H = np.asarray(system.H_tilde, dtype=np.float64)
        r = np.asarray(system.r_tilde, dtype=np.float64)
        rows.extend(
            _baseline_for_matrix(
                H,
                r,
                alpha,
                case=case,
                dense=H.shape[1] <= 64,
                timing_repeats=int(resolved["timing_repeats"]),
                warmup_repeats=int(resolved["warmup_repeats"]),
                timing_samples=timing_samples,
            )
        )

    frame = pd.DataFrame(rows, columns=BASELINE_COLUMNS)
    summary_csv = output_dir / "baseline_summary.csv"
    frame.to_csv(summary_csv, index=False)

    timing_samples_csv = output_dir / "timing_samples.csv"
    pd.DataFrame(timing_samples).to_csv(timing_samples_csv, index=False)

    metadata_json = output_dir / "baseline_metadata.json"
    write_json(metadata_json, _metadata(frame, resolved))

    readme = output_dir / "README.md"
    readme.write_text(_readme(frame), encoding="utf-8")

    artifacts = {
        "baseline_summary_csv": summary_csv,
        "timing_samples_csv": timing_samples_csv,
        "baseline_metadata_json": metadata_json,
        "readme_md": readme,
    }
    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="classical_selected_observable_baseline",
        description=(
            "Classical selected-observable baselines (dense direct, sparse factorized, "
            "adjoint functional) for y_l = l^T (H^T H + alpha I)^{-1} H^T r. Establishes the "
            "conservative practical baseline against which the QSVT pathway is an "
            "implementation study."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[
            block["source"],
            *[f"build_engineering_system:{c}" for c in resolved["cases"]],
        ],
        extra={
            "alpha": alpha,
            "cases": list(resolved["cases"]),
            "timing_repeats": int(resolved["timing_repeats"]),
            "warmup_repeats": int(resolved["warmup_repeats"]),
            "random_seeds": {"system_seed": int(resolved["seed"])},
            "seed_provenance": {
                "status": "recorded",
                "seed": int(resolved["seed"]),
                "source": (
                    "scripts/run_classical_selected_observable_baseline.py (--seed default 123)"
                ),
                "applies_to": (
                    "IEEE measurement-row generation in build_engineering_system for "
                    "every case; the 4x4 block is loaded from the demo's "
                    "selected_block.npy when present; the dense/sparse/adjoint solves "
                    "themselves use no RNG"
                ),
            },
        },
        manifest_name="baseline_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "summary": frame, "artifacts": artifacts}


def _baseline_for_matrix(
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    *,
    case: str,
    dense: bool,
    timing_repeats: int = 1,
    warmup_repeats: int = 0,
    timing_samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if timing_repeats < 1 or warmup_repeats < 0:
        raise ValueError("timing_repeats must be positive and warmup_repeats nonnegative")
    n = int(H.shape[1])
    nnz = int(np.count_nonzero(np.abs(H) > 1.0e-12))
    observables = _predetermined_observables(n)
    reference_update = ridge_svd_solution(H, r, alpha=alpha)
    reference_values = {
        obs["observable_id"]: float(obs["vector"] @ reference_update) for obs in observables
    }
    rows: list[dict[str, Any]] = []

    if dense:
        dense_update, dense_timing = _time_dense_direct(H, r, alpha, timing_repeats, warmup_repeats)
        _record_timing_samples(
            timing_samples,
            case,
            "dense_direct",
            observables,
            dense_timing,
        )
        dense_residual = float(np.linalg.norm(H @ dense_update - r))
        for obs in observables:
            value = float(obs["vector"] @ dense_update)
            rows.append(
                _row(
                    case=case,
                    matrix_kind="dense",
                    H=H,
                    nnz=nnz,
                    obs=obs,
                    alpha=alpha,
                    method="dense_direct",
                    timing=dense_timing,
                    solver_type="NumPy dense solve (LAPACK)",
                    timing_scope="Gram/RHS formation + dense factorization/solve",
                    residual=dense_residual,
                    value=value,
                    reference=reference_values[obs["observable_id"]],
                    factorization_reused=False,
                    num_observables=len(observables),
                )
            )

    # Sparse factorized full update (factorization reused across observables).
    H_sparse = sp.csr_matrix(H)
    sparse_update, sparse_timing = _time_sparse_full_update(
        H_sparse, r, alpha, timing_repeats, warmup_repeats
    )
    _record_timing_samples(
        timing_samples,
        case,
        "sparse_factorized",
        observables,
        sparse_timing,
    )
    sparse_residual = float(np.linalg.norm(H_sparse @ sparse_update - r))
    for obs in observables:
        value = float(obs["vector"] @ sparse_update)
        rows.append(
            _row(
                case=case,
                matrix_kind="sparse",
                H=H,
                nnz=nnz,
                obs=obs,
                alpha=alpha,
                method="sparse_factorized",
                timing=sparse_timing,
                solver_type="SciPy sparse SuperLU full-update solve",
                timing_scope="Sparse Gram/RHS formation + factorization + full-update solve",
                residual=sparse_residual,
                value=value,
                reference=reference_values[obs["observable_id"]],
                factorization_reused=True,
                num_observables=len(observables),
            )
        )

    # Adjoint functional solve: (H^T H + alpha I) w = l ; y = w^T (H^T r).
    # The factorization is preprocessing shared by all queried functionals.
    adjoint_values, adjoint_timings = _time_adjoint_functionals(
        H_sparse,
        r,
        alpha,
        observables,
        timing_repeats,
        warmup_repeats,
    )
    for obs, timing in zip(observables, adjoint_timings, strict=True):
        value = adjoint_values[obs["observable_id"]]
        _record_timing_samples(timing_samples, case, "adjoint_functional", [obs], timing)
        rows.append(
            _row(
                case=case,
                matrix_kind="sparse",
                H=H,
                nnz=nnz,
                obs=obs,
                alpha=alpha,
                method="adjoint_functional",
                timing=timing,
                solver_type="SciPy sparse SuperLU adjoint solve",
                timing_scope="Shared sparse Gram/RHS/factorization + one functional solve",
                residual=float("nan"),
                value=value,
                reference=reference_values[obs["observable_id"]],
                factorization_reused=True,
                num_observables=len(observables),
            )
        )
    return rows


def _timing_summary(preprocessing: list[float], query: list[float]) -> dict[str, Any]:
    prep = np.asarray(preprocessing, dtype=np.float64)
    qry = np.asarray(query, dtype=np.float64)
    total = prep + qry
    return {
        "preprocessing": prep,
        "query": qry,
        "total": total,
        "median": float(np.median(total)),
        "q1": float(np.quantile(total, 0.25)),
        "q3": float(np.quantile(total, 0.75)),
        "minimum": float(np.min(total)),
        "maximum": float(np.max(total)),
        "preprocessing_median": float(np.median(prep)),
        "query_median": float(np.median(qry)),
        "repeats": int(total.size),
    }


def _time_dense_direct(
    H: np.ndarray, r: np.ndarray, alpha: float, repeats: int, warmups: int
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessing: list[float] = []
    query: list[float] = []
    update = np.empty(H.shape[1])
    for repeat in range(warmups + repeats):
        start = time.perf_counter()
        gram = H.T @ H + alpha * np.eye(H.shape[1])
        rhs = H.T @ r
        middle = time.perf_counter()
        update = np.linalg.solve(gram, rhs)
        stop = time.perf_counter()
        if repeat >= warmups:
            preprocessing.append(middle - start)
            query.append(stop - middle)
    return update, _timing_summary(preprocessing, query)


def _time_sparse_full_update(
    H: sp.csr_matrix, r: np.ndarray, alpha: float, repeats: int, warmups: int
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessing: list[float] = []
    query: list[float] = []
    update = np.empty(H.shape[1])
    for repeat in range(warmups + repeats):
        start = time.perf_counter()
        gram = _sparse_normal_matrix(H, alpha)
        rhs = np.asarray(H.T @ r).ravel()
        lu = spla.splu(gram)
        middle = time.perf_counter()
        update = lu.solve(rhs)
        stop = time.perf_counter()
        if repeat >= warmups:
            preprocessing.append(middle - start)
            query.append(stop - middle)
    return update, _timing_summary(preprocessing, query)


def _time_adjoint_functionals(
    H: sp.csr_matrix,
    r: np.ndarray,
    alpha: float,
    observables: list[dict[str, Any]],
    repeats: int,
    warmups: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    preprocessing = [[] for _ in observables]
    query = [[] for _ in observables]
    values: dict[str, float] = {}
    for repeat in range(warmups + repeats):
        start = time.perf_counter()
        gram = _sparse_normal_matrix(H, alpha)
        htr = np.asarray(H.T @ r).ravel()
        lu = spla.splu(gram)
        middle = time.perf_counter()
        prep_seconds = middle - start
        for index, obs in enumerate(observables):
            solve_start = time.perf_counter()
            w = lu.solve(np.asarray(obs["vector"], dtype=np.float64))
            values[obs["observable_id"]] = float(w @ htr)
            solve_seconds = time.perf_counter() - solve_start
            if repeat >= warmups:
                preprocessing[index].append(prep_seconds)
                query[index].append(solve_seconds)
    return values, [
        _timing_summary(prep, qry) for prep, qry in zip(preprocessing, query, strict=True)
    ]


def _record_timing_samples(
    target: list[dict[str, Any]] | None,
    case: str,
    method: str,
    observables: list[dict[str, Any]],
    timing: dict[str, Any],
) -> None:
    if target is None:
        return
    for obs in observables:
        for index, (prep, query, total) in enumerate(
            zip(timing["preprocessing"], timing["query"], timing["total"], strict=True)
        ):
            target.append(
                {
                    "case": case,
                    "method": method,
                    "observable_id": obs["observable_id"],
                    "repeat": index,
                    "preprocessing_seconds": float(prep),
                    "query_seconds": float(query),
                    "total_seconds": float(total),
                }
            )


def _row(
    *,
    case: str,
    matrix_kind: str,
    H: np.ndarray,
    nnz: int,
    obs: dict[str, Any],
    alpha: float,
    method: str,
    timing: dict[str, Any],
    solver_type: str,
    timing_scope: str,
    residual: float,
    value: float,
    reference: float,
    factorization_reused: bool,
    num_observables: int,
) -> dict[str, Any]:
    return {
        "case": case,
        "matrix_kind": matrix_kind,
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "nnz": int(nnz),
        "observable_id": obs["observable_id"],
        "observable_type": obs["observable_type"],
        "alpha": float(alpha),
        "method": method,
        "runtime_seconds": timing["median"],
        "runtime_q1_seconds": timing["q1"],
        "runtime_q3_seconds": timing["q3"],
        "runtime_iqr_seconds": timing["q3"] - timing["q1"],
        "runtime_min_seconds": timing["minimum"],
        "runtime_max_seconds": timing["maximum"],
        "preprocessing_median_seconds": timing["preprocessing_median"],
        "query_median_seconds": timing["query_median"],
        "timing_repeats": timing["repeats"],
        "solver_type": solver_type,
        "timing_scope": timing_scope,
        "residual_norm": float(residual),
        "selected_functional_value": float(value),
        "abs_difference_from_dense_reference": float(abs(value - reference)),
        "factorization_reused": bool(factorization_reused),
        "num_observables_queried": int(num_observables),
    }


def _load_or_build_block(resolved: dict[str, Any]) -> dict[str, Any]:
    block_path = Path(resolved["block_npy"])
    residual_path = Path(resolved["residual_npy"])
    if block_path.is_file() and residual_path.is_file():
        loaded = np.asarray(np.load(block_path), dtype=np.float64)
        return {
            "H": loaded,
            "r": np.asarray(np.load(residual_path), dtype=np.float64),
            "label": f"selected_block_{loaded.shape[0]}x{loaded.shape[1]}",
            "source": str(block_path),
        }
    system, _ = build_engineering_system(
        {
            "case_name": resolved["cases"][0],
            "case_source": resolved["case_source"],
            "matrix_source": "weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
    )
    H_block, r_block, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    return {
        "H": np.asarray(H_block, dtype=np.float64),
        "r": np.asarray(r_block, dtype=np.float64),
        "label": "selected_block_4x4",
        "source": f"build_engineering_system:{resolved['cases'][0]}:weighted_jacobian",
    }


def _metadata(frame: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    max_diff = float(frame["abs_difference_from_dense_reference"].max()) if not frame.empty else 0.0
    return {
        "alpha": float(resolved["alpha"]),
        "cases": list(resolved["cases"]),
        "methods": ["dense_direct", "sparse_factorized", "adjoint_functional"],
        "timing_repeats": int(resolved["timing_repeats"]),
        "warmup_repeats": int(resolved["warmup_repeats"]),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported by platform",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "max_abs_difference_across_methods": max_diff,
        "manuscript_implication": (
            "For the tested matrices, classical selected-observable computation remains the "
            "practical baseline. The QSVT pathway is an implementation study under explicit "
            "access, preparation, and readout assumptions."
        ),
        "claim_boundary": (
            "Classical baselines only; no quantum component. Establishes the reference cost, "
            "not a speedup comparison."
        ),
    }


def _readme(frame: pd.DataFrame) -> str:
    lines = [
        "# Classical Selected-Observable Baseline",
        "",
        "Classical computation of `y_l = l^T (H~^T H~ + alpha I)^{-1} H~^T r~` by dense "
        "direct, sparse factorized (factorization reused across functionals), and adjoint "
        "functional solves.",
        "",
        "**Manuscript implication (conservative):** for the tested matrices, classical "
        "selected-observable computation remains the practical baseline. The QSVT pathway is "
        "an implementation study under explicit access, preparation, and readout assumptions.",
        "",
        "Timing reports median and interquartile range over repeated runs; preprocessing "
        "and query time are retained separately in `baseline_summary.csv`, and every raw "
        "sample is retained in `timing_samples.csv`.",
        "",
        "| Case | Shape | Method | Observable | median [IQR] (s) | functional value | "
        "diff vs dense ref | factorization reused |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['case']} | {row['matrix_shape']} | {row['method']} | "
            f"`{row['observable_id']}` | {row['runtime_seconds']:.3e} "
            f"[{row['runtime_q1_seconds']:.3e}, {row['runtime_q3_seconds']:.3e}] | "
            f"{row['selected_functional_value']:+.5e} | "
            f"{row['abs_difference_from_dense_reference']:.2e} | "
            f"{row['factorization_reused']} |"
        )
    lines.append("")
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(CLASSICAL_BASELINE_DIR),
        "block_npy": "outputs/selected_observable_qsvt_demo/selected_block.npy",
        "residual_npy": "outputs/selected_observable_qsvt_demo/selected_residual.npy",
        "cases": list(DEFAULT_CASES),
        "case_source": "pypower",
        "seed": 123,
        "alpha": DEFAULT_ALPHA,
        "timing_repeats": 30,
        "warmup_repeats": 3,
        "command": "run_classical_selected_observable_baseline",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    resolved["alpha"] = float(resolved["alpha"])
    if resolved["alpha"] <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the classical selected-observable baseline")
    parser.add_argument("--output-dir", default=str(CLASSICAL_BASELINE_DIR))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    run = run_classical_selected_observable_baseline(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "seed": args.seed,
            "alpha": args.alpha,
            "timing_repeats": args.timing_repeats,
            "warmup_repeats": args.warmup_repeats,
            "command": (
                "scripts/run_classical_selected_observable_baseline.py " + " ".join(argv or [])
            ),
        }
    )
    print(f"Classical baseline complete: {run['artifacts']['baseline_summary_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
