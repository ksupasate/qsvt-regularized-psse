#!/usr/bin/env python3
"""Randomized-sketching classical selected-output baseline (W7, strongly-rec #7).

Adds sketch-and-solve baselines to the frozen classical comparison for the IDENTICAL IEEE-14
headline configuration used by ``scripts/run_generalized_classical_baselines.py``: same system
(ieee14, pypower, weighted Jacobian, seed 123), same ``lambda = 1e-5`` and
``alpha = lambda * beta^2``, same dense-Ridge reference, same 30-repetition median timing with
factorization amortized separately.  Two declared methods:

* ``sketched_Ridge_s{s}``  - Gaussian sketch-and-solve: solve
  ``min ||S(Hx - r)||^2 + alpha ||x||^2`` with ``S in R^{s x m}``, entries ``N(0, 1/s)``.
* ``row_subsampled_Ridge_s{s}`` - uniform row subsampling with importance weight ``m/s``.

The per-query timing for sketch methods includes sketching the fresh residual (the honest
query cost); the solve-only timing matching the dense row's amortized convention is recorded
alongside.  Results land in a NEW output directory; the frozen baseline CSVs are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "sketched_classical_baseline"
REPS = 30
FACTOR_REPS = 5
LAMBDA = 1.0e-5
SKETCH_SEED = 20260718
SKETCH_SIZES = (32, 54, 82)
PRIMARY_SKETCH_SIZE = 54  # 2n for n=27; the row promoted into the manuscript table


def _time(fn, reps: int = REPS):
    samples = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)), float(np.mean(samples)), out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _write_manifest(directory: Path) -> None:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "checksums.sha256"}:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(directory)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps({"artifact_count": len(artifacts), "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "checksums.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in artifacts),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system, _source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    m, n = H.shape
    beta = float(np.linalg.svd(H, compute_uv=False)[0])
    alpha = LAMBDA * beta * beta

    G = H.T @ H + alpha * np.eye(n)
    b = H.T @ r
    dx_ref = np.linalg.solve(G, b)
    y_ref = float(dx_ref[0])

    rng = np.random.default_rng(SKETCH_SEED)
    rows: list[dict] = []
    for s in SKETCH_SIZES:
        # ---------------- Gaussian sketch-and-solve
        S = rng.standard_normal((s, m)) / np.sqrt(s)

        def factor_sketch(S=S):
            SH = S @ H
            G_sk = SH.T @ SH + alpha * np.eye(n)
            return SH, lu_factor(G_sk)

        t_factor, _, (SH, lu_sk) = _time(factor_sketch, reps=FACTOR_REPS)

        def query_full(S=S, SH=SH, lu_sk=lu_sk):
            Sr = S @ r
            return lu_solve(lu_sk, SH.T @ Sr)

        t_query_full, t_query_full_mean, dx_sk = _time(query_full)
        b_sk = SH.T @ (S @ r)

        def query_solve_only(lu_sk=lu_sk, b_sk=b_sk):
            return lu_solve(lu_sk, b_sk)

        t_query_solve, _, _ = _time(query_solve_only)
        rows.append(
            {
                "method": f"sketched_Ridge_s{s}",
                "family": "gaussian_sketch_and_solve",
                "sketch_rows": int(s),
                "sketch_seed": SKETCH_SEED,
                "is_primary_table_row": bool(s == PRIMARY_SKETCH_SIZE),
                "preprocessing": "sketch SH + LU factor (amortized)",
                "factorization_s": t_factor,
                "per_query_s": t_query_full,
                "per_query_mean_s": t_query_full_mean,
                "per_query_solve_only_s": t_query_solve,
                "matvecs_per_query": 2,  # S r and (SH)^T (S r)
                "memory_kb": float((SH.nbytes + S.nbytes) / 1e3),
                "output_error_vs_ridge": float(np.linalg.norm(dx_sk - dx_ref)),
                "selected_output_rel_error_vs_ridge": abs(float(dx_sk[0]) - y_ref)
                / max(abs(y_ref), 1e-300),
                "note": "per-query includes sketching the fresh residual",
                "timing_reps": REPS,
            }
        )

        # ---------------- uniform row subsampling (importance-weighted)
        indices = np.sort(rng.choice(m, size=s, replace=False))
        weight = m / s

        def factor_subsample(indices=indices, weight=weight):
            H_s = H[indices]
            G_sub = weight * (H_s.T @ H_s) + alpha * np.eye(n)
            return H_s, lu_factor(G_sub)

        t_factor_sub, _, (H_s, lu_sub) = _time(factor_subsample, reps=FACTOR_REPS)

        def query_sub(indices=indices, weight=weight, H_s=H_s, lu_sub=lu_sub):
            return lu_solve(lu_sub, weight * (H_s.T @ r[indices]))

        t_query_sub, t_query_sub_mean, dx_sub = _time(query_sub)
        rows.append(
            {
                "method": f"row_subsampled_Ridge_s{s}",
                "family": "uniform_row_subsampling",
                "sketch_rows": int(s),
                "sketch_seed": SKETCH_SEED,
                "is_primary_table_row": False,
                "preprocessing": "row subset + LU factor (amortized)",
                "factorization_s": t_factor_sub,
                "per_query_s": t_query_sub,
                "per_query_mean_s": t_query_sub_mean,
                "per_query_solve_only_s": t_query_sub,
                "matvecs_per_query": 1,
                "memory_kb": float(H_s.nbytes / 1e3),
                "output_error_vs_ridge": float(np.linalg.norm(dx_sub - dx_ref)),
                "selected_output_rel_error_vs_ridge": abs(float(dx_sub[0]) - y_ref)
                / max(abs(y_ref), 1e-300),
                "note": "uniform rows, importance weight m/s",
                "timing_reps": REPS,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "sketched_baseline_rows.csv", index=False)

    provenance = {
        "study_id": "sketched_classical_baseline_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "generator": "scripts/run_sketched_classical_baseline.py",
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "protocol_match": {
            "system": "ieee14 pypower weighted_jacobian seed 123 (identical to "
            "scripts/run_generalized_classical_baselines.py)",
            "lambda": LAMBDA,
            "alpha": alpha,
            "matrix_shape": [int(m), int(n)],
            "reference": "dense Ridge solve of (H^T H + alpha I) x = H^T r",
            "timing": f"median of {REPS} repetitions; factorization amortized "
            f"({FACTOR_REPS} repetitions)",
        },
        "sketch": {
            "seed": SKETCH_SEED,
            "sizes": list(SKETCH_SIZES),
            "primary_table_row": f"sketched_Ridge_s{PRIMARY_SKETCH_SIZE}",
            "gaussian_entries": "N(0, 1/s)",
        },
        "evidence_tier": "executed classical baseline",
        "claim_boundary": (
            "Fixed-workload timing/accuracy comparison on one host; not a scaling statement."
        ),
        "manuscript_assets": [
            "manuscript/tables/audit_classical_comparison.tex (extended additively via "
            "scripts/build_final_falsification_manuscript_assets.py --classical-table-only)"
        ],
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_manifest(OUTPUT_DIR)
    print(
        frame[
            [
                "method",
                "per_query_s",
                "per_query_solve_only_s",
                "output_error_vs_ridge",
                "selected_output_rel_error_vs_ridge",
                "is_primary_table_row",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
