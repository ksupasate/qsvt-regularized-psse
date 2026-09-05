from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import run_alpha_degree_tradeoff  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QSVT alpha-degree tradeoff study")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument("--submatrix-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
    )
    parser.add_argument("--degrees", nargs="+", type=int, default=[15, 25, 35, 51, 75, 101])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_alpha_degree_tradeoff")
    args = parser.parse_args()
    run = run_alpha_degree_tradeoff(
        {
            "case": args.case,
            "case_source": args.case_source,
            "matrix_source": args.matrix_source,
            "submatrix_sizes": args.submatrix_sizes,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    summary = run["summary"]
    ok = summary[summary["status"] == "ok"]
    best_error = float(ok["qsvt_state_error_vs_ridge"].min()) if not ok.empty else float("nan")
    print(f"QSVT alpha-degree tradeoff complete: {run['output_dir']}")
    print(f"rows={len(summary)} best_state_error={best_error:.3e}")


if __name__ == "__main__":
    main()
