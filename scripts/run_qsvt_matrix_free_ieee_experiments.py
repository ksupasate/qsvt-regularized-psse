from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.matrix_free_qsvt_action import (  # noqa: E402
    run_matrix_free_ieee_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["ieee14", "ieee30", "ieee57"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0e-4, 1.0e-3, 1.0e-2])
    parser.add_argument("--degrees", nargs="+", type=int, default=[35, 51, 75])
    parser.add_argument(
        "--method",
        choices=["chebyshev", "cg_krylov"],
        default="cg_krylov",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--resource-estimate-only",
        action="store_true",
        help="Write resource rows without running matrix-free polynomial action.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_matrix_free_ieee_experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_matrix_free_ieee_experiments(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "method": args.method,
            "seed": args.seed,
            "resource_estimate_only": args.resource_estimate_only,
        }
    )
    summary = run["summary"]
    best = summary.get("relative_error_vs_ridge")
    best_text = "n/a" if best is None or best.dropna().empty else f"{best.min():.3e}"
    print(f"Matrix-free IEEE QSVT action complete: {run['output_dir']}")
    print(f"rows={len(summary)} best_relative_error_vs_ridge={best_text}")


if __name__ == "__main__":
    main()
