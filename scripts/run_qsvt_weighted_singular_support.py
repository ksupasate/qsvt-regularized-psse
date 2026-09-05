from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.weighted_singular_support import (  # noqa: E402
    DEFAULT_CURRENT_DEGREE,
    run_qsvt_weighted_singular_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted singular-support diagnostic for co-designed QSVT targets."
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    )
    parser.add_argument("--current-degree", type=int, default=DEFAULT_CURRENT_DEGREE)
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_weighted_singular_support")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "alphas": args.alphas,
        "current_degree": args.current_degree,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_weighted_singular_support(config)
    rows = run["rows"]
    high = sum(1 for row in rows if bool(row.get("high_weight_direction")))
    print(f"Wrote weighted singular-support outputs to {run['output_dir']}")
    print(f"Total rows: {len(rows)}; high-weight directions: {high}")


if __name__ == "__main__":
    main()
