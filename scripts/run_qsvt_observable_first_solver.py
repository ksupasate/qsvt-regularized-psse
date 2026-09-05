from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.observable_first_qsvt_solver import (  # noqa: E402
    run_observable_first_solver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observable-first QSVT state-estimation solver prototype"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument("--shots", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--target-tolerances", type=float, nargs="+", default=[1e-1, 5e-2, 1e-2])
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_observable_first_solver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_observable_first_solver(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "shots": args.shots,
            "target_tolerances": args.target_tolerances,
            "topk": args.topk,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"QSVT observable-first solver complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
