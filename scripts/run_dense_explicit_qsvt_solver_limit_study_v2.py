from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.dense_explicit_limit_study_v2 import (  # noqa: E402
    run_dense_explicit_limit_study_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense explicit QSVT limit study v2")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=35)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/dense_explicit_qsvt_limit_study_v2")
    parser.add_argument("--transpile-qubit-limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_dense_explicit_limit_study_v2(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_sizes": args.submatrix_sizes,
            "alpha": args.alpha,
            "degree": args.degree,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "transpile_qubit_limit": args.transpile_qubit_limit,
        }
    )
    print(f"Dense explicit QSVT limit study v2 complete: {run['output_dir']}")
    print(f"validated_solver_rows={len(run['executed_rows'])}")


if __name__ == "__main__":
    main()
