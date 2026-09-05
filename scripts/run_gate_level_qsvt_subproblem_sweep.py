from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.subproblem_sweep import run_gate_level_qsvt_subproblem_sweep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gate-level QSVT subproblem sweep")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--selection-modes", nargs="+", default=["high_leverage"])
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/gate_level_qsvt_subproblem_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_gate_level_qsvt_subproblem_sweep(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "selection_modes": args.selection_modes,
            "alpha": args.alpha,
            "degree": args.degree,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"Gate-level QSVT subproblem sweep complete: {run['output_dir']}")
    print(f"subproblems={len(run['summary_rows'])}")


if __name__ == "__main__":
    main()
