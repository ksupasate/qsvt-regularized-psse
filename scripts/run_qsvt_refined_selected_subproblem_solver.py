from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.refined_selected_subproblem_solver import (  # noqa: E402
    run_refined_selected_subproblem_solver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refined selected-subproblem QSVT solver")
    parser.add_argument(
        "--selection-file",
        default="outputs/qsvt_subproblem_selection_policy/selected_subproblems.csv",
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0e-4])
    parser.add_argument("--degrees", type=int, nargs="+", default=[51])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_refined_selected_subproblem_solver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_refined_selected_subproblem_solver(
        {
            "selection_file": args.selection_file,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"Refined selected-subproblem QSVT solver complete: {run['output_dir']}")
    print(f"runs={len(run['all_rows'])} selected_subproblems={len(run['best_rows'])}")


if __name__ == "__main__":
    main()
