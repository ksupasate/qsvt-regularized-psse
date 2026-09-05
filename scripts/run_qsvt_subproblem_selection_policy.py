from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.subproblem_selection_policy import (  # noqa: E402
    run_qsvt_subproblem_selection_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT subproblem selection policy")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument(
        "--candidate-modes",
        nargs="+",
        default=[
            "best_conditioned",
            "high_leverage",
            "metadata_mapped",
            "residual_supported",
            "random_seeded_pool",
            "worst_conditioned_control",
        ],
    )
    parser.add_argument("--condition-threshold", type=float, default=1.0e8)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_subproblem_selection_policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_qsvt_subproblem_selection_policy(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "candidate_modes": args.candidate_modes,
            "condition_threshold": args.condition_threshold,
            "alpha": args.alpha,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    selected = [row for row in run["candidate_rows"] if row["selected"]]
    print(f"QSVT subproblem selection policy complete: {run['output_dir']}")
    print(f"candidates={len(run['candidate_rows'])} selected={len(selected)}")


if __name__ == "__main__":
    main()
