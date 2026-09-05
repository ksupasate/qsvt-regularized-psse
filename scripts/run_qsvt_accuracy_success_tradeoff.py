from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.accuracy_success_tradeoff import (  # noqa: E402
    run_accuracy_success_tradeoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT accuracy-success tradeoff")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            "outputs/qsvt_alpha_degree_residual_refinement",
            "outputs/qsvt_refined_selected_subproblem_solver",
            "outputs/qsvt_success_amplification_cost",
        ],
    )
    parser.add_argument("--output-dir", default="outputs/qsvt_accuracy_success_tradeoff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_accuracy_success_tradeoff(
        {"input_dirs": args.input_dirs, "output_dir": args.output_dir}
    )
    print(f"QSVT accuracy-success tradeoff complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
