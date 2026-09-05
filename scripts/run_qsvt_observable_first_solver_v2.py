from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.observable_first_solver_v2 import (  # noqa: E402
    run_qsvt_observable_first_solver_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observable-first QSVT solver v2 using co-designed targets."
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument(
        "--input",
        default="outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1.0e-2)
    parser.add_argument("--max-configs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_observable_first_solver_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "input": args.input,
        "shots": args.shots,
        "topk": args.topk,
        "tolerance": args.tolerance,
        "max_configs": args.max_configs,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_observable_first_solver_v2(config)
    rows = run["rows"]
    practical = sum(
        1 for row in rows if str(row.get("practical_status")) != "full_vector_like_not_recommended"
    )
    print(f"Wrote observable-first v2 outputs to {run['output_dir']}")
    print(f"Total rows: {len(rows)}; practical rows: {practical}")


if __name__ == "__main__":
    main()
