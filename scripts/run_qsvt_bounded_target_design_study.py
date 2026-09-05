from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.bounded_target_designs import (  # noqa: E402
    run_qsvt_bounded_target_design_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QSVT alternative bounded-target design study"
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
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument(
        "--margin-factors",
        type=float,
        nargs="+",
        default=[1.01, 1.05, 1.10, 1.25, 1.50],
    )
    parser.add_argument("--gain-cap-factor", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_bounded_target_design_study",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_qsvt_bounded_target_design_study(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "margin_factors": args.margin_factors,
            "gain_cap_factor": args.gain_cap_factor,
            "grid_size": args.grid_size,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    completed = [row for row in run["rows"] if row["run_status"] == "completed"]
    safe = [row for row in completed if row.get("qsvt_compatibility_status") == "qsvt_safe"]
    print(f"QSVT bounded-target design study complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])} completed={len(completed)} qsvt_safe={len(safe)}")


if __name__ == "__main__":
    main()
