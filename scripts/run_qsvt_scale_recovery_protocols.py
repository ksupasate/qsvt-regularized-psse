from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.scale_recovery_protocols import (  # noqa: E402
    run_scale_recovery_protocols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT scale-recovery protocol comparison")
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--eps-rel", type=float, default=1.0e-2)
    parser.add_argument("--amplitude-max-queries", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_scale_recovery_protocols",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_scale_recovery_protocols(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "seed": args.seed,
            "grid_size": args.grid_size,
            "eps_rel": args.eps_rel,
            "amplitude_max_queries": args.amplitude_max_queries,
            "output_dir": args.output_dir,
        }
    )
    print(f"QSVT scale-recovery protocol comparison complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
