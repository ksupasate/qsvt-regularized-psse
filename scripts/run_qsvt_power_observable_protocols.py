from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.power_observable_protocols import (  # noqa: E402
    run_qsvt_power_observable_protocols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify power-system observable readout protocols for the QSVT pathway"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument(
        "--target-tolerances",
        type=float,
        nargs="+",
        default=[1.0e-1, 5.0e-2, 1.0e-2, 1.0e-3],
    )
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_power_observable_protocols")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_qsvt_power_observable_protocols(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "target_tolerances": args.target_tolerances,
            "topk": args.topk,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"QSVT power observable protocols complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
