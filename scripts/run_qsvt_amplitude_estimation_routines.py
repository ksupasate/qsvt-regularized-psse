from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.amplitude_estimation_routines import (  # noqa: E402
    run_amplitude_estimation_routines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actual small-circuit amplitude-estimation routines for gate-level QSVT"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--grover-powers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_amplitude_estimation_routines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_amplitude_estimation_routines(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "shots": args.shots,
            "grover_powers": args.grover_powers,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    succeeded = [row for row in run["rows"] if row["routine_status"] == "succeeded"]
    print(f"QSVT amplitude-estimation routines complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])} succeeded={len(succeeded)}")


if __name__ == "__main__":
    main()
