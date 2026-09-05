from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.power_observable_mapping import run_power_observable_readout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT power-observable readout mapping")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_power_observable_readout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_power_observable_readout(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"QSVT power-observable readout complete: {run['output_dir']}")
    print(f"observables={len(run['exact_rows'])}")


if __name__ == "__main__":
    main()
