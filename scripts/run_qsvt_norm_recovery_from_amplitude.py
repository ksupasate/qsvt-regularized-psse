from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.norm_recovery_from_amplitude import (  # noqa: E402
    NORM_RECOVERY_METHODS,
    run_norm_recovery_from_amplitude,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover QSVT update scale from actual amplitude/norm-estimation routines"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--methods", nargs="+", default=list(NORM_RECOVERY_METHODS))
    parser.add_argument("--grover-powers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_norm_recovery_from_amplitude")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_norm_recovery_from_amplitude(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "shots": args.shots,
            "methods": args.methods,
            "grover_powers": args.grover_powers,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    print(f"QSVT norm recovery from amplitude complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
