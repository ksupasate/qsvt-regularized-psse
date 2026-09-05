from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.norm_residual_gap_audit import run_norm_residual_gap_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT norm/residual-gap audit")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_norm_residual_gap_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_norm_residual_gap_audit(vars(args))
    summary = run["summary"]
    print(f"Norm/residual-gap audit complete: {run['output_dir']}")
    print(
        "best_scalar_residual="
        f"{summary['residual_qsvt_best_scalar']:.3e} "
        "dominant_gap_source="
        f"{summary['dominant_gap_source']}"
    )


if __name__ == "__main__":
    main()
