from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.gate_level_qsvt import run_gate_level_qsvt_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small gate-level dense QSVT demo")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/gate_level_qsvt_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_gate_level_qsvt_demo(
        {
            "case": args.case,
            "case_name": args.case,
            "matrix_source": args.matrix_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    comparison = run["statevector_comparison"]
    print(f"Gate-level QSVT demo complete: {run['output_dir']}")
    print(
        "state_error="
        f"{comparison['best_sign_state_l2_error_vs_ridge']:.3e} "
        f"success_probability={comparison['success_probability']:.3e}"
    )


if __name__ == "__main__":
    main()
