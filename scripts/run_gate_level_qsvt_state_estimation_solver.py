from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.gate_level_qsvt_convention import (  # noqa: E402
    run_gate_level_qsvt_convention_debug,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (  # noqa: E402
    run_ieee_gate_level_state_estimation_solver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small gate-level QSVT state-estimation solver"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument(
        "--model",
        default="ac_linearized",
        choices=["ac_linearized", "dc_linearized"],
    )
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/gate_level_qsvt_ieee14_solver")
    parser.add_argument("--observable-mode", default="default")
    parser.add_argument("--use-corrected-extraction-rule", action="store_true")
    parser.add_argument("--attempt-rescaled-update", action="store_true")
    parser.add_argument("--transpile-qubit-limit", type=int, default=4)
    parser.add_argument(
        "--skip-convention-debug",
        action="store_true",
        help="Do not refresh outputs/gate_level_qsvt_convention_debug before solving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_convention_debug:
        run_gate_level_qsvt_convention_debug()
    run = run_ieee_gate_level_state_estimation_solver(
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
            "observable_mode": args.observable_mode,
            "use_corrected_extraction_rule": bool(args.use_corrected_extraction_rule),
            "attempt_rescaled_update": bool(args.attempt_rescaled_update),
            "transpile_qubit_limit": args.transpile_qubit_limit,
        }
    )
    summary = run["summary"]
    print(f"Gate-level QSVT state-estimation solver complete: {run['output_dir']}")
    print(
        "state_error="
        f"{summary['phase_or_sign_aligned_state_error']:.3e} "
        "residual_ratio="
        f"{summary['residual_reduction_ratio_vs_no_update']:.3e} "
        "success_probability="
        f"{summary['success_probability']:.3e}"
    )


if __name__ == "__main__":
    main()
