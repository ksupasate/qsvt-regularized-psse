from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.full_vector_readout import run_full_vector_readout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selected-subproblem full-vector readout prototype: statevector extraction, "
            "shot-based magnitude sampling, relative-sign recovery, and norm/scaling audit "
            "for validated small QSVT subproblems. Not full IEEE-scale readout."
        )
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/full_vector_readout")
    parser.add_argument("--cases", nargs="+", default=["ieee14", "ieee30", "ieee57", "ieee118"])
    parser.add_argument(
        "--subproblem-types",
        nargs="+",
        default=["high_leverage", "metadata_mapped", "residual_supported"],
    )
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--shots", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["exact_qsvt_target_statevector"],
        choices=["exact_qsvt_target_statevector", "gate_level_qsvt_statevector"],
        help="readout modes; gate_level re-runs the actual QSP circuit (slower)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-error", type=float, default=1.0e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_full_vector_readout(
        {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "cases": args.cases,
            "subproblem_types": args.subproblem_types,
            "alpha": args.alpha,
            "shots": args.shots,
            "trials": args.trials,
            "methods": args.methods,
            "seed": args.seed,
            "target_error": args.target_error,
        }
    )
    print(f"Wrote full-vector readout outputs to {run['output_dir']}")
    print(
        f"subproblems={run['subproblem_count']} "
        f"statevector_successes={run['statevector_successes']} "
        f"gate_level_subproblems={run['gate_level_subproblems']} "
        f"missing={len(run['missing'])} "
        f"best_statevector_rel_l2_error={run['best_statevector_error']:.2e} "
        f"mean_reliable_sign_fraction={run['reliable_sign_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
