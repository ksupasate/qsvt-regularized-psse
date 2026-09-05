from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.readout_alpha_degree_codesign import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    DEFAULT_DEGREE_GRID,
    run_alpha_degree_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted alpha/degree co-design sweep for high-error gate-level full-vector readout "
            "configs. Preserves the matched-alpha diagnostic view and adds a per-subproblem "
            "QSVT-validated view. Not full IEEE-scale readout."
        )
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/full_vector_readout")
    parser.add_argument(
        "--diagnostic-file",
        default="outputs/full_vector_readout/readout_config_diagnostic_classification.csv",
        help="path the diagnostic classification is (re)written to (informational)",
    )
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(DEFAULT_ALPHA_GRID))
    parser.add_argument("--degree-grid", nargs="+", type=int, default=list(DEFAULT_DEGREE_GRID))
    parser.add_argument(
        "--overshoot-degrees",
        nargs="*",
        type=int,
        default=[],
        help="diagnostic-only overshoot degrees; never selected as best configs",
    )
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-error", type=float, default=1.0e-2)
    parser.add_argument("--mandatory-case", default="ieee14")
    parser.add_argument("--mandatory-subproblem-type", default="metadata_mapped")
    parser.add_argument("--include-top-k-high-error", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_alpha_degree_sweep(
        {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "alpha_grid": args.alpha_grid,
            "degree_grid": args.degree_grid,
            "overshoot_degrees": args.overshoot_degrees,
            "shots": args.shots,
            "trials": args.trials,
            "seed": args.seed,
            "target_error": args.target_error,
            "mandatory": [(args.mandatory_case, args.mandatory_subproblem_type)],
            "top_k_high_error": args.include_top_k_high_error,
        }
    )
    improved = sum(1 for r in run["best_rows"] if r["status"] == "improved_by_codesign")
    limited = sum(1 for r in run["best_rows"] if r["status"] == "qsvt_polynomial_limited")
    print(f"Wrote alpha/degree co-design sweep to {run['output_dir']}")
    print(
        f"targeted={len(run['targeted'])} "
        f"sweep_rows={len(run['sweep_rows'])} "
        f"best_configs={len(run['best_rows'])} "
        f"improved_by_codesign={improved} "
        f"qsvt_polynomial_limited={limited}"
    )


if __name__ == "__main__":
    main()
