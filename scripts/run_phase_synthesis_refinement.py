from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.phase_synthesis_refinement import (  # noqa: E402
    run_phase_synthesis_refinement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QSP phase-synthesis and finite-degree approximation refinement for the readout-"
            "limited selected QSVT subproblems: freeze the baseline, run a degree 47/49 "
            "diagnostic-only sweep, compare phase-synthesis solver variants and weighted "
            "singular-support fits, and emit the three-view summary, leakage audit, failure "
            "taxonomy, cost trade-offs, scope note, and figure data."
        )
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--readout-dir", default="outputs/full_vector_readout")
    parser.add_argument("--output-dir", default="outputs/phase_synthesis_refinement")
    parser.add_argument(
        "--targeted-configs-from",
        default="outputs/full_vector_readout/readout_two_view_summary.csv",
        help="Co-design two-view summary used to freeze the baseline (informational).",
    )
    parser.add_argument("--degree-grid-extra", type=int, nargs="*", default=[47, 49])
    parser.add_argument(
        "--alpha-grid", type=float, nargs="*", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    )
    parser.add_argument("--refinement-degrees", type=int, nargs="*", default=[41, 45])
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-error", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_phase_synthesis_refinement(
        {
            "input_root": args.input_root,
            "readout_dir": args.readout_dir,
            "output_dir": args.output_dir,
            "degree_grid_extra": args.degree_grid_extra,
            "alpha_grid": args.alpha_grid,
            "refinement_degrees": args.refinement_degrees,
            "shots": args.shots,
            "trials": args.trials,
            "seed": args.seed,
            "target_error": args.target_error,
        }
    )
    three_view = run["three_view_rows"]
    relaxed = [r for r in three_view if r["view"] == "per_subproblem_relaxed_alpha_qsvt_validated"]
    passing = sum(1 for r in relaxed if r["status"] == "pass")
    improved = sum(1 for r in relaxed if r["status"] == "improved_but_polynomial_limited")
    limited = sum(1 for r in relaxed if r["status"] == "qsvt_polynomial_limited")
    print(f"Wrote phase-synthesis refinement artifacts to {run['output_dir']}")
    print(
        f"targeted={len(run['targeted'])} "
        f"degree_47_49_rows={len(run['diagnostic_rows'])} "
        f"variant_rows={len(run['variant_rows'])} "
        f"fit_comparison_rows={len(run['fit_comparison_rows'])} "
        f"three_view_rows={len(three_view)}"
    )
    print(
        f"relaxed-view: pass={passing} "
        f"improved_but_polynomial_limited={improved} "
        f"qsvt_polynomial_limited={limited}"
    )


if __name__ == "__main__":
    main()
