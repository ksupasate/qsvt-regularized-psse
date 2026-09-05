from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.phase_synthesis_refinement import (  # noqa: E402
    run_best_config_sampling_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run shot-based sampling validation for the selected best phase-synthesis "
            "refinement configs (the relaxed-alpha QSVT-validated view) across shot levels and "
            "trials, confirming the reconstruction is stable under stochastic measurement."
        )
    )
    parser.add_argument("--input-dir", default="outputs/phase_synthesis_refinement")
    parser.add_argument("--readout-dir", default="outputs/full_vector_readout")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/phase_synthesis_refinement")
    parser.add_argument("--shots", type=int, nargs="*", default=[1_000, 10_000, 100_000])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_best_config_sampling_validation(
        {
            "input_dir": args.input_dir,
            "readout_dir": args.readout_dir,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "shots": args.shots,
            "trials": args.trials,
            "seed": args.seed,
        }
    )
    print(f"Wrote best-config sampling validation to {run['output_dir']}")
    print(f"trial_rows={len(run['trial_rows'])} summary_rows={len(run['summary_rows'])}")


if __name__ == "__main__":
    main()
