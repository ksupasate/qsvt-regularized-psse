from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.success_amplification_cost import (  # noqa: E402
    run_success_amplification_cost_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT success/amplification cost study")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            "outputs/gate_level_qsvt_ieee14_solver",
            "outputs/qsvt_matrix_free_ieee_experiments",
            "outputs/qsvt_matrix_free_ieee_resource_only",
        ],
    )
    parser.add_argument("--output-dir", default="outputs/qsvt_success_amplification_cost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_success_amplification_cost_study(
        {"input_dirs": args.input_dirs, "output_dir": args.output_dir}
    )
    print(f"QSVT success/amplification cost study complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
