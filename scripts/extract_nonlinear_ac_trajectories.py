from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.nonlinear_trajectory_extraction import (  # noqa: E402
    build_nonlinear_trajectory_extraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: extract nonlinear AC per-iteration trajectory figure data."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir", default="outputs/final_manuscript_package/nonlinear_trajectories"
    )
    parser.add_argument("--package-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"input_root": args.input_root, "output_dir": args.output_dir}
    if args.package_root:
        config["package_root"] = args.package_root
    run = build_nonlinear_trajectory_extraction(config)
    print(f"Wrote nonlinear trajectories to {run['output_dir']}")
    print(f"logs_found={run['logs_found']} extracted_rows={run['n_extracted']} ")
    print(f"missing={len(run['missing_rows'])}")


if __name__ == "__main__":
    main()
