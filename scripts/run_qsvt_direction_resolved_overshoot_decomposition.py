from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.direction_resolved_overshoot_decomposition import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_CASES,
    DEFAULT_DEGREES,
    DEFAULT_TARGET_FAMILIES,
    run_qsvt_direction_resolved_overshoot_decomposition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction-resolved decomposition of the QSVT overshoot onset by singular "
        "direction."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--target-families", nargs="+", default=list(DEFAULT_TARGET_FAMILIES))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir", default="outputs/qsvt_direction_resolved_overshoot_decomposition"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "cases": args.cases,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "degrees": args.degrees,
        "target_families": args.target_families,
        "alphas": args.alphas,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_direction_resolved_overshoot_decomposition(config)
    print(f"Wrote direction-resolved overshoot decomposition to {run['output_dir']}")
    print(f"Per-direction rows: {len(run['rows'])}")


if __name__ == "__main__":
    main()
