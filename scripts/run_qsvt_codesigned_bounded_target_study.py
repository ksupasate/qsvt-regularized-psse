from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.codesigned_bounded_targets import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_DEGREES,
    DEFAULT_TARGET_FAMILIES,
    run_qsvt_codesigned_bounded_target_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Co-designed QSVT-safe bounded-target family study."
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--target-families", nargs="+", default=list(DEFAULT_TARGET_FAMILIES))
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_codesigned_bounded_target_study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "alphas": args.alphas,
        "degrees": args.degrees,
        "target_families": args.target_families,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_codesigned_bounded_target_study(config)
    rows = run["rows"]
    safe = sum(1 for row in rows if bool(row.get("qsvt_safe")))
    deployable = sum(
        1
        for row in rows
        if str(row.get("deployability_class")) in {"general_qsvt_safe", "instance_aware_qsvt_safe"}
    )
    print(f"Wrote co-designed bounded-target study to {run['output_dir']}")
    print(f"Total rows: {len(rows)}; QSVT-safe: {safe}; deployable-class: {deployable}")


if __name__ == "__main__":
    main()
