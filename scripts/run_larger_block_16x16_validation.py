#!/usr/bin/env python3
"""Run Track B - larger-block (16x16 IEEE-14) validation."""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from robust_qsvt_se.cross_case_validation.larger_block import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    run_larger_block,
)

CONFIG_ROOT = Path("configs/cross_case_larger_block_validation")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    config = args.config or (
        CONFIG_ROOT
        / ("larger_block_16x16_smoke.json" if args.mode == "smoke" else "larger_block_16x16.json")
    )
    result = run_larger_block(config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
