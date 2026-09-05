#!/usr/bin/env python3
"""Run the output-aware resource-constrained sparse-support selection campaign."""

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

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    STAGES,
    make_context,
    run_study,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override only the predeclared random-support seed base",
    )
    args = parser.parse_args(argv)
    context = make_context(
        output_dir=args.output_dir,
        config_path=args.config,
        resume=args.resume,
        force=args.force,
        max_workers=args.max_workers,
        seed=args.seed,
    )
    result = run_study(context, stage=args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
