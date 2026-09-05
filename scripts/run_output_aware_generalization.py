#!/usr/bin/env python3
"""Run the frozen multi-instance output-aware sparsification benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_qsvt_se.qsvt.output_aware_generalization import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    STAGES,
    make_context,
    run_generalization_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = make_context(
        output_dir=args.output_dir,
        config_path=args.config,
        resume=args.resume,
        force=args.force,
        max_workers=args.max_workers,
        seed=args.seed,
    )
    result = run_generalization_study(context, stage=args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
