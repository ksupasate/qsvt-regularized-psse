#!/usr/bin/env python3
"""Run the end-to-end 8x8 sparse-access selected-output QSVT experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_SHOT_COUNTS,
    run_sparse_integrated_chain,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHOT_COUNTS),
        help="per-circuit shot budgets",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="deterministic simulator seeds",
    )
    args = parser.parse_args(argv)
    run = run_sparse_integrated_chain(
        {
            "output_dir": args.output_dir,
            "shot_counts": tuple(args.shots),
            "seeds": tuple(args.seeds),
        }
    )
    print(f"Configuration: {run['inputs'].config.configuration_id}")
    print(f"Artifacts: {run['output_dir']}")
    print(run["statevector_frame"].to_string(index=False, max_colwidth=40))
    print(run["finite_shot_summary"].to_string(index=False, max_colwidth=40))


if __name__ == "__main__":  # pragma: no cover
    main()
