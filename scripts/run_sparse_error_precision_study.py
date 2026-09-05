#!/usr/bin/env python3
"""Run the sparse QSVT error-source ablation and precision-resource study.

Conservative single-threaded numerical defaults are applied before any heavy import
(the previous integrated campaign hit an OpenMP shared-memory error otherwise).
"""

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

from robust_qsvt_se.qsvt.sparse_error_precision_study import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_STUDY_DIR,
    STAGES,
    make_context,
    run_study,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", *STAGES],
        help="single stage to run, or 'all' for the full ordered campaign",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip units whose checkpoint parts are verified complete",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="clear checkpoints for the selected stage(s) and recompute",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="worker cap; heavy statevector/sampling stages always run serially "
        "(conservative OpenMP-safe configuration), the value is recorded for provenance",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="base simulator seed (seeds = seed..seed+9)"
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
    results = run_study(context, stage=args.stage)
    print(json.dumps({stage: result for stage, result in results.items()}, indent=2,
                     default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
