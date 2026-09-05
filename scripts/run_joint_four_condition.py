#!/usr/bin/env python3
"""Entrypoint for the joint four-condition single-candidate pipeline.

Runs the full deterministic pipeline (freeze -> benchmark reference -> statevector validation
-> finite-shot reproduction -> resources -> classical comparators -> four-condition decision)
and writes ``outputs/joint_four_condition/``.

Reproduce:
    MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/run_joint_four_condition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.tqe_extensions.joint_four_condition import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_joint_four_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_joint_four_condition(args.config, args.output, progress=not args.quiet)


if __name__ == "__main__":
    main()
