#!/usr/bin/env python
"""Runner for Workstream C - measurement-row-level sparsification.

    MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/run_tqe_measurement_row_sparsification.py
"""

from __future__ import annotations

import argparse
import json

from robust_qsvt_se.tqe_extensions.row_sparsification import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    run_row_sparsification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    summary = run_row_sparsification(args.config, args.output_dir, progress=args.progress)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
