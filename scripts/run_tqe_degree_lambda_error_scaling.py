#!/usr/bin/env python
"""Runner for Workstream A - degree / normalized-regularization / target-tolerance feasibility map.

Usage:
    MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/run_tqe_degree_lambda_error_scaling.py

Options:
    --config PATH          frozen grid config (default configs/tqe_degree_lambda_error_scaling.yaml)
    --output-dir PATH      output root (default outputs/tqe_degree_lambda_error_scaling)
    --no-phase-synthesis   analytic-only pass (skips phase synthesis; for pipeline validation)
    --progress             print progress every 25 rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_qsvt_se.tqe_extensions.common import load_yaml_config
from robust_qsvt_se.tqe_extensions.degree_lambda_scaling import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    run_degree_lambda_scaling,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-phase-synthesis", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if args.no_phase_synthesis:
        # Materialize an override config in the output dir so the run is still fully
        # self-describing.
        config = load_yaml_config(config_path)
        config["attempt_phase_synthesis"] = False
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        override = out / "_config_no_synthesis.yaml"
        import yaml

        override.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        config_path = str(override)

    summary = run_degree_lambda_scaling(config_path, args.output_dir, progress=args.progress)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
