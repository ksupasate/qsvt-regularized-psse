#!/usr/bin/env python3
"""Build cross-track comparison tables, figures, and the comparison report.

Run after both track runs (and the IEEE-14 8x8 reference) have completed.
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

from robust_qsvt_se.cross_case_validation.comparison import build_all_comparisons  # noqa: E402
from robust_qsvt_se.cross_case_validation.figures import render_all_figures  # noqa: E402

DEFAULT_ROOT = Path("outputs/cross_case_larger_block_validation")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    comparison = build_all_comparisons(args.root)
    figures = render_all_figures(args.root)
    print(json.dumps({"comparison": comparison, "figures": figures}, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
