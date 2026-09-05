#!/usr/bin/env python3
"""Regenerate derived research figures into an isolated output bundle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/generated/paper_ready_results",
        help="Destination for the regenerated table/figure bundle.",
    )
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    cache_root = REPO_ROOT / "outputs" / "generated" / "_runtime_cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    from robust_qsvt_se.experiments.paper_ready_results import build_paper_ready_results

    result = build_paper_ready_results(args.output_dir)
    figures = result["manifest"]["generated_figures"]
    print(f"Regenerated {len(figures)} figure bundles in {result['output_dir']}:")
    for name, paths in sorted(figures.items()):
        print(f"  - {name}: {paths}")
    print("The shared builder also refreshes tables in the same isolated bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
