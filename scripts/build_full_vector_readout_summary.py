from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.full_vector_readout import (  # noqa: E402
    build_full_vector_readout_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate the full-vector readout CSVs into a single summary."
    )
    parser.add_argument("--input-dir", default="outputs/full_vector_readout")
    parser.add_argument("--output-dir", default="outputs/full_vector_readout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_full_vector_readout_summary(
        {"input_dir": args.input_dir, "output_dir": args.output_dir}
    )
    print(f"Wrote full-vector readout summary to {run['output_dir']}")
    for row in run["summary_rows"]:
        print(f"  {row['metric']} = {row['value']}")


if __name__ == "__main__":
    main()
