from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.readout_alpha_degree_codesign import (  # noqa: E402
    run_readout_two_view_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the two-view (matched-alpha diagnostic vs per-subproblem QSVT-validated) "
            "gate-level readout summary, figure data, and co-design scope note."
        )
    )
    parser.add_argument("--input-dir", default="outputs/full_vector_readout")
    parser.add_argument("--output-dir", default="outputs/full_vector_readout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_readout_two_view_summary({"input_dir": args.input_dir, "output_dir": args.output_dir})
    rows = run["summary_rows"]
    improved = sum(1 for r in rows if r["final_status"] == "improved_by_codesign")
    limited = sum(1 for r in rows if r["final_status"] == "polynomial_limited_after_sweep")
    print(f"Wrote two-view readout summary to {run['output_dir']}")
    print(
        f"summary_rows={len(rows)} "
        f"improved_by_codesign={improved} "
        f"polynomial_limited_after_sweep={limited} "
        f"figure_two_view_rows={len(run['figure_two_view_rows'])} "
        f"figure_decomposition_rows={len(run['figure_decomposition_rows'])}"
    )


if __name__ == "__main__":
    main()
