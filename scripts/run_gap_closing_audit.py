from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.gap_closing_audit import build_gap_closing_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: triage remaining manuscript evidence gaps without fabricating."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/gap_closing_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_gap_closing_audit({"input_root": args.input_root, "output_dir": args.output_dir})
    print(f"Wrote gap-closing audit to {run['output_dir']}")
    print(
        f"triaged={len(run['triage_rows'])} closeable={len(run['closeable_rows'])} "
        f"new_fast_runs={len(run['new_run_rows'])} "
        f"must_remain_missing={len(run['must_remain_rows'])}"
    )


if __name__ == "__main__":
    main()
