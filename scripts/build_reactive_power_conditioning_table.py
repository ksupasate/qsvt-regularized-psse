from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.reactive_power_conditioning_table import (  # noqa: E402
    build_reactive_power_conditioning_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: build the reactive-power / P-Q conditioning paper-facing table."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir", default="outputs/final_manuscript_package/reactive_power_conditioning"
    )
    parser.add_argument("--package-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"input_root": args.input_root, "output_dir": args.output_dir}
    if args.package_root:
        config["package_root"] = args.package_root
    run = build_reactive_power_conditioning_table(config)
    print(f"Wrote reactive-power conditioning table to {run['output_dir']}")
    print(f"rows={run['n_rows']} cases={run['cases']} missing={len(run['missing_rows'])}")


if __name__ == "__main__":
    main()
