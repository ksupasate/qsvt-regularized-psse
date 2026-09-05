from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.alpha_sensitivity_consolidation import (  # noqa: E402
    build_alpha_sensitivity_consolidation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: consolidate alpha-sensitivity evidence from existing artifacts."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/phase3_alpha_sensitivity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_alpha_sensitivity_consolidation(
        {
            "input_root": args.input_root,
            "config_root": args.config_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote Phase 3 alpha-sensitivity consolidation to {run['output_dir']}")
    print(
        f"sensitivity_rows={len(run['sensitivity_rows'])} "
        f"with_alpha={run['rows_with_alpha']} without_alpha={run['rows_without_alpha']} "
        f"tradeoff_rows={len(run['tradeoff_rows'])} alpha_grid={run['alpha_grid']}"
    )


if __name__ == "__main__":
    main()
