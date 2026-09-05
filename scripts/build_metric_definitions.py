from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.metric_definitions import build_metric_definitions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: build manuscript metric definitions and interpretation notes."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/final_manuscript_package/metrics")
    parser.add_argument("--package-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"input_root": args.input_root, "output_dir": args.output_dir}
    if args.package_root:
        config["package_root"] = args.package_root
    run = build_metric_definitions(config)
    print(f"Wrote metric definitions to {run['output_dir']}")
    print(f"metrics_defined={run['n_metrics']}")


if __name__ == "__main__":
    main()
