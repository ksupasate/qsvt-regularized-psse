from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.experiments.report_builder import build_report, load_report_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manuscript-oriented tables and figures from completed benchmark outputs."
    )
    parser.add_argument("--config", required=True, help="Path to YAML report config.")
    args = parser.parse_args()

    config = load_report_config(Path(args.config))
    report = build_report(config)
    manifest = report["manifest"]
    print(f"Report complete: {report['output_dir']}")
    print(f"Metric rows: {manifest['n_metric_rows']}")
    print(f"Summary rows: {manifest['n_summary_rows']}")
    print(f"PDF compiled: {manifest['pdf_compiled']}")


if __name__ == "__main__":
    main()
