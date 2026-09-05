from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.experiments.paper_ready_results import build_paper_ready_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-ready result tables, figures, claims, and audit files."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_ready_results",
        help="Directory where paper-ready artifacts will be written.",
    )
    args = parser.parse_args()

    result = build_paper_ready_results(Path(args.output_dir))
    manifest = result["manifest"]
    print(f"Paper-ready results complete: {result['output_dir']}")
    print(f"Tables: {len(manifest['generated_tables'])}")
    print(f"Figures: {len(manifest['generated_figures'])}")
    print(f"Issues noted: {len(manifest['issues_encountered'])}")


if __name__ == "__main__":
    main()
