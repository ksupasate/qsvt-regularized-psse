from __future__ import annotations

import argparse

from robust_qsvt_se.qsvt.pennylane_demo import load_pennylane_demo_config, run_pennylane_demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a tiny PennyLane QSVT demo")
    parser.add_argument("--config", required=True, help="Path to a PennyLane QSVT demo config")
    args = parser.parse_args(argv)
    run_pennylane_demo(load_pennylane_demo_config(args.config))


if __name__ == "__main__":
    main()
