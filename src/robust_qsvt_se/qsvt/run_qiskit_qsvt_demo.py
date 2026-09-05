from __future__ import annotations

import argparse

from robust_qsvt_se.qsvt.qiskit_demo import load_qiskit_demo_config, run_qiskit_demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Qiskit QSP/QSVT-style demo")
    parser.add_argument("--config", required=True, help="Path to a Qiskit demo config")
    args = parser.parse_args(argv)
    run_qiskit_demo(load_qiskit_demo_config(args.config))


if __name__ == "__main__":
    main()
