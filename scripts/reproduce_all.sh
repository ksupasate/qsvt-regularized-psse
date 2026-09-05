#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(dirname -- "$script_dir")"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON_BIN:-python}"
validation_report="outputs/generated/reproduction_validation.md"
example_output="outputs/examples/ieee14_quickstart"

printf 'Repository: qsvt-regularized-psse\n'
printf 'Python command: %s\n' "$python_bin"
"$python_bin" --version

"$python_bin" -c 'import numpy, pandas, scipy, yaml; import robust_qsvt_se; print("Core imports: OK")'

printf 'Running reproducibility validation...\n'
"$python_bin" scripts/validate_reproduction.py --report "$validation_report"

printf 'Running deterministic IEEE14 quickstart...\n'
"$python_bin" scripts/run_experiment.py \
  --config examples/ieee14_quickstart/config.yaml \
  --output-dir "$example_output"

printf 'Generated example files:\n'
find "$example_output" -type f -print | sort

printf 'Validation report: %s\n' "$validation_report"
printf 'Lightweight reproduction workflow: PASS\n'
