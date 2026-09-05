# Troubleshooting

## `python: command not found`

Activate the environment first:

```bash
conda activate qsvt-se
python --version
```

For a venv, use `.venv/bin/python` on macOS/Linux or
`.venv\Scripts\python.exe` on Windows.

## `ModuleNotFoundError: robust_qsvt_se`

Install the checked-out package from the repository root:

```bash
python -m pip install -e . --no-deps
```

The public entry scripts also add `src/` to their import path, but direct
`python -m robust_qsvt_se...` commands require installation.

## A pinned package cannot be installed

Confirm CPython 3.12 and a 64-bit environment. Quantum simulator wheels can be
platform-specific. If a platform cannot install the full stack, create the core
environment from `pyproject.toml` and run:

```bash
python scripts/validate_reproduction.py --core-only --skip-regeneration
```

This is only a reduced preflight. It does not certify QSVT or figure-generation
paths, and warnings remain in the report.

## The validator reports missing output folders

Raw campaign output trees are not ordinary Git objects. Restore the DOI-backed
data archive, when available, or rerun the registered config with the safe
wrapper:

```bash
python scripts/run_experiment.py --config CONFIG_PATH
```

The rerun goes to `outputs/generated/`. If the manifest points to a canonical
released path, retain the failure as an availability finding; do not copy an
unrelated run into that directory merely to satisfy validation.

## `output directory is nonempty`

This occurs only when `--fail-if-exists` was requested. Choose a new explicit
directory or rerun without that guard. The default destination is regenerable
and is separate from canonical evidence.

## A long sweep was interrupted

For standard sweep configurations with checkpoint files:

```bash
python scripts/run_experiment.py --config CONFIG_PATH --resume
```

Use the exact same config and output destination. `trial_results.jsonl`,
`checkpoint_state.json`, and `progress.log` document progress. See
`docs/ieee300_run_guide.md` for persistent-shell examples.

## Matplotlib cannot open a display

The reporting code uses a noninteractive backend. Ensure `MPLBACKEND=Agg` if a
site-specific matplotlib configuration overrides it:

```bash
MPLBACKEND=Agg python scripts/generate_figures.py
```

This changes rendering infrastructure, not scientific inputs.

## Phase synthesis is slow or reports a backend error

Verify the pinned PennyLane, pyqsp, NumPy, and SciPy versions. The safe runner
uses `outputs/generated/_phase_cache/`; canonical phase data are never updated.
Do not substitute phases from a different degree, parity, domain, scaling, or
convention. Preserve a structured failure if the requested backend is
unavailable.

## Qiskit circuit results differ slightly

Check Qiskit, Aer, transpiler basis gates, optimization level, simulator seed,
and shot count in the resolved config. Gate counts and timings can vary across
versions even when numerical action agrees. Compare the declared accuracy
metric and tolerance rather than assuming identical transpilation.

## Numerical values differ at the last digits

BLAS/LAPACK reduction order, platform libraries, and solver builds can create
small floating-point differences. Confirm:

- the same resolved config and seeds;
- the direct dependency versions in `requirements.txt`;
- the same case source (`builtin` versus `pypower`);
- the same measurement inclusion switches and standard deviations;
- the applicable test tolerance.

Large differences require diagnosis; do not widen tests merely to hide them.

## Ruff scans generated or frozen material

Run Ruff from the repository root so it reads `pyproject.toml`:

```bash
ruff check .
```

The configured exclusions cover generated outputs, local archives, and caches.
Maintained `src/`, `scripts/`, and `tests/` code remains in scope.

## A frozen-evidence checksum fails after a rerun

Researcher reruns should not write to canonical evidence paths. Inspect the
selected output destination first. Do not refresh phase banks, registries, or
checksums as an automatic fix; those are protected scientific evidence and
require an intentional evidence-release workflow.
