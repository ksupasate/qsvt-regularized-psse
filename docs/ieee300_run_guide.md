# IEEE300 Nonlinear Run Guide

The full IEEE300 nonlinear AC seed10 config is:

```text
configs/nonlinear_ac_ieee300_seed10.yaml
```

It contains 90 planned trials:

- 3 independent sweeps;
- 3 values per sweep;
- 10 seeds per value;
- 5 estimators per trial;
- up to 8 nonlinear Gauss-Newton iterations per estimator.

On the current laptop, an earlier full run was stopped after roughly one hour
because it was still early in the first sweep. The run was later resumed and
completed successfully. The completed output lives at
`outputs/nonlinear_ac_ieee300_seed10/` and is valid paper-level nonlinear
IEEE300 evidence.

## Resume Files

The runner writes:

```text
outputs/nonlinear_ac_ieee300_seed10/trial_results.jsonl
outputs/nonlinear_ac_ieee300_seed10/checkpoint_state.json
outputs/nonlinear_ac_ieee300_seed10/progress.log
```

Completed and failed trials are skipped on `--resume`. At the end, the runner
regenerates the standard CSV/JSON artifacts.

## Local Command

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
.venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \
  --config configs/nonlinear_ac_ieee300_seed10.yaml \
  --resume
```

## nohup Command

```bash
mkdir -p outputs/nonlinear_ac_ieee300_seed10

nohup env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
.venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \
  --config configs/nonlinear_ac_ieee300_seed10.yaml \
  --resume \
  > outputs/nonlinear_ac_ieee300_seed10/nohup.out 2>&1 &
```

## tmux Command

```bash
tmux new -s ieee300_nonlinear
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
.venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \
  --config configs/nonlinear_ac_ieee300_seed10.yaml \
  --resume
```

## Completion Check

The full run is complete only when:

- `checkpoint_state.json` has `"status": "complete"`;
- `remaining_trials` is `0`;
- `aggregate_metrics.csv`, `summary_metrics.csv`, `trial_results.json`,
  `trial_results.jsonl`, `singular_values.csv`, `iteration_trace.csv`, and
  `run.log` exist;
- no required trial is missing from `trial_results.jsonl`.

The reduced-runtime IEEE300 output remains a diagnostic fallback and must not be
described as full seed10 nonlinear IEEE300 evidence.

The current completed run satisfies these checks: `checkpoint_state.json`
reports `"status": "complete"`, 90 completed trials, 0 remaining trials, and 0
failed trials.
