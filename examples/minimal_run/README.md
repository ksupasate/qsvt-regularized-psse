# Minimal Run

This example is a six-state, eighteen-row controlled weighted system. It is
intended to verify installation, deterministic seeding, estimator dispatch, and
artifact writing. It is not an IEEE network or a physical measurement model.

Run from the repository root:

```bash
python scripts/run_experiment.py \
  --config examples/minimal_run/config.yaml \
  --output-dir outputs/examples/minimal_run
```

Expected artifacts are documented in `examples/expected_outputs/README.md`.
Ridge and `qsvt_regularized` use the same `alpha` and should agree to numerical
precision because this example evaluates the exact classical target, not a
quantum circuit.
