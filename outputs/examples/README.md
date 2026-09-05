# Example Outputs

This directory is reserved for isolated smoke and minimal-example runs. The
generated subdirectories are ignored by Git because runtime fields, logs, and
platform-specific floating-point formatting are not canonical evidence.

Generate the smoke outputs with:

```bash
python scripts/run_smoke_test.py
```

Expected schemas are described in `examples/expected_outputs/README.md` and
checked by `scripts/validate_outputs.py`.
