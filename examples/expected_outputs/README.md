# Expected Output Schemas

## Single standard run

`metrics.csv` must include at least:

```text
estimator
rmse
residual_norm
condition_number
runtime_seconds
failed
```

Other expected files are `config_resolved.yaml`, `estimator_results.json`,
`singular_values.csv`, and `run.log`. When enabled, resource diagnostics and
plots are additional artifacts.

## Standard sweep

`aggregate_metrics.csv` adds trial identifiers, sweep parameter/value, and
seed. `summary_metrics.csv` must include grouped estimator, RMSE mean, condition
number mean, runtime mean, and failure rate. Checkpoint-capable runs may also
contain `trial_results.jsonl`, `checkpoint_state.json`, and `progress.log`.

## QSVT phase response

`qsp_validation_grid.csv` must identify the normalized singular value, exact
target, implemented phase response, and absolute phase-response error. The
resolved config fixes degree, parity, domain, scaling, method, and seed.

## Circuit scaling

`circuit_scaling_results.csv` must retain the case, selected matrix size,
polynomial degree, status, feasibility, and error relative to the matched
classical polynomial action. Infeasible and failed rows are expected evidence,
not records to discard.

## Resource estimation

`qsvt_resource_estimates.csv` must retain matrix dimensions, condition number,
degree, estimated qubits, and query/resource proxies. These values are model
estimates, not hardware execution measurements.

Run `python scripts/validate_outputs.py` to validate the registered publication
outputs against these minimum schemas.
