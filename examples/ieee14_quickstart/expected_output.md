# Expected Output

The command creates `outputs/examples/ieee14_quickstart/` and prints elapsed
runtime plus every generated file. The directory should contain at least:

| File | Purpose |
|---|---|
| `config_resolved.yaml` | Complete resolved configuration and runtime provenance |
| `metrics.csv` | Estimator-level RMSE, residual, conditioning, runtime, and failure fields |
| `estimator_results.json` | Structured estimator results |
| `singular_values.csv` | Weighted-system singular-value diagnostics |
| `qsvt_resource_estimates.csv` | Degree/resource proxy values for the configured grid |
| `run.log` | Human-readable execution summary |

Plot files may also be created when the configured plotting dependencies are
available. Exact wall-clock values and floating-point formatting can vary by
platform; compare numerical fields using repository tolerances rather than
byte-for-byte equality.

The run is successful when the command exits with status zero, all required
files are present, and Ridge and `qsvt_regularized` remain equal within numerical
precision for their identical `alpha`.
