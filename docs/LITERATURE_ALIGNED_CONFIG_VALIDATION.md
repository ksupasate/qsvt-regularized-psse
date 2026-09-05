# Literature-Aligned Config Validation

These configs are literature-inspired approximations, not field-calibrated PMU/SCADA measurement datasets.

Validated draft configs:

- `configs/literature_aligned_ieee14_wls.yaml`
- `configs/literature_aligned_ieee30_robust_ac.yaml`

## 1. What These Configs Approximate

The configs approximate common power-system state-estimation setups that use
voltage magnitudes, active/reactive bus injections, and active/reactive branch
flows in a weighted least-squares or robust-estimation study.

They are intended to resemble standard PSSE-style measurement categories often
used in AC state-estimation literature, but they do not reproduce a specific
paper's exact sensor-placement table.

## 2. Literature Alignment Intent

The intended alignment is measurement-type-level:

- voltage magnitude rows,
- active and reactive power injection rows,
- active and reactive branch-flow rows,
- diagonal weighting by configured standard deviations,
- optional sparse bad-data stress for robust-estimator comparison.

This is suitable for supplementary sensitivity or setup discussion, not as a
claim of exact reproduction of a published field measurement deployment.

## 3. Assumptions Supported by Code

The repository supports:

- IEEE/PYPOWER benchmark network loading,
- generated AC voltage, injection, and branch-flow equations,
- row-level standard deviations,
- implicit diagonal covariance `R_ii = sigma_i^2`,
- row weighting by division by `sigma_i`,
- random missing-row removal,
- sparse signed additive bad data,
- weak-area standard-deviation multipliers,
- classical estimators and QSVT-target spectral filtering.

## 4. Assumptions Still Approximate

The configs do not provide:

- field-calibrated PMU/SCADA measurement standard deviations,
- exact sensor placement from a utility or a specific paper,
- correlated measurement covariance,
- telemetry timing or communication effects,
- topology error modeling,
- real bad-data event distributions.

## 5. Placement Precision

Placement is measurement-type-level, not exact bus/branch-level reproduction of
a literature case. The current config keys enable or disable full measurement
families and weak-area stress sets. They do not yet express arbitrary per-row
sensor placement masks such as "measure only these five branches and these
three bus injections."

## 6. PMU/SCADA Classes

PMU and SCADA are conceptual labels in the current repository. The code
implements generated measurement equations and standard deviations, but it does
not implement separate PMU/SCADA device classes, sampling rates, channel
bundles, or field-data import.

## 7. Recommended Use

Use these configs as supplementary evidence or future-work scaffolding. The
main evidence should remain the already documented controlled benchmark and
PYPOWER experiment suites unless exact literature placement controls are added.

For journal wording, use:

```text
We include draft literature-inspired measurement setups using generated
PSSE-style measurement categories from IEEE/PYPOWER benchmark cases.
```

Avoid:

```text
We reproduce field-calibrated PMU/SCADA measurement datasets.
```
