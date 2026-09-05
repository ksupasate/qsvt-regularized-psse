# Claim Scope

## Supported statements

The repository supports evidence-backed statements about:

- numerical conditioning and regularization on controlled weighted systems;
- estimator behavior under configured synthetic noise, missing rows, bad data,
  and weak-area weighting;
- generated DC, AC-linearized, and nonlinear AC measurement systems derived
  from built-in or PYPOWER IEEE network models;
- equality of the exact classical Ridge/Tikhonov solution and the matched
  `qsvt_regularized` spectral target when inputs and `alpha` are identical;
- bounded polynomial and scalar phase-response accuracy within the configured
  domains;
- correctness or structured infeasibility of explicitly tested small QSVT
  constructions;
- transparent proxy resource calculations under their declared access,
  encoding, postselection, and readout assumptions.

## Unsupported statements

The repository does not establish:

- validation on real PMU measurements or SCADA streams;
- field-calibrated sensor placement, covariance, bad-data rates, or outage
  statistics;
- operational deployment readiness;
- quantum speedup, quantum advantage, or lower end-to-end cost than classical
  state estimation;
- full quantum-hardware execution of IEEE14/30/57/118/300 state estimation;
- that QSVT outperforms Ridge in classical simulation;
- that a small selected block preserves the full-system answer outside the
  measured support-error evidence;
- an exhaustive impossibility result for QSVT-based state estimation.

## Safe wording matrix

| Topic | Use | Avoid |
|---|---|---|
| Data | “generated measurements from IEEE/PYPOWER benchmark network models” | “real PMU/SCADA data” |
| Statistics | “controlled configured noise/covariance” | “field-calibrated sensor statistics” |
| QSVT role | “possible implementation pathway” or “feasibility study” | “quantum acceleration” |
| Classical target | “QSVT-compatible regularized spectral target” | “QSVT beats Ridge” |
| Circuits | “small selected-block simulator/circuit validation” | “full IEEE quantum execution” |
| Resources | “proxy estimate under declared assumptions” | “demonstrated hardware cost” |
| Negative result | “no jointly credible regime was identified in the tested scope” | “QSVT is impossible for PSSE” |

## Evidence ordering

Interpret evidence in this order:

1. Does a retained support or approximation preserve the full matched
   regularized computation?
2. Is that computation accurate against the generated benchmark reference?
3. Does the polynomial/circuit implement its declared filter?
4. Are access, state preparation, postselection, readout, and classical
   alternatives credibly accounted for?

A later implementation PASS cannot repair an earlier application-fidelity
failure. Conversely, a negative result for the registered settings is a scoped
boundary, not a universal theorem.

## Review checklist for new prose

Before adding a claim to the README, documentation, or release notes, verify that:

- the cited config and output are present in the experiment manifest;
- the stated metric exists in the underlying CSV/JSON artifact;
- failures and infeasible cases remain counted;
- “field,” “hardware,” “speedup,” and “advantage” are not implied without new
  evidence;
- scalar, selected-block, and full-matrix results are not conflated;
- exact target, polynomial, circuit, and finite-shot errors are named
  separately.
