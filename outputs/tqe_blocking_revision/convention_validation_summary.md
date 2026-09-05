# Convention Validation Summary (Phase 5)

Convention status: `formally_derived_and_independently_validated`.

Two independent campaigns, kept separate (never merged):

- generalized campaign: 245 held-out real rectangular
  matrices (7 dims x 7 spectral families, reserved seed range
  [770000,779999]), 9 distinct odd degrees {1,3,5,7,15,31,63,127,255}
  (11 odd rows), 6 even probes rejected by the API, 245 complex probes
  unsupported, 5 symbolic identity checks.
- final campaign: 150 held-out matrices, 8 degree rows,
  5 independent-evaluator rows.
- strict live boundary checks: 13 rows, including tall
  and wide matrices, d=1 mod 4 and d=3 mod 4, repeated/near-zero/zero
  singular values, random orthogonal singular vectors, residual-state
  global-phase invariance, and explicit even/complex/nonfinite/length
  rejection checks.

Scope of the validated rule: odd degree, real rectangular matrices,
PyQSP sym_qsp plus-i phases -> dense-Julia PCPhase, global +pi/2 phase
offset, signed-imaginary top-left extraction with sign (-1)^((d+1)/2).
Complex matrices and even degrees are explicitly unsupported.
The formal derivation is `docs/RECTANGULAR_QSVT_CONVENTION_DERIVATION.md`.
