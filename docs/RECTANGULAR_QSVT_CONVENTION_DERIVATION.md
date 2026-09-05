# Rectangular QSVT Convention Derivation (PyQSP sym_qsp → production PCPhase)

Convention status: **`formally_derived_and_independently_validated`** (odd
degree, real rectangular matrices). This document supersedes and extends
`outputs/generalized_rectangular_qsvt/rectangular_convention_derivation.md`
by replacing the signal-processing-polynomial recursion sketch with a closed
conjugation identity that holds for **arbitrary** phase vectors and is verified
numerically in `tests/test_convention_identity_and_global_phase.py`.

Implementation anchors:
`src/robust_qsvt_se/qsvt/rectangular_convention.py` (primitives),
`src/robust_qsvt_se/generalized/convention_api.py` (guarded API).

## 1. Objects and conventions

**Matrix and normalization.** Let `A ∈ R^{m×n}` (real; the complex case is
excluded, see §6) with `‖A‖₂ ≤ 1` after normalization by `β ≥ σ_max`, and thin
SVD `A = U Σ Vᵀ`, singular values `σ_i ∈ [0, 1]`.

**Padding.** `A` is zero-padded to `M ∈ R^{pad×pad}`, `pad = 2^⌈log₂ max(m,n)⌉`
in the production path; padding introduces zero singular values only.

**Julia dilation (signal unitary, reflection type).**

```
W_A = [ M                &  sqrt(I − M Mᵀ) ]
      [ sqrt(I − Mᵀ M)   &  −Mᵀ            ]        (2·pad × 2·pad, unitary)
```

Scalar reduction (`A = x ∈ [−1,1]`, `s = sqrt(1−x²)`):
`W_x = [[x, s], [s, −x]] = x Z + s X` — a **reflection** (`W_x² = I`, det −1).

**PCPhase (projector-controlled phase).** With `Z_pad = diag(I_pad, −I_pad)`
(top encoded subspace = first `pad` indices):

```
P(φ) = exp(i φ Z_pad) = diag(e^{iφ} I_pad, e^{−iφ} I_pad);   scalar: R(φ) = e^{iφZ}.
```

**Production sequence and ordering** (implemented in `pcphase_qsvt_operator`).
For positive odd degree `d`, with the rightmost factor acting first,

```
Π_prod(φ) = P(φ_d) W_A P(φ_{d-1}) W_A^† P(φ_{d-2}) …
            … P(φ_2) W_A^† P(φ_1) W_A P(φ_0).
```

Thus the signal factor immediately to the left of `P(φ_0)` is `W_A`, and the
factors alternate `W_A,W_A^†,…,W_A` (d factors) between d+1 phases. Counting
convention: `N_U = d`, `N_φ = d+1`, alternating length `L_alt = 2d+1`.

**PyQSP sym_qsp convention (plus-i signal, rotation type).**

```
Q(x) = [[x, i s], [i s, x]] = x I + i s X       (rotation: det = 1),
Π_pyqsp^fwd(φ; x) = R(φ_0) Q(x) R(φ_1) … Q(x) R(φ_d),
Im ⟨0| Π_pyqsp^fwd |0⟩ = P(x)
    for the calibrated pure-odd bounded target P.
```

(`sym_qsp_circuit_action.py`: `SIGNAL_CONVENTION="plus_i"`,
`RESPONSE_COMPONENT="imag"`, `PRODUCT_ORDER="existing_order"`.)  Every scalar
factor is symmetric, so
`Π_pyqsp^rev := R(φ_d)Q…Q R(φ_0) = (Π_pyqsp^fwd)^T`; consequently the two
orders have the same top-left entry.  The reverse form is the one that aligns
term-by-term with the production operator above.

## 2. Two exact conjugation identities

**(I1) Phase shift extracts iZ.** `R(φ + π/2) = e^{i(π/2)Z} R(φ) = (iZ) R(φ)`,
since `e^{i(π/2)Z} = diag(i, −i) = iZ`.

**(I2) Reflection ↔ rotation conjugation.** With `Y = ZX/i` and
`e^{−iπ/4·Z} X e^{iπ/4·Z} = Y`:

```
Z W_x = x I + s ZX = x I + i s Y = e^{−iπ/4·Z} (x I + i s X) e^{iπ/4·Z}
      = e^{−iπ/4·Z} · Q(x) · e^{iπ/4·Z}.
```

Block form (verified to float64 precision): with
`Q_A = [[M, i·sqrt(I−MMᵀ)], [i·sqrt(I−MᵀM), Mᵀ]]`,

```
Z_pad · W_A = e^{−iπ/4·Z_pad} · Q_A · e^{+iπ/4·Z_pad},
```

because conjugation by `e^{iπ/4·Z_pad}` multiplies the top-right block by
`e^{−iπ/2} = −i` and the bottom-left block by `+i`, and `Z_pad` flips the sign
of the bottom row of blocks. The analogous identity for `W_A^†` defines
`Q_A^‡ = e^{iπ/4·Z_pad} Z_pad W_A^† e^{−iπ/4·Z_pad}`, which reduces to `Q(x)`
in every scalar (cosine–sine) block for real `A`.

## 3. Proposition (convention transfer)

**Assumptions.**
(A1) `d` odd and positive; the target `P` is pure-odd with `|P| ≤ 1` on
[−1,1].
(A2) `{φ_k}_{k=0}^{d}` are calibrated PyQSP `sym_qsp` plus-i phases for `P`.
(A3) `A` real with `‖A‖₂ ≤ 1` (Julia dilation unitary).

**Claim.** Define `φ_k^prod = φ_k + π/2` for every k. Then

```
(i)  Π_prod(φ^prod; x) = i^{d+1} · e^{−iπ/4·Z}
                         Π_pyqsp^rev(φ; x) e^{+iπ/4·Z} · Z
     for ARBITRARY real phases (not only calibrated ones), hence
     ⟨0|Π_prod|0⟩ = i^{d+1} ⟨0|Π_pyqsp^fwd|0⟩ and, for calibrated phases,
     sign(d) · Im ⟨0|Π_prod|0⟩ = P(x)  with  sign(d) = (−1)^{(d+1)/2}.

(ii) For real rectangular A, the signed-imaginary top-left block of
     Π_prod(φ^prod), restricted to the original [0,m)×[0,n) indices, equals
     U P(Σ) Vᵀ.

(iii) Padding zero singular values map to zero (P odd ⇒ P(0)=0), so the
     recovered m×n block is uncontaminated.
```

**Proof.**

*Step 1 (scalar factorization).* Substitute (I1) into every phase of
`Π_prod(φ^prod)`: each factor `R(φ_k + π/2)` contributes `iZ`, giving a global
factor `i^{d+1}` and interleaved `Z`s:

```
Π_prod(φ^prod) = i^{d+1} · R(φ_d) Z W_x R(φ_{d−1}) Z W_x … R(φ_0) Z .
```

(`R(φ)` commutes with `Z`; in the scalar case `W_x = W_x^†`, so the
alternation is immaterial here.) Group each `Z W_x` and apply (I2):

```
Π_prod(φ^prod) = i^{d+1} e^{−iπ/4·Z}
                 [R(φ_d) Q(x) … R(φ_1) Q(x) R(φ_0)]
                 e^{+iπ/4·Z} Z ,
```

where the trailing `Z` is the unpaired phase-side factor and the `e^{±iπ/4·Z}`
from adjacent groups cancel internally (both commute with every `R`). The
bracket is `Π_pyqsp^rev`; its top-left entry equals that of the implemented
forward PyQSP product by the transpose observation in Section 1. This is
identity (i); it holds for arbitrary phases and is machine-verified for
d ∈ {1,3,5,7,9,11}, three random phase vectors per degree, and five signal
points per case (`test_pi_over_2_offset_identity_for_arbitrary_phases`,
max error < 1e−12).

*Step 2 (top-left entry and sign).* `e^{±iπ/4·Z}|0⟩ = e^{±iπ/4}|0⟩` and
`Z|0⟩ = |0⟩`, so the boundary phases cancel:
`⟨0|Π_prod|0⟩ = i^{d+1} ⟨0|Π_pyqsp|0⟩`. For odd `d`, `i^{d+1} = (−1)^{(d+1)/2}`
is real, so with `⟨0|Π_pyqsp|0⟩ = ρ + iP(x)` (calibrated convention),

```
Im ⟨0|Π_prod|0⟩ = (−1)^{(d+1)/2} P(x)  ⇒  sign(d)·Im recovers P(x).
```

`d ≡ 1 (mod 4) ⇒ sign = −1` (neg_imag); `d ≡ 3 (mod 4) ⇒ sign = +1` (imag) —
exactly `predict_extraction(degree)` in the API
(`test_sign_rule_matches_i_power`).

*Step 3 (rectangular lift and the two singular spaces).* Let the padded real
matrix have a full SVD `M=UΣVᵀ`, where `U,V∈R^{pad×pad}` are orthogonal,
`Σ=diag(σ_i)`, and `S=sqrt(I−Σ²)`. Functional calculus gives
`sqrt(I−MMᵀ)=USUᵀ` and `sqrt(I−MᵀM)=VSVᵀ`. Define

```
L = diag(U,V),        R_b = diag(V,U),
J_Σ = [[Σ,S],[S,−Σ]].
```

Then the exact rectangular left/right factorization is

```
W_A   = L J_Σ R_bᵀ,             W_A^† = R_b J_Σ Lᵀ.
```

This is why `W_A` itself need not be Hermitian when `M` is rectangular or
nonnormal; only each scalar block `J_{σ_i}` is a Hermitian reflection.  Every
projector phase commutes with both basis changes because
`P(φ)=diag(e^{iφ}I,e^{−iφ}I)`.  Substitution into the odd-degree alternating
product makes adjacent `R_bᵀR_b` and `LᵀL` factors cancel, yielding

```
Π_prod(φ^prod;A) = L [direct sum_i Π_prod(φ^prod;σ_i)] R_bᵀ.
```

Consequently its encoded top block is
`U diag(f(σ_i)) Vᵀ`, where `f` is the scalar top-left response from Steps 1--2.
For calibrated phases, `sign(d) Im f(σ_i)=P(σ_i)`. Because `U,V` are real,
elementwise imaginary-part extraction commutes with the outer factors, so

```
sign(d) Im([Π_prod]top) = U P(Σ) Vᵀ.
```

The same factorization covers tall and wide matrices after zero padding and
keeps the rectangular left (`U`) and right (`V`) singular spaces distinct. On
padding modes `P(0)=0` because `P` is odd, proving (ii)--(iii). ∎

## 4. Why the offset must be global (not endpoint-only)

Identity (i) consumes one `iZ` per phase; shifting only a subset of phases
leaves unpaired `Z` factors between signal operators, which changes the
realized polynomial. The recorded check
`identity_global_offset_required_max_err_when_first_phase_unshifted` is O(1)
in `rectangular_convention_symbolic_checks.csv`.

## 5. Global-phase handling

- The `i^{d+1}` factor is the only scalar global factor introduced by the
  conversion identity; the remaining quarter-turn conjugations and trailing
  `Z` cancel on the extracted top-left scalar entry. It is real (±1) for odd
  `d` and is absorbed into the extraction sign.
- Phase vectors are 2π-periodic: `φ_k → φ_k + 2π` leaves the operator
  unchanged (`test_phase_two_pi_periodicity`).
- A global phase on the prepared residual state passes through the (linear)
  sequence and cancels in every measured probability
  (`test_residual_state_global_phase_equivariance`).
- The full operator's overall phase is fixed by the sequence itself; no
  additional phase convention is applied at extraction.

## 6. Verified scope and exclusions

| Aspect | Status |
|---|---|
| odd degree, real rectangular A | derived (this doc) + validated: 245/245 generalized held-out matrices (7 dims × 7 spectral families incl. repeated, clustered, near-zero, exact-zero singular values), 150 final-campaign held-out matrices, 9 distinct odd degrees {1,3,5,7,15,31,63,127,255}, strict live tall and wide cases, full IEEE-14 82×27 at d=255 (error 1.6e−14), IEEE-30 172×59 and IEEE-57 331×113 sweeps (~1e−14) |
| even degree | **unsupported**: the target is pure-odd here, sym_qsp is calibrated for odd parity, and Step 2's real `i^{d+1}` requires odd `d` (even `d` makes `i^{d+1}` imaginary, moving the target to the real channel with a different rule that is not derived or implemented). API refuses with `ConversionError`; probes d ∈ {0,2,4,8,16,32} recorded as rejected. |
| complex A | **unsupported**: Step 3 uses real `U, V` so that `Im` isolates `U P(Σ) Vᵀ`; for complex `A` the imaginary part mixes target and complementary channels. Empirically 240/245 complex probes fail with O(1) error (5 degenerate zero-reference rows). |
| other phase conventions / other targets | out of scope; only PyQSP `sym_qsp` plus-i → dense-Julia PCPhase is claimed. |

## 7. Machine verification map

| Statement | Check |
|---|---|
| (i) arbitrary-phase identity | `tests/test_convention_identity_and_global_phase.py::test_pi_over_2_offset_identity_for_arbitrary_phases` |
| sign rule = i^{d+1} | `…::test_sign_rule_matches_i_power`; `tests/test_rectangular_convention_derivation.py` |
| calibrated scalar response | `rectangular_convention_symbolic_checks.csv` (response equality ~1e−13 for exact monomial targets) |
| (ii) rectangular block | held-out sweeps (both campaigns) + IEEE-14/30/57 rows |
| global offset necessity | symbolic checks column `identity_global_offset_required…` |
| 2π-periodicity, state global phase | `…::test_phase_two_pi_periodicity`, `…::test_residual_state_global_phase_equivariance` |
| even/complex refusal | `tests/test_rectangular_convention_all_degrees.py`, `tests/test_rectangular_convention_complex.py`, API validation CSV (18/18) |
