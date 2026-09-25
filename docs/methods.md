# ABP Methods Reference

Version 0.1.0 · 2026-09-25

This document states, for every implemented method, the model, its
assumptions, the equations as implemented, the matrix dimensions and data
ordering, the code location, and the tests that verify it. Conventions:
`n` records, `p` fixed-effect columns (after constraints), `q` levels of a
random term, `t` traits, all arithmetic in IEEE Float64. `'` denotes
transpose.

Contents: [1 Estimands](#1-estimands) · [2 Pedigree](#2-pedigree-relationships) ·
[3 Fixed effects](#3-fixed-effects-and-estimability) · [4 MME/BLUP/PEV](#4-mixed-model-equations-blup-and-pev) ·
[5 REML](#5-reml) · [6 Genomic](#6-genomic-relationships-and-gblup) ·
[7 Single step](#7-single-step) · [8 Multi-trait](#8-multi-trait-blup) ·
[9 Indices](#9-selection-indices) · [10 Numerics](#10-numerical-safeguards) ·
[11 Errata](#11-source-errata-handled)

---

## 1. Estimands

ABP 0.1 implements the estimand `additive_ebv`. This is the conditional
expectation of the additive genetic value `u` given the data, under the
stated model and (co)variances, relative to the declared genetic base. It is
distinct from phenotype prediction (which adds fixed and environmental
parts), total genetic value (which adds dominance and epistasis) and mating
utility (which refers to offspring distributions). The other three task names
are reserved and rejected (`abp.core.spec.IMPLEMENTED_TASKS`).

## 2. Pedigree relationships

**Code:** `src/abp/core/pedigree.py`, `src/abp/_native.cpp` ·
**Tests:** `tests/test_pedigree.py`

*Assumptions.* Diploid autosomal additive inheritance. Every unknown parent
is a distinct, unrelated, non-inbred base animal. Different unknown parents
are never merged into one common ancestor. Unknown-parent groups and
metafounders are not implemented.

*Ordering.* Kahn's algorithm with ties broken by input position. It yields
an order with parents before offspring, and it is deterministic. A remaining
unplaced set means a cycle (`ABP-E200`).

*Recursive definition* (tabular method; used only as the independent test
reference):

```
A_ij = (A_{s_i j} + A_{d_i j}) / 2   (j < i),      A_ii = 1 + F_i,   F_i = A_{s_i d_i} / 2
```

*Decomposition used in production* (Henderson 1976; Quaas 1976):

```
A = T D T',   T = (I - P)^{-1},   P[i, s_i] = P[i, d_i] = 1/2 (known parents)
d_i = 1/2 - (F_s + F_d)/4      both parents known
d_i = 3/4 - F_p/4              one parent p known
d_i = 1                        no parent known
A^{-1} = (I - P)' D^{-1} (I - P)          (sparse; at most 9 contributions per animal)
log|A| = sum_i log d_i                     (|T| = 1)
```

*Inbreeding* (Meuwissen & Luo 1992): for animal `i` with both parents known,
row `i` of `T` is accumulated by visiting ancestors in decreasing order.
Then `F_i = sum_j T_ij^2 d_j − 1`. An animal with an unknown parent has
`F_i = 0`. Full sibs reuse the family value. The C++ kernel (ADR 0002)
implements the same recursion and is checked against the Python reference to
1e-13.

*Products without A* (Colleau 2002): `A x = T (D (T' x))` is computed with
two sparse unit-triangular solves of `L = I − P` (SciPy `spsolve_triangular`).
`A22` for a subset is obtained column by column this way; it is never
obtained by inverting `A^{-1}`.

*Verification.* Spec gold standard T03 (F5 = 0.25; row 5 of A; exact A⁻¹).
Agreement with the tabular method on random inbred pedigrees (A, F, A⁻¹A = I,
log|A|). Classic relationships (half sibs 1/4, full sibs 1/2, offspring of a
half-sib mating F = 1/8). Unknown parents as distinct base animals.
Invariance to input order. Rejection of cycles, self-parenthood and the same
animal as sire and dam.

## 3. Fixed effects and estimability

**Code:** `src/abp/core/design.py` · **Tests:** `tests/test_blup.py`

Candidate columns are scanned in the fixed order: intercept, then terms in
spec order, then factor levels sorted as strings. With `X'X` of all candidate
columns, a left-looking Cholesky in natural order keeps column `j` iff

```
x_j'x_j − l_j'l_j > rank_tol · x_j'x_j,        rank_tol = 1e-9,
```

i.e. iff the squared sine of the angle between `x_j` and the span of the kept
columns exceeds `1e-9`. The test is scale-free. Dropped columns are fixed at
zero ("set-to-zero constraints") and reported. `X b`, differences between
connected levels, and all random-effect predictions and PEVs are invariant to
this choice (tested by reordering terms). Individual level solutions are not
estimable and the report says so.

## 4. Mixed-model equations, BLUP and PEV

**Code:** `src/abp/solvers/mme.py`, `src/abp/solvers/blup.py` ·
**Tests:** `tests/test_blup.py`, `tests/test_workflow.py`

Model: `y = Xb + Σ_k Z_k u_k + e`, `u_k ~ N(0, Σ_k)`, `e ~ N(0, R)`, with
`u` and `e` uncorrelated. With `W = [X, Z_1, …, Z_K]`:

```
C s = r,     C = W' R^{-1} W + blockdiag(0_p, Σ_1^{-1}, …, Σ_K^{-1}),     r = W' R^{-1} y
```

`C` is **not** multiplied by the residual variance. Consequently

```
Var(u − û) = [C^{-1}]_uu      (includes the uncertainty of the GLS estimate of b)
```

conditional on the variance parameters (Henderson 1975). For a single trait
`Σ_k = σ_k² K_k` and `R = σ_e² I`. Per animal `i` of a genetic term:
`PEV_i = [C^{-1}]_ii`, `SEP_i = sqrt(PEV_i)`,
`reliability_i = 1 − PEV_i / (σ_a² K_ii)` (`K_ii = 1 + F_i` for A), and
`accuracy_i = sqrt(reliability_i)`.

*Solvers.*

* `dense`: `C = L L'` (LAPACK `potrf`). For PEV, `diag(C^{-1})` comes from
  the column norms of `L^{-1}` (`trtri`). The full inverse (`potri`) is
  formed only when REML needs traces.
* `sparse_direct`: SuperLU with `MMD_AT_PLUS_A` ordering, diagonal pivoting
  disabled and symmetric mode. This is valid because `C` is SPD, and all
  pivots are checked to be positive. PEV comes from solves against unit
  vectors in batches.
* `pcg`: conjugate gradients preconditioned with `diag(C)`. The recurrence
  residual is re-verified against the true residual `b − C x`, with restarts.

After every solve the relative residual `‖C s − r‖ / ‖r‖` is recomputed on the
assembled system. Solutions above 1e-8 (direct) or above `tol` (PCG) are
rejected (`ABP-E401`).

*Reliability range.* Values outside `[−1e−8, 1 + 1e−8]` raise `ABP-E402`.
Values inside that round-off band are clamped for display, and the number of
clamped values is recorded.

*Verification.* Spec T04: intercept 3, EBVs ∓0.5, PEV 0.75, reliability
0.25. Mrode (2005) Example 3.1: independent V-form to 1e-9 and the printed
values to 6e-4. Rank-deficient repeatability model against the V-form with a
generalized inverse (`pinv`). Agreement of the three solvers. kg→g scaling
(EBV ×1000, PEV ×1e6, reliability unchanged). Invariance to constraints and
record order.

## 5. REML

**Code:** `src/abp/solvers/reml.py` · **Tests:** `tests/test_reml.py`

Parameters `θ = (θ_1 … θ_K, θ_0)`, with `Var(u_k) = θ_k K_k` and
`Var(e) = θ_0 I`. With `C` as in §4:

```
−2 log L = n log θ_0 + Σ_k (q_k log θ_k + log|K_k|) + log|C| + y'Py,
y'Py     = y'y/θ_0 − s' W'y/θ_0.
```

This is identical to `log|V| + log|X'V^{-1}X| + y'Py` (the constant
`(n − r_X) log 2π` is dropped). The identity
`|V||X'V^{-1}X| = |R||G||C|` is verified numerically in the tests against the
V-form.

*Scores* (derivatives of log L). Here `C^{kk}` is the `u_k` block of `C^{-1}`:

```
∂/∂θ_k = ½ [ û_k' K_k^{-1} û_k / θ_k² − q_k/θ_k + tr(K_k^{-1} C^{kk}) / θ_k² ]
∂/∂θ_0 = ½ [ ê'ê/θ_0² − tr(P) ],
tr(P)  = [ n − r_X − Σ_k (q_k − tr(K_k^{-1} C^{kk})/θ_k) ] / θ_0.
```

These follow from `∂V/∂θ_k = Z_k K_k Z_k'` (for the animal term this is
`Z A Z'`, **not** `Z Z'`), `Z_k'Py = K_k^{-1} û_k / θ_k`,
`Z'PZ = G^{-1} − G^{-1} C^{ZZ} G^{-1}` and `tr(PV) = n − r_X`.

*Average information* (Gilmour et al. 1995). With working variates
`f_k = Z_k û_k / θ_k` and `f_0 = ê/θ_0`, collected in `F = [f_1 … f_K f_0]`:

```
AI = ½ F' P F,     P F = (F − W S)/θ_0,     C S = W'F / θ_0.
```

*EM updates* (REML form):

```
θ_k ← (û_k'K_k^{-1}û_k + tr(K_k^{-1} C^{kk})) / q_k
θ_0 ← (ê'ê + tr(C^{-1} W'W)) / n
```

*Algorithm.* Take the AI step `Δ = AI^{-1} ∇`, halving it up to 12 times so
that all components stay above `10^{-6} Σθ` and log L does not decrease;
otherwise take an EM step. Convergence requires
`‖Δθ‖/‖θ‖ < tol` **and** the Newton decrement `∇' AI^{-1} ∇ < tol`.

*Boundary (active set).* A random-term component becomes a boundary
candidate if the full AI step would take it below the floor while its score
is negative, or if it stays within 10× the floor for 3 iterations with a
negative score. The candidate is fixed at 0 and the sub-model re-fitted. The
zero is then accepted only if the Kuhn-Tucker condition holds, i.e. the score
at zero is non-positive:

```
∂ log L/∂θ_k |_{θ_k=0} = ½ [ (Z_k'Py)' K_k (Z_k'Py) − tr(Z_k'PZ_k K_k) ] ≤ 0,
```

evaluated with the sub-model's `P`. Otherwise the term is reinstated and
excluded from rule (a). Components on the boundary are reported as 0 with
status `converged_boundary`, and their standard errors are withheld. A
residual variance of zero is never accepted.

*Standard errors.* `sqrt(diag(AI^{-1}))` at an interior optimum. Heritability
`h² = θ_a / Σθ` has a delta-method SE with gradient
`∂h²/∂θ_j = (δ_{ja} Σθ − θ_a)/(Σθ)²`.

*Verification.* Spec T05: genetic score −0.01463020355 matches the V-form
and a finite difference to 1e-8. The wrong-derivative counterexample
(`Z Z'` instead of `Z A Z'`) gives +1.03497570401 and is detected.
Log-likelihood, score and AI match the V-form for a three-component model.
The optimum matches independent Nelder-Mead on the V-form likelihood (rtol
2e-5). EM and AI agree. An engineered boundary case gives exactly
`σ_e² = var(y)`. Non-convergence is an error. Checkpoint resume reproduces a
fresh fit. In a 40-replicate simulation the mean estimates lie within 3
Monte-Carlo SE of the truth.

## 6. Genomic relationships and GBLUP

**Code:** `src/abp/core/genomic.py`, `src/abp/qc/genotype.py`,
`src/abp/workflows/genomic_inputs.py` · **Tests:** `tests/test_genomic.py`,
`tests/test_genomic_workflow.py`

`M` (`n × m`) holds dosages of the declared counted allele. `p_j` is the
counted-allele frequency in the declared reference sample; missing dosages
are excluded from `p`. The code computes (VanRaden 2008, method 1):

```
W_ij = M_ij − 2 p_j   (missing: W_ij = 0, i.e. imputed at the reference mean)
d    = 2 Σ_j p_j (1 − p_j)      (d = 0 → ABP-E221)
G    = W W' / d
MAF_j = min(p_j, 1 − p_j)       (minimum, not maximum)
```

*Policies* (never implicit; recorded with the smallest eigenvalue before and
after):

```
match_a22 : G* = a + b G with mean diag(G*) = mean diag(A22), mean offdiag(G*) = mean offdiag(A22)
blend     : G* = (1 − α) G* + α A22
ridge     : G* = G* + λ I
error     : stop unless G* is positive definite (ABP-E302)
```

*GBLUP* uses `K = G*` and `K^{-1}` from the Cholesky factor
(`log|G*|` is available for REML).

*SNP-BLUP equivalence.* With the same `W` and
`β ~ N(0, I σ_a²/d)`, `W β̂` equals the GBLUP of every genotyped animal,
including animals without phenotypes. After a ridge, the equivalence holds
with an additional iid polygenic term of variance `λ σ_a²`. Both identities
are tested to 1e-8. Marker effects are only identifiable as the linear
combinations `W β`.

*Allele coding.* Flipping the counted allele of any marker
(`M → 2 − M`, `p → 1 − p`) leaves `G` unchanged (tested). Strand-ambiguous
A/T and C/G markers are flagged because their orientation cannot be checked.

## 7. Single step

**Code:** `abp.core.genomic.single_step` · **Tests:**
`tests/test_genomic.py::test_t06_single_step_identities`,
`tests/test_genomic_workflow.py::test_single_step_workflow_matches_explicit_h`

With genotyped animals indexed `2` and the others `1`:

```
H^{-1} = A^{-1} + [[0, 0], [0, G*^{-1} − A22^{-1}]]
diag(H)_i = A_ii + b_i'(G* − A22) b_i,      b_i = A22^{-1} A_{2,i}     (= G*_ii for genotyped i)
log|H| = log|A| + log|G*| − log|A22|
```

`A22` comes from Colleau products and `A22^{-1}` from its own Cholesky
factor. The 22-block of `A^{-1}` is a different matrix; the test suite
includes this counterexample. When `G* = A22`, `H^{-1} = A^{-1}` exactly
(spec T06). For a non-trivial `G*`, `H^{-1}` equals the inverse of the
explicitly assembled `H` to 1e-12. The workflow result matches a V-form BLUP
with that explicit `H` to 1e-8.

## 8. Multi-trait BLUP

**Code:** `src/abp/solvers/multitrait.py` · **Tests:** `tests/test_multitrait.py`

Records `r` carry observed trait subsets `o_r`:

```
Var(vec_animal-major(U)) = K ⊗ G0,     Var(e_r) = R0[o_r, o_r],     records independent
```

*Stacking.* Random equation `offset + i·t + j` holds animal `i`, trait `j`,
so the precision block is `K^{-1} ⊗ G0^{-1}` (sparse Kronecker product).
Observations are stacked record-major. The residual precision is
block-diagonal with one `(R0[o_r, o_r])^{-1}` per record; inverses are cached
by missing pattern. Fixed effects are block-diagonal by trait, each block
full rank (§3). PEV blocks per animal are the `t × t` diagonal blocks of
`C^{-1}`: `(L^{-1}[:, block])' L^{-1}[:, block]` on the dense path, or batched
unit solves on the sparse path. Reliability is
`1 − PEV_jj/(G0_jj K_ii)`.

*Verification.* A 2-animal × 2-trait case against an independent trait-major
V-form (`G0 ⊗ A`): EBVs and PEV blocks to 1e-12, with information
propagating to the unobserved trait. A three-trait case with missing
patterns, trait-specific fixed effects and a structural residual zero
against the V-form, on both dense and sparse paths. With zero covariances
the model decomposes into the single-trait analyses. With one trait it
equals the single-trait model. Changing one trait's unit scales its EBVs
and leaves the reliabilities unchanged.

## 9. Selection indices

**Code:** `src/abp/decision/selection_index.py` · **Tests:** `tests/test_selection_index.py`

*Smith-Hazel.* With objective `H = a'g`, information `x`, `P = Var(x)` and
`C = Cov(x, g')`:

```
q = C a,   P b = q,   Var(I) = b'Pb,   Var(H) = a' G_H a,
r_IH² = Var(I)/Var(H),   r_IH = sqrt(Var(I)/Var(H))
```

Accuracy is the square root of a variance ratio, never a product of
variances. Expected responses to one round of truncation selection with
intensity `i = φ(z)/p_sel` are `R_H = i σ_I` and `R_j = i (C'b)_j/σ_I`.

*Restricted index* (Kempthorne & Nordskog 1959). The condition `C_r'b = 0`
gives

```
b_r = b − P^{-1}C_r (C_r'P^{-1}C_r)^{-1} C_r' b,       r² = (b_r'q)² / (Var(I) Var(H))
```

Infeasible or redundant restrictions are reported. The solution is checked
against the KKT system `[P C_r; C_r' 0][b; λ] = [q; 0]` solved
independently. A singular `P` (duplicated information) is reported together
with the zero-variance combination of sources.

*EBV index.* With multi-trait EBVs, `I_i = a'û_i` and
`rel(I_i) = 1 − a' PEV_i a / (a' G0 a · K_ii)`.

*Verification.* Spec T02: `b = [3/7, 2/7]`, reliability 12/35, accuracy
0.585540043769, invariance to scaling of the objective. Invariance of the
information units: `b` scales inversely and `r` is unchanged.

## 10. Numerical safeguards

* Float64 everywhere. The default acceptance tolerance for well-conditioned
  analytical cases is `|x − ref| ≤ 1e−10 + 1e−8 |ref|` (spec proposal).
* Dense memory is estimated before allocation (`16 N²` or `24 N²` bytes) and
  compared with `resources.max_memory_gb`.
* Exact PEV on the sparse path is limited to 30,000 equations. Beyond that,
  `pev = "none"` is required, because approximate reliabilities are not yet
  implemented.
* Covariance matrices must be symmetric positive definite (Cholesky test).
  Variances must be finite and > 0; a zero variance means the term should be
  removed from the model.

## 11. Source errata handled

The project specification identified errors in the teaching sources. ABP
implements the corrected forms and tests them:

| Source (physical PDF page) | Issue | ABP |
|---|---|---|
| Legarra et al., *Bases for Genomic Prediction* (gsip.pdf p.15) | MAF text says minimum, code uses `maxval` | `minor_allele_frequency` uses the minimum; test T01 |
| gsip.pdf p.125 | correlation formula denominator reuses one variance | validation statistics (planned) will use the product of both SDs |
| Mrode 4th ed. PDF p.63 | shortcut for index accuracy multiplies instead of dividing | `r_IH = sqrt(Var(I)/Var(H))`; test T02 |
| Misztal course PDF p.146 | sign of the random part in the conditional mean of missing records | not used in 0.1 (no data augmentation); planned modules must follow the corrected form in the spec |
| Early R teaching code accompanying Mrode (uploaded ZIP) | REML genetic-variance derivative without A | score uses `Z A Z'`; test T05 detects the wrong form |
