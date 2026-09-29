# ABP Methods Reference

Version 0.4.0 · 2026-09-29

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
[11 Errata](#11-source-errata-handled) · [12 Genetic groups](#12-unknown-parent-groups) ·
[13 LR validation](#13-forward-in-time-validation-lr-method) ·
[14 OCS and mating](#14-optimal-contributions-and-mating-allocation) ·
[15 PLINK input](#15-plink-1-binary-input) ·
[16 Bayesian marker models](#16-bayesian-marker-models-and-mcmc-diagnostics) ·
[17 Metafounders](#17-metafounders) ·
[18 Selected inversion](#18-sparse-selected-inversion-exact-pev-at-scale) ·
[19 Multi-trait REML](#19-multi-trait-reml) ·
[20 Threshold model](#20-threshold-probit-model-for-categorical-traits) ·
[21 Sparse LDL'](#21-sparse-ldl-factorization) ·
[22 APY](#22-apy-inverse-of-g)

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
* Exact PEV on the sparse path uses selected inversion (§18). Its limit is
  the memory of the symbolic factor (24 bytes per entry, checked against
  `resources.max_memory_gb` before any numeric work), not a number of
  equations.
* Covariance matrices must be symmetric positive definite (Cholesky test).
  Variances must be finite and > 0; a zero variance means the term should be
  removed from the model.

## 11. Source errata handled

The project specification identified errors in the teaching sources. ABP
implements the corrected forms and tests them:

| Source (physical PDF page) | Issue | ABP |
|---|---|---|
| Legarra et al., *Bases for Genomic Prediction* (gsip.pdf p.15) | MAF text says minimum, code uses `maxval` | `minor_allele_frequency` uses the minimum; test T01 |
| gsip.pdf p.125 | correlation formula denominator reuses one variance | LR `ρ_wp` uses `sqrt(Var(u_w)·Var(u_p))`; `test_lr_statistics_definitions` |
| Mrode 4th ed. PDF p.63 | shortcut for index accuracy multiplies instead of dividing | `r_IH = sqrt(Var(I)/Var(H))`; test T02 |
| Misztal course PDF p.146 | sign of the random part in the conditional mean of missing records | not used: no implemented model augments missing records (Bayesian marker models use observed records only); future modules must follow the corrected form in the spec |
| Early R teaching code accompanying Mrode (uploaded ZIP) | REML genetic-variance derivative without A | score uses `Z A Z'`; test T05 detects the wrong form |

## 12. Unknown-parent groups

Code: `abp/core/upg.py` (`ainv_with_groups`, `group_fractions`,
`upg_structure_parts`, `check_fixed_groups_estimable`, `upg_structure`);
pedigree codes in `abp/qc/pedigree.py`. Registry id `ped.upg`.

**Model.** Unknown parents may be assigned to groups `k = 1..g`. With `Q`
(`n × g`) the expected gene fractions, `Qᵢ = (Q_s + Q_d)/2` where a group
parent contributes its unit vector `e_k` and a plain unknown parent
contributes 0, the breeding value is

    u*ᵢ = uᵢ + Qᵢ g,     u ~ N(0, σ²a A),

with `A` built as if every unknown parent (grouped or not) were an
unrelated, non-inbred base animal. For `θ = (u*, g)` the mixed-model
equations use (Quaas 1988; Westell et al. 1988)

    A*⁻¹ = [[ A⁻¹,       −A⁻¹Q  ],
            [ −Q'A⁻¹,   Q'A⁻¹Q ]],

which equals Henderson's rules for `A⁻¹` applied with groups acting as
parents: for animal `i` with `δᵢ = 1/dᵢ` (`dᵢ` computed from the *animal*
parents only, including their inbreeding) the contributions
`δᵢ·[1, −½, −½] ⊗ [1, −½, −½]` are added to rows/columns `(i, s, d)`, where
`s` or `d` is the group column `n + k` when that parent is group `k`.

* **Random groups:** `g ~ N(0, σ²g I)` with a declared ratio
  `r = σ²g/σ²a`. Then `K⁻¹ = A*⁻¹ + blockdiag(0, I/r)` is the inverse of
  `K = Var(θ)/σ²a = [[A + rQQ', rQ], [rQ', rI]]`, `diag(K) = (1 + Fᵢ + r‖Qᵢ‖², r)`
  (used for reliabilities) and `log|K| = log|A| + g·log r` (used by REML,
  which therefore estimates σ²a and σ²e with `r` fixed).
* **Fixed groups:** `K⁻¹ = A*⁻¹` is singular (improper prior). The system for
  `(b, u*, g)` is a non-singular linear transformation of the MME of
  `y = Xb + ZQg + Zu + e` with fixed `(b, g)`, so it has a unique solution
  exactly when `[X, ZQ]` has full column rank. `check_fixed_groups_estimable`
  scales each column of `[X, ZQ]` to unit norm and requires every singular
  value to exceed `10⁻⁹ ×` the largest; groups whose `ZQ` column is zero
  (no recorded descendants) are reported by name. `Var(u*ᵢ)` is undefined,
  so `diag(K)` is `NaN` and no reliabilities are produced; `PEV(u*)`
  includes the estimation error of `g`. REML is refused.

The workflow appends the groups after the animals in equation order, writes
animal solutions (`u*`) to the EBV file and group solutions to
`upg_solutions_<trait>.csv`.

**Tests.** `test_qp_identity`: `A*⁻¹` from the Henderson construction equals
the explicit block formula with `A⁻¹` from the tabular method, and `Q` equals
an independent recursion. `test_random_groups_blup_equals_explicit_model`:
solutions and PEV of `(u*, g)` equal the V-form BLUP with the explicit
covariance `K σ²a` (1e−9), and `log|K|` equals the log-determinant of that
covariance. `test_fixed_groups_blup_equals_explicit_model`: solutions and
`PEV = diag(T C⁻¹ T')` equal the explicit MME in `(b, g, u)` transformed by
`T: (b, g, u) → (u + Qg, g)` (1e−10). `test_fixed_groups_confounded_*`:
confounding with the intercept and empty groups stop with `ABP-E300`
(unit and workflow level). Simulation evidence: `benchmarks/upg_study.py`
(§3 of the validation report).

## 13. Forward-in-time validation (LR method)

Code: `abp/workflows/validation_lr.py`. Registry id `val.lr`.

Records are split by their date against `validation.cutoff`: a record is
*hidden* if its whole date range (e.g. all of `2025` for a `YYYY` date) lies
after the cutoff; a date range that straddles the cutoff is an error, never
guessed. The **partial** evaluation is built from a copy of the record set
in which the hidden trait values are set to missing before the model is
constructed; the **whole** evaluation uses all records. Both use the same
variance components: the spec values (`known`) or REML on the partial
records only (`reml`). A genomic or single-step structure whose allele
frequencies come from the phenotyped animals
(`frequency_source = "training_genotyped"`) is rebuilt from the partial
records; otherwise both evaluations share the main structure.

Focal animals: `new_records_only` = animals whose records of the trait are
all hidden; `born_after_cutoff` = animals born after the cutoff. On the
focal EBVs `u_p` (partial) and `u_w` (whole):

    Δp     = mean(u_p) − mean(u_w)
    b_w|p  = Cov(u_w, u_p) / Var(u_p)
    ρ_wp   = Cov(u_w, u_p) / sqrt(Var(u_w) · Var(u_p))

(Legarra and Reverter 2018). Under a correct model with the same
parameters, `E[Δp] = 0`, `E[b_w|p] = 1` and `ρ_wp` estimates
`acc_p/acc_w`. Uncertainty: `B` bootstrap replicates resampling clusters
(sire families; animals with unknown sire form their own cluster) with a
recorded PCG64 seed; standard errors and 95% percentile intervals.
Replicates in which the statistics are undefined (fewer than 3 focal
animals, or zero variance of `u_p` or `u_w`) are counted as failed and
reported. The
LR population-accuracy estimator is not computed (see the registry entry:
its published form has a 2019 erratum that could not be verified here).

**Tests.** `test_lr_statistics_definitions` (hand-computed values),
`test_split_by_cutoff_rules`, `test_cluster_bootstrap_is_seeded_and_counts_clusters`,
`test_hidden_phenotypes_cannot_leak_into_partial_evaluation` (adding 25 kg
to every hidden record leaves the partial EBVs bit-identical while the whole
EBVs change),
`test_partial_reml_ignores_hidden_records`, `test_validation_spec_rules`.

## 14. Optimal contributions and mating allocation

Code: `abp/decision/ocs.py`, `abp/decision/mating.py`,
`abp/workflows/mating_plan.py`. Registry ids `dec.ocs`, `dec.mating`.

**Contributions** (Meuwissen 1997). For candidates with merit `g`, sex and
pedigree relationships `A` (positive definite):

    maximise  c'g   subject to  Σ_{males} cᵢ = ½,  Σ_{females} cᵢ = ½,
                                0 ≤ cᵢ ≤ capacityᵢ / (2N),   c'Ac/2 ≤ C_max.

With `ΔF` declared, `C_max = C_t + ΔF(1 − C_t)`, `C_t` the mean coancestry
`1'A1/(2n²)` of the candidates. For a penalty `μ ≥ 0` the problem
`max c'g − μ c'Ac/2` over the linear constraints is a strictly convex QP,
solved exactly by a primal active-set method; its coancestry is
non-increasing in `μ`. `μ = 0` is the linear programme (greedy fill per
sex, the maximum-merit solution); the QP with objective `c'Ac/2` alone
gives the minimum achievable coancestry. If `C_max` lies between them, `μ`
is found by bisection so that the coancestry constraint is active. The KKT
certificate reports the stationarity residual
`‖μAc − g + K'η − ν_lower + ν_upper‖∞` (`K` the two sex-indicator rows,
`η` their multipliers, `ν ≥ 0` the bound multipliers), the sex-sum and bound
residuals, the coancestry slack, complementary slackness and the smallest
bound multiplier. `C_max` below the minimum is reported as infeasible with
that minimum.

**Integer plan.** Matings `nᵢ = 2N cᵢ` are rounded by largest remainder per
sex (totals `N`, capacities respected). If rounding violates `C_max`, a
local search moves single matings within a sex until the ceiling
holds, then improves merit by moves that keep it; the merit gap to the
continuous optimum is reported (it bounds the integer loss from above).
A move transfers one mating from candidate `i` to `j` of the same sex, with
exact changes `(g_j − g_i)/(2N)` in merit and
`((Ac)_j − (Ac)_i)/(2N) + (A_ii + A_jj − 2A_ij)/(8N²)` in coancestry. Repair
applies the move with the smallest merit loss per unit of coancestry
reduction while the ceiling is exceeded; improvement then applies the move
with the largest merit gain that keeps the ceiling. The result is feasible
and locally optimal for single moves, not a proven integer optimum.

**Pairs.** With integer counts `n_s`, `m_d`, choose `x_sd ∈ {0,1}` with
`Σ_d x_sd = n_s`, `Σ_s x_sd = m_d`, `x_sd = 0` on forbidden pairs, minimising
`Σ x_sd A_sd/2`. The expected progeny merit `Σ x_sd (g_s + g_d)/2` is the
same for every feasible plan. The constraint matrix is that of a bipartite
transportation problem (totally unimodular), so the LP relaxation (HiGHS
dual simplex) has integral optimal vertices; ABP verifies integrality and
every constraint of the returned plan. Forbidden pairs: `A_sd` above
`max_pair_relationship`; `p_s p_d / 4 > max_affected_risk` for declared
carrier probabilities of an autosomal recessive; explicit pairs. If no plan
exists, an elastic LP (slack on each parent's count) names the parents that
cannot be matched.

**Tests.** `test_ocs_matches_independent_optimizer_and_kkt` (SciPy
trust-constr reference and an independent KKT check), `test_closed_form_for_unrelated_candidates`,
`test_unconstrained_case_equals_linear_programme`, `test_infeasible_ceiling_reports_minimum`,
`test_tighter_ceiling_never_increases_merit`, `test_integer_repair_meets_ceiling_and_is_bounded_by_continuous_optimum`,
`test_mating_plan_is_optimal_by_enumeration` (all assignments of small
problems enumerated), `test_carrier_risk_forbids_only_risky_pairs`,
`test_infeasible_plan_names_unmatchable_dam`,
`test_example09_plan_satisfies_every_hard_constraint`.

## 15. PLINK 1 binary input

Code: `abp/io/plink.py`. Registry id `io.plink`.

`.bed` must start with `0x6c 0x1b 0x01` (SNP-major). Each variant occupies
`⌈n/4⌉` bytes; sample `i` is in byte `⌊i/4⌋`, bits `2(i mod 4)` and
`2(i mod 4)+1` (lowest-order first). Two-bit codes map to dosages of the
first `.bim` allele A1: `00 → 2`, `01 → missing`, `10 → 1`, `11 → 0`.
Decoding uses a 256-entry lookup table. The file size must equal
`3 + m⌈n/4⌉` bytes. The counted allele is A1; ref/alt are recorded as
unknown and the assembly must be declared. The loaded genotypes enter the
same QC and G construction as dosage files.

**Tests.** `test_hand_derived_bytes` (a byte pattern decoded by hand),
`test_round_trip_and_loader`, `test_bad_files` (magic number,
individual-major mode, wrong file size, identical alleles),
`test_plink_and_dosage_inputs_give_identical_gblup` (identical EBVs from
both formats).

## 16. Bayesian marker models and MCMC diagnostics

Code: `abp/solvers/bayes.py`, `abp/solvers/mcmc_diagnostics.py`,
`abp/workflows/bayes_eval.py`, native sweep `bayes_sweep` in
`abp/_native.cpp`. Registry ids `bayes.brr`, `bayes.a`, `bayes.b`,
`bayes.c`, `bayes.cpi`, `bayes.r`, `diag.mcmc`.

**Model.** `y = Xb + Wβ + e`, `W = M − 2p'` (missing dosages at their
expectation, i.e. `W = 0`), flat prior on `b`, `e ~ N(0, σ²e I)`,
`σ²e ~ Inv-χ²(ν_e, S²_e)`. Marker priors (π₀ = probability of a zero
effect):

| method | prior |
|---|---|
| BRR | `βⱼ ~ N(0, σ²b)`, `σ²b ~ Inv-χ²(ν, S²)` |
| BayesA | `βⱼ ~ N(0, vⱼ)`, `vⱼ ~ Inv-χ²(ν, S²)` |
| BayesB | `βⱼ = 0` w.p. π₀, else `N(0, vⱼ)`; `vⱼ` drawn from its prior when `βⱼ = 0` (Habier et al. 2011) |
| BayesC | `βⱼ = 0` w.p. π₀, else `N(0, σ²b)` |
| BayesCπ | as BayesC, `π₀ ~ Beta(1, 1)` |
| BayesR | component `k` w.p. `π_k`, `βⱼ ~ N(0, γ_k σ²b)`, `γ = (0, 10⁻⁴, 10⁻³, 10⁻²)`, `π ~ Dirichlet(1,…,1)` (Erbe et al. 2012; ABP scales `γ` on centred, not standardised, genotypes) |

**Default hyper-parameters.** `V_y` = residual variance of `y` after the
fixed effects; `V_g = r²·V_y`, `V_e = (1 − r²)·V_y` with `r² = prior_r2`.
The expected variance of an included marker is
`V_g / (Σⱼ 2pⱼ(1 − pⱼ) · E[inclusion] · E[γ])`, and each scale is set so
that the prior mean `ν S²/(ν − 2)` equals its target.

**Sampler.** Per iteration: `b | ·` jointly from its normal full
conditional; for each marker `j` in order (residual updating, `e = y − Xb − Wβ`):
`rⱼ = (wⱼ'e + wⱼ'wⱼ βⱼ)/σ²e`; for a slab of variance `v`,
`Cⱼ = wⱼ'wⱼ/σ²e + 1/v`, log-weight
`log π_k − ½(log Cⱼ + log v) + ½ rⱼ²/Cⱼ` against `log π₀` for the zero
component (normalised by log-sum-exp); if included,
`βⱼ ~ N(rⱼ/Cⱼ, 1/Cⱼ)`; then the variance parameters, mixture weights and
σ²e from their conjugate full conditionals. The normal and uniform numbers
of a sweep are drawn beforehand from the chain's PCG64 stream
(`SeedSequence(seed).spawn(chains)`), so the compiled sweep can be compared
with the Python reference on identical inputs.
`test_inclusion_probability_matches_exact_marginal_likelihood` checks the
inclusion probability against the ratio of the exact Gaussian marginal
likelihoods `N(r; 0, σ²I + v ww')` and `N(r; 0, σ²I)`.

**Diagnostics** (Vehtari et al. 2021). Draws after burn-in and thinning,
arrays `(chains, draws)`, each chain split in halves. R-hat: the maximum of
the classic `sqrt(V̂⁺/W)` computed on rank-normalised draws (normal scores
of pooled fractional ranks, offset 3/8) and on rank-normalised folded draws
`|x − median|`. ESS: multi-chain autocorrelation from FFT autocovariances,
`ρ_t = 1 − (W − mean_chain acov_t)/V̂⁺`; pairs `(ρ_{2k}, ρ_{2k+1})` are kept
while their sum is positive (Geyer's initial positive sequence; lags 0 and 1
always), a positive last even `ρ` is kept, the pairs are made
non-increasing (initial monotone sequence), and
`τ = −1 + 2Σ_{t≤T} ρ_t + ρ_{T+1}`, `ESS = mn/max(τ, 1/log10(mn))` (Stan's
reference algorithm). Bulk ESS on rank-normalised split draws; tail ESS =
min of the ESS of the indicators `x ≤ q05` and `x ≤ q95`. MCSE of the mean =
SD of all draws / `sqrt(ESS of the split draws)`. All four agree with the
independent ArviZ 0.23.4 implementation to ≤ 8·10⁻¹⁶ relative on eleven
chain types (`benchmarks/mcmc_diagnostics_crosscheck.py`,
`docs/validation/mcmc_diagnostics_crosscheck.json`); ArviZ is used only as a
test oracle, not as a dependency. Gate: every
monitored scalar (except the count of non-zero effects) and every GEBV must
have R-hat `< rhat_max` and bulk and tail ESS `≥ ess_min`; otherwise the
run length is doubled up to `max_iterations`, and then the run fails with
`ABP-E405`. If storing the GEBV draws would exceed `5·10⁷` values,
GEBV-level diagnostics are skipped and reported as not computed.

**Traces and posterior predictive checks.** Every saved draw of every
monitored scalar is written per chain to `mcmc_trace_<trait>.csv`. At each
saved draw a replicate `y_rep = Xb + Wβ + e_rep`, `e_rep ~ N(0, σ²e I)`, is
simulated from a separate PCG64 stream per chain (so the chains are
unchanged) and `p_T = P(T(y_rep) ≥ T(y) | y)` is estimated for `T` = SD,
skewness, minimum and maximum; `p_T` outside `[0.01, 0.99]` is logged as a
warning. Predictive checks test the fit of the model to the data; they do
not replace the convergence diagnostics or SBC, which test the computation.

**Simulation-based calibration** (Talts et al. 2018;
`benchmarks/sbc_bayes.py`). With hyper-parameters fixed and passed to the
sampler (`BayesConfig.priors`), parameters are drawn from the prior, data
simulated from the model, the sampler run, and the rank of each true value
among `L` posterior draws recorded; under a correct computation the ranks
are uniform on `{0, …, L}` (ties at point masses broken at random). The
sampler's intercept prior is flat, so the intercept cannot be drawn from
it; because the posterior of every other quantity depends on `y` only
through the residuals after projecting out `X`, a fixed intercept is valid
for SBC of those quantities. Results: §3 of the validation report.

**Tests.** `test_brr_fixed_variances_matches_exact_gaussian_posterior`
(posterior means within 5 MCSE and SDs within 10% of the exact MME
posterior), `test_bayescpi_recovers_sparse_architecture`,
`test_all_methods_run_and_report_diagnostics`,
`test_seed_determinism_and_chain_independence`,
`test_native_sweep_matches_python_reference` (1e−12),
`test_iid_draws`, `test_ar1_effective_sample_size` (ESS of AR(1) chains
against `n(1 − φ)/(1 + φ)`), `test_nonmixing_chains_are_flagged`,
`test_agreement_with_arviz_reference_values` (RNG-free chains, 1e−10),
`test_example10_converges_and_writes_outputs`,
`test_nonconvergence_withholds_results`.

## 17. Metafounders

Code: `abp/core/metafounders.py` (`MetafounderPedigree`, `ml_general`,
`estimate_gamma_gls`, `single_step_mf`, `read_gamma_file`);
`abp/workflows/metafounder_inputs.py` (assignment, provenance, contrast with
the base); C++ kernel `ml_general` in `abp/_native.cpp`. Registry id
`ped.metafounders`.

**Why.** Ordinary `A` treats unknown parents as unrelated, non-inbred base
animals. Genomic relationships built with allele frequency 0.5 (`G05`) refer
to an older, related base. Single step that mixes the two bases (finding F6)
biases genotyped animals; tuning `G` to `A22` (`match_a22`) moves `G` to the
pedigree base only on average. Metafounders move the pedigree to the genomic
base instead (Legarra et al. 2015; Christensen 2012 for one metafounder).

**Model.** `k` metafounders (base populations) with relationship matrix
`Γ` (`k × k`, symmetric positive definite). Every unknown parent is assigned
to one metafounder (a pedigree code with the declared prefix, or the declared
default). Breeding values of animals and metafounders satisfy
`uᵢ = (u_s + u_d)/2 + mᵢ`, where a parent may be a metafounder, and
`u_MF ~ N(0, σ² Γ)`. The extended relationship matrix over (animals,
metafounders) is defined recursively:

    A[p, p'] = Γ[p, p'],   A[i, j] = (A[j, sᵢ] + A[j, dᵢ])/2,   A[i, i] = 1 + A[sᵢ, dᵢ]/2.

A plain unknown parent would be a metafounder with `γ = 0`, whose inverse does
not exist; ABP therefore refuses unassigned unknown parents (`ABP-E205`).

**Computation (derived and tested here).** With `T` the animal-only
transmission matrix and `Q` the gene fractions from each metafounder (the same
recursion as for genetic groups, §12):

    A^Γ = T D T' + Q Γ Q',     dᵢ = 1 − (A_ss + A_dd)/4,

where `A_pp := Γ_pp` for a metafounder parent. A founder whose parents both
come from metafounder `p` has `dᵢ = 1 − γ_pp/2` and `Aᵢᵢ = 1 + γ_pp/2`; two
such founders have relationship `γ_pp`. `diag(A^Γ)` uses a generalised
Meuwissen–Luo trace (`ml_general`): `Aᵢᵢ = 1 + A_sd/2` directly when a parent
is a metafounder (with `A_{s,p} = (QΓ)_{s,p}`), and
`Aᵢᵢ = Σⱼ Tᵢⱼ² dⱼ + qᵢ'Γqᵢ` otherwise. With `c = e = fext = 0` the trace is the
ordinary algorithm, which is tested. The inverse of the extended matrix is
Henderson's rules with metafounders acting as parents (contributions
`dᵢ⁻¹ [1, −½, −½] ⊗ [1, −½, −½]`) plus `Γ⁻¹` on the metafounder block, and
`log|A_ext| = log|Γ| + Σ log dᵢ`. `dᵢ ≤ 10⁻⁸` means that `Γ` is inconsistent
with the pedigree and is refused. Products `A^Γ x` use Colleau's two
triangular solves plus `QΓQ'x`.

**Γ.** Either from a file (`metafounder_1, metafounder_2, gamma`; every pair
exactly once; a non-empty `gamma_provenance` is required and written to the
manifest with the file's SHA-256), or estimated from the genotypes
(`gamma_source = "genotypes_gls"`):

1. base allele frequencies by generalized least squares (Gengler et al.
   2007; Garcia-Baccino et al. 2017): `E[M] = 2 Q₂ P` for genotyped animals,
   `Cov` of one marker's dosages ≈ `2p(1−p) A₂₂` (ordinary `A`), so
   `P̂ = C Q₂'A₂₂⁻¹ M / 2` with `C = (Q₂'A₂₂⁻¹Q₂)⁻¹`; missing dosages are
   replaced by `2Q₂P̂` and the estimate is iterated to a fixed point
   (tolerance 1e−10); estimates outside [0, 1] are clipped and counted;
2. `Γ_raw = 8 (P̂ − ½)(P̂ − ½)'/m`, which is on the scale of `G05`;
3. sampling correction (default on, derived here):
   `E[Γ_raw] = Γ + 4 E[p(1−p)] C` under the covariance above, so
   `Γ = Γ_raw − 4 s̄ C` with `s̄` the mean of `p̂(1−p̂)` over metafounders and
   markers (a pooled approximation).

A metafounder without genotyped descendants (or collinear gene fractions:
smallest eigenvalue of `Q₂'A₂₂⁻¹Q₂` ≤ 10⁻¹⁰ × largest) stops with `ABP-E300`;
a non-positive-definite result stops with `ABP-E302`. Neither is repaired.

**Single step.** `G` must be `G05` (`frequency_source = "fixed_0.5"`) and is
not tuned (`tuning = "none"`); a blend uses `A₂₂^Γ`. Then

    H⁻¹ = A_ext⁻¹ + embed(G*⁻¹ − (A₂₂^Γ)⁻¹),   log|H| = log|A_ext| + log|G*| − log|A₂₂^Γ|,

over (animals, metafounders), with `diag(H)` for non-genotyped equations
from `Hᵢᵢ = Aᵢᵢ + bᵢ'(G* − A₂₂)bᵢ`, `bᵢ = A₂₂⁻¹A[2, i]` (metafounder rows use
`A[p, 2] = (Γ Q₂')_p`).

**Outputs and the base contrast.** Animal solutions `uᵢ` include the genetic
level of the base population, which is shared by all its descendants; its
uncertainty (prior variance `γσ²`) enters every animal's PEV but not the
ranking. The estimable quantity used for decisions is the contrast with the
reference metafounder `r` (spec `metafounders.reference`, default the default
metafounder or the only one):

    cᵢ = uᵢ − u_r,   PEV(cᵢ) = C^{ii} + C^{rr} − 2C^{ir},
    Var(cᵢ) = σ²(Aᵢᵢ − 2(QΓ)ᵢᵣ + Γ_rr),   reliability = 1 − PEV(cᵢ)/Var(cᵢ).

`C^{ir}` needs one extra solve per analysis. `Var(cᵢ) = σ²(1 − γ/2)` for a
founder of `r`, i.e. the variance within the base population. The EBV file
has `ebv`, `pev`, `reliability` (absolute scale) and `ebv_vs_base`,
`pev_vs_base`, `reliability_vs_base`; the `inbreeding` column is relative to
the metafounder base (founders have `F = γ/2`). Metafounder solutions are in
`metafounder_solutions_<trait>.csv`.

**Scale of σ².** σ² under metafounders refers to the metafounder base; the
equivalent within-population variance is `σ²(1 − γ/2)` (Legarra et al. 2015).
REML under metafounders estimates σ² on that scale; a variance estimated
without metafounders must not be reused unchanged.

**Tests.** `test_extended_matrix_inverse_and_logdet_match_tabular_reference`
(native and Python kernels): `A_ext`, its inverse and log-determinant equal
an independent implementation of the recursive definition
(`tests/reference/dense_reference.py::tabular_a_metafounders`) to 1e−11;
a mutation of the `dᵢ` formula makes it fail.
`test_zero_inputs_reduce_to_ordinary_meuwissen_luo`,
`test_native_and_python_kernels_agree_on_a_random_pedigree` (4,000 animals,
3 metafounders; Colleau product against the sparse inverse),
`test_blup_with_metafounders_equals_v_form_model` (solutions, PEV,
reliability to 1e−9), `test_reml_loglik_and_score_under_metafounders_match_v_form`
(1e−10), `test_single_step_with_metafounders_matches_dense_h`,
`test_gamma_estimation_recovers_simulated_base_and_correction_reduces_bias`
(gene drop from two correlated base populations, 4 seeds; error < 0.03, the
correction reduces the diagonal bias), `test_g05_is_on_the_metafounder_scale`,
`test_invalid_inputs_are_refused`, and the workflow tests in
`tests/test_metafounder_workflow.py` (including the base contrast against the
V-form PEV matrix). Simulation evidence: §7.1 of the validation report.

## 18. Sparse selected inversion (exact PEV at scale)

Code: `abp/solvers/selinv.py` (`SelectedInverse`, `symbolic_cholesky`,
`takahashi`), C++ kernels `symbolic_cholesky` and `takahashi` in
`abp/_native.cpp`; used by `abp/solvers/mme.py` (`SparseLU.selected_inverse`,
`pev_diagonal`), `abp/solvers/multitrait.py` (PEV blocks) and
`abp/solvers/reml.py` (traces). Registry id `num.selected_inversion`.

**Factor.** SuperLU with the minimum-degree ordering of `C + C'`, symmetric
mode and no pivoting gives `B = C[q][:, q] = L U` with `q = argsort(perm_c)`
(verified numerically); for a symmetric positive-definite `C`, `U = D L'`
with `D = diag(U)`, which is checked (relative deviation ≤ 1e−8) before use.

**Pattern.** SuperLU stores only numerically non-zero entries, so exact
cancellations can remove an entry that the recurrence below needs. ABP
therefore computes the symbolic Cholesky pattern of `B` from the elimination
tree (column `j` = rows `> j` of `B[:, j]` ∪ the patterns of its children
minus `j`; the parent of `j` is the smallest row; Liu 1990), embeds SuperLU's
values into it and verifies that every stored SuperLU entry lies inside it.
A symbolic Cholesky pattern is closed under the recurrence.

**Recurrence** (Takahashi, Fagan & Chin 1973; Erisman & Tinney 1975). For
`Z = B⁻¹` and columns `j = n−1, …, 0` with strictly lower pattern `S_j`:

    Z_ij = − Σ_{k ∈ S_j} L_kj Z_ik    (i ∈ S_j),     Z_jj = 1/d_j − Σ_{k ∈ S_j} L_kj Z_kj.

The C++ kernel finds `Z_ik` by a merge walk over the sorted patterns of
columns `j` and `k` (`S_j ∩ {> k} ⊆ S_k`), so the cost is of the order of the
factorization. The result holds every entry of `C⁻¹` on the pattern of
`L + L'`, which contains the pattern of `C`: diagonal PEV, per-animal
multi-trait blocks whenever the traits are coupled, `tr(K⁻¹C^{kk})` and
`tr(C⁻¹W'W)` for REML. A request outside the pattern is refused
(`ABP-E303`); for multi-trait blocks of uncoupled traits (zero genetic and
residual covariances) ABP falls back to solves for unit vectors.

**Where it is used.** `solver.method = "auto"` with `pev = "exact"` above the
dense limit selects sparse direct with selected inversion (previously refused
above 30,000 equations). REML uses the dense path up to 12,000 equations
within the memory budget, otherwise sparse factorization with selected
inversion; the choice is recorded as `trace_method` in the REML output.

**Tests.** `test_selected_inverse_equals_dense_inverse_on_pattern` (every
stored entry against `numpy.linalg.inv`, 1e−15),
`test_native_kernels_equal_python_reference`,
`test_symbolic_pattern_closes_over_numeric_cancellation` (a 4 × 4 matrix whose
numeric factor has an exact zero that SuperLU drops),
`test_entries_outside_the_pattern_are_refused`, `test_memory_guard`,
`test_sparse_pev_and_reliability_equal_dense_path`,
`test_sparse_reml_traces_equal_dense_path` (log-likelihood, score, EM update
and AI matrix), `test_multitrait_pev_blocks_sparse_equal_dense` (coupled and
uncoupled traits). Scale: `benchmarks/run_benchmarks.py --only selinv`.

## 19. Multi-trait REML

Code: `abp/solvers/multitrait_reml.py` (`MTREMLEvaluator`, `mt_reml_fit`);
workflow `abp/workflows/multitrait.py`. Registry id `reml.multi_trait`.

**Model.** As §8: `Var(u) = K ⊗ G0` (animal-major), `Var(e_r) = R0[o_r, o_r]`
for the observed traits `o_r` of record `r`, trait-specific fixed designs.
Parameters `θ = (vech G0, vech R0)`; direction `E_jk = e_j e_k' + e_k e_j'`
(`e_j e_j'` on the diagonal).

**Likelihood** (Henderson's form; equal to the marginal V-form, tested):

    −2 log L = Σ_r log|R0[o_r,o_r]| + t log|K| + q log|G0| + log|C| + y'Py,
    y'Py = y'R⁻¹y − s'W'R⁻¹y.

**Quantities of C⁻¹** (both on the pattern of `C`, hence available from
selected inversion, §18): `T_jk = Σ_ab K⁻¹_ab C^{(a,j),(b,k)}` and
`Q_r = W_r C⁻¹ W_r'` for every record.

**Scores.** Genetic direction `E`:
`½[tr(G0⁻¹EG0⁻¹(U'K⁻¹U + T)) − q tr(EG0⁻¹)]`. Residual direction `E`:
`½Σ_r[e_r'R_r⁻¹E_rR_r⁻¹e_r + tr(R_r⁻¹E_rR_r⁻¹Q_r) − tr(R_r⁻¹E_r)]` with
`E_r = E[o_r,o_r]`. **AI matrix** `½F'PF` with working variates
`f_E = Z vec(U M')`, `M = EG0⁻¹` (genetic; from `Z'Py = (K⊗G0)⁻¹û`) and
`f_E = D_E R⁻¹e` (residual), one multi-RHS solve.

**EM update** (first three iterations and whenever an AI step fails):
`G0 ← (U'K⁻¹U + T)/q`; `R0 ← (1/n_rec) Σ_r E[e_r e_r' | y]`, where for a
record with missing traits `m` the second moments are completed by
`B = R_mo R_oo⁻¹`: `E[e_m e_o'] = B S_oo`,
`E[e_m e_m'] = B S_oo B' + (R_mm − B R_om)`, `S_oo = ê_oê_o' + Q_r`. Missing
traits are handled exactly; nothing is imputed into `y`. EM keeps both
matrices positive semi-definite.

**Control.** AI steps are halved (at most 6 times) until both matrices are
positive definite and `log L` does not decrease; otherwise EM. Convergence:
relative parameter change and Newton decrement `< reml.tol`. A (nearly)
singular estimate (smallest eigenvalue ≤ 10⁻⁸ × largest) stops with
`ABP-E300`: boundary handling is not implemented for the multi-trait case.
Stored zeros of `K⁻¹` (exact cancellation, e.g. a son mated to his own dam)
are dropped before the trace pairs are formed. Trace path: dense inverse up
to 2,000 equations, sparse selected inversion above (example 13, 6,390
equations: 44.2 s dense, 7.9 s sparse, identical log L).

**Tests.** `test_loglik_equals_v_form_up_to_constant_and_scores_match_finite_differences`
(exact log L; scores against central differences of the V-form, 1e−6),
`test_em_step_increases_loglik_and_keeps_pd`,
`test_optimum_equals_independent_nelder_mead` (Cholesky-parametrised V-form,
2e−3), `test_dense_and_sparse_trace_paths_agree`,
`test_three_traits_with_structural_missing_pattern`,
`test_stored_zeros_in_k_inverse_are_ignored`, `test_refusals`,
`test_example13_multitrait_reml_end_to_end`. Simulation evidence:
`benchmarks/mt_calibration_study.py` (validation report §7).

## 20. Threshold (probit) model for categorical traits

Code: `abp/solvers/threshold.py` (`threshold_blup`); workflow
`abp/workflows/evaluate.py::_run_threshold_trait`. Registry id
`threshold.probit`.

**Model** (Gianola & Foulley 1983; Harville & Mee 1984). Ordered categories
`y_i ∈ {1..K}` from a liability `l_i = x_i'b + Σ z_ik'u_k + e_i`,
`e_i ~ N(0, 1)`, thresholds `−∞ = τ_0 < τ_1 < … < τ_{K−1} < τ_K = ∞`:
`P(y_i = k) = Φ(τ_k − η_i) − Φ(τ_{k−1} − η_i)`. With an intercept `τ_1 = 0`.
Random terms on the liability scale with **known** variances; the residual
variance is 1 (identification; the spec refuses another value).

**Estimation.** Joint posterior mode of `(b, u, τ)` (flat priors on `b`, `τ`)
of `Σ log P(y_i) − ½ Σ_k u_k'K_k⁻¹u_k/σ_k²`, strictly concave (Pratt 1981).
With `a_i = τ_{y−1} − η_i`, `b_i = τ_y − η_i`, `P_i = Φ(b) − Φ(a)`:

    g_a = −φ(a)/P,  g_b = φ(b)/P,
    H_aa = aφ(a)/P − φ(a)²/P²,  H_bb = −bφ(b)/P − φ(b)²/P²,  H_ab = φ(a)φ(b)/P²,

with `φ(±∞) = (±∞)φ(±∞) = 0`; the chain rule through `a` and `b` gives the
gradient and Hessian in `(η, τ)`. Newton–Raphson with step-halving (thresholds
kept ordered, log posterior non-decreasing) until the largest step is
< 10⁻¹⁰. `log P` uses the upper-tail form when `a > 0` for numerical
stability.

**Uncertainty.** PEV is the diagonal of the inverse negative Hessian at the
mode (Laplace approximation) and reliabilities `1 − PEV/(σ²K_ii)` are on the
liability scale; both are labelled approximate.

**Tests.** `test_mode_equals_independent_optimizer_and_laplace_pev_equals_numerical_hessian`
(BFGS on an independent `scipy.stats.norm` objective, 2e−5; PEV against the
inverse of a central-difference Hessian, 0.2%),
`test_binary_trait_and_category_relabelling` (codes ×10 give identical
results; reversed order negates the solutions), `test_refusals`,
`test_example14_threshold_workflow`, `test_spec_rules_for_categorical_traits`.
Simulation evidence: `benchmarks/threshold_study.py`.

## 21. Sparse LDL' factorization

Code: `abp/solvers/cholesky.py` (`SparseLDL`, `mindegree_order`,
`ldl_numeric_python`, `ldl_solve_python`), C++ kernels `mindegree_order`,
`ldl_numeric`, `ldl_solve`; factory `abp/solvers/mme.py::make_sparse_factor`.
Registry id `num.sparse_ldl`.

* **Ordering:** minimum degree on the explicit elimination graph (George &
  Liu 1981): eliminate a node of smallest current degree (ties: smallest
  index), join its neighbours into a clique. Rows of degree
  `> max(16, 10√n)` (an intercept linked to every recorded animal) are set
  aside and eliminated last, as in AMD — this took the ordering of the
  100,500-equation benchmark from 83 s to about 2 s. The C++ and Python
  orders are identical (tested).
* **Symbolic:** the elimination-tree pattern of §18 on `B = C[q][:, q]`.
* **Numeric (up-looking):** for each `k`, scatter `B[:k, k]`, take its
  elimination-tree reach, process it in increasing index order (a
  descendant always precedes its ancestors):
  `y_i −= L_ij y_j` over the filled part of column `j`, `L_kj = y_j/d_j`,
  `d_k = B_kk − Σ L_kj y_j`. `d_k ≤ 0` stops with `ABP-E404`.
* **Solve:** `x = P'L⁻ᵀD⁻¹L⁻¹Pb` for many right-hand sides; `log|C| = Σ log d_k`.

`solver.factorization = "auto"` uses this factor when the compiled kernel is
present and SuperLU otherwise; `"superlu"` keeps the independent reference
path. Both give the same selected inverse (tested to 1e−12 and on the
100,500-equation benchmark to 6e−14).

**Tests.** `test_ldl_solve_logdet_inverse_equal_dense` (native and Python,
with and without a dense row), `test_native_ordering_and_numeric_equal_python_reference`,
`test_minimum_degree_reduces_fill_against_natural_order`,
`test_not_positive_definite_is_refused`, `test_blup_ldl_equals_superlu_and_dense`.

## 22. APY inverse of G

Code: `abp/core/genomic.py::apy_inverse`; workflow
`abp/workflows/genomic_inputs.py`. Registry id `gen.apy`.

With core animals `c` and non-core animals `n` (Misztal, Legarra & Aguilar
2014):

    G_APY⁻¹ = [[G_cc⁻¹, 0], [0, 0]] + [[−G_cc⁻¹G_cn], [I]] M⁻¹ [[−G_ncG_cc⁻¹, I]],
    m_i = g_ii − g_ic G_cc⁻¹ g_ci,   log|G_APY| = log|G_cc| + Σ log m_i.

`G_APY` equals `G*` on the core blocks and the non-core diagonal; non-core
off-diagonals become `G_nc G_cc⁻¹ G_cn`. **APY is a different model**, not an
approximation of `G*⁻¹` that ABP hides: the manifest records the core size,
the selection rule (random with `genomic.apy_seed`), the core IDs and their
hash. `G_cc` must be positive definite and every `m_i > 0` (`ABP-E302`
otherwise); a singular `G*` is acceptable when the core is below its rank,
so the full-`G*` definiteness check is skipped with APY. Cost
`O(c³ + nc²)`. Limitation of this version: the single-step blocks
(`A22⁻¹`, `G_APY⁻¹`) are still stored densely (memory `O(n₂²)`).

**Tests.** `test_apy_inverse_equals_inverse_of_implied_g_and_logdet`,
`test_all_animals_in_core_reproduce_the_exact_inverse`,
`test_singular_g_is_handled_when_core_is_full_rank`,
`test_single_step_with_apy_matches_explicit_h`, `test_workflow_example05_with_apy`.

## 23. References (additions)

* Erbe M, Hayes BJ, Matukumalli LK, et al. (2012) J Dairy Sci 95:4114–4129.
* Habier D, Fernando RL, Kizilkaya K, Garrick DJ (2011) BMC Bioinformatics 12:186.
* Legarra A, Reverter A (2018) Genet Sel Evol 50:53.
* Meuwissen THE (1997) J Anim Sci 75:934–940.
* Meuwissen THE, Hayes BJ, Goddard ME (2001) Genetics 157:1819–1829.
* Quaas RL (1988) J Dairy Sci 71:1338–1345.
* Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021) Bayesian Analysis 16:667–718.
* Westell RA, Quaas RL, Van Vleck LD (1988) J Dairy Sci 71:1310–1318.
* Christensen OF (2012) Genet Sel Evol 44:37.
* Erisman AM, Tinney WF (1975) Commun ACM 18:177–179.
* Garcia-Baccino CA, Legarra A, Christensen OF, et al. (2017) Genet Sel Evol 49:34.
* Gengler N, Mayeres P, Szydlowski M (2007) Animal 1:21–28.
* Legarra A, Christensen OF, Vitezica ZG, Aguilar I, Misztal I (2015) Genetics 200:455–468.
* Liu JWH (1990) SIAM J Matrix Anal Appl 11:134–172.
* Takahashi K, Fagan J, Chin M-S (1973) Proc 8th PICA Conference, Minneapolis, 63–71.
* George A, Liu JWH (1981) Computer Solution of Large Sparse Positive Definite Systems. Prentice-Hall.
* Gianola D, Foulley JL (1983) Genet Sel Evol 15:201–224.
* Gilmour AR, Thompson R, Cullis BR (1995) Biometrics 51:1440–1450.
* Harville DA, Mee RW (1984) Biometrics 40:393–408.
* Johnson DL, Thompson R (1995) J Dairy Sci 78:449–456.
* Misztal I, Legarra A, Aguilar I (2014) J Dairy Sci 97:3943–3952.
* Pratt JW (1981) J Am Stat Assoc 76:103–106.
