# ABP Python API

Version 0.8.0. Every snippet below is taken from `examples/api_example.py`,
which the test suite runs (`tests/test_examples.py::test_api_example_script_runs`).
The mathematics behind each function is in [`methods.md`](methods.md).

The public surface of this version is the set of modules and names listed here. Other
names (leading underscore, or not listed) may change without notice. All
arrays are NumPy `float64`, and sparse matrices are SciPy CSR.

## Layers

| Package | Role |
|---|---|
| `abp.core` | pedigree (`pedigree`), unknown-parent groups (`upg`), metafounders (`metafounders`), matrix-free single-step operators (`ssop`), fixed-effect design (`design`), genomic relationships (`genomic`), model compiler (`model`), analysis spec (`spec`) |
| `abp.io` | delimited-table reader with provenance (`tables`), PLINK 1 binary reader (`plink`) |
| `abp.qc` | pedigree, phenotype and genotype QC with structured findings |
| `abp.solvers` | MME assembly and solvers (`mme`), sparse selected inversion (`selinv`), sparse LDL' (`cholesky`), multi-trait REML (`multitrait_reml`), threshold model (`threshold`), threshold Gibbs sampler (`threshold_gibbs`), PEV with REML uncertainty (`vc_uncertainty`), sampled PEV (`pev_sampling`), single-trait BLUP (`blup`), REML (`reml`), multi-trait BLUP (`multitrait`), Bayesian marker models (`bayes`), MCMC diagnostics (`mcmc_diagnostics`) |
| `abp.decision` | selection indices (`selection_index`), optimal contributions (`ocs`), mating allocation (`mating`) |
| `abp.workflows` | end-to-end evaluation, LR validation (`validation_lr`), mating plans (`mating_plan`), outputs, manifests, reports |
| `abp.errors` | `ABPError` and the error-code table |

## Errors

Every detected failure raises `abp.errors.ABPError`:

```python
from abp.errors import ABPError
try:
    ...
except ABPError as err:
    err.code          # "ABP-E200"
    err.spec.name     # "PEDIGREE_CYCLE"
    err.exit_status   # 4
    err.message       # occurrence-specific text
    err.details       # machine-readable context (animals, lines, values ...)
    err.to_dict()     # JSON-ready form, as written to manifest.json
```

## 1. Pedigree: `abp.core.pedigree.Pedigree`

```python
from abp.core.pedigree import Pedigree

ped = Pedigree.from_parent_ids(
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
    sires=[None, None, None, "1", "3", "1", "4", "3"],
    dams=[None, None, None, None, "2", "2", "5", "6"])
F = ped.inbreeding()          # internal order = ped.ids (parents before offspring)
Ainv = ped.ainv()             # scipy.sparse CSR, exact Henderson/Quaas rules
ped.relationship(["7"], ["8"])  # -> array([[0.25]])
```

| Member | Returns |
|---|---|
| `Pedigree.from_parent_ids(ids, sires, dams)` | ordered, validated pedigree. `None` = unknown parent. Every named parent must be listed (the QC layer adds missing parents). Raises on duplicates, cycles, self-parenthood. |
| `.ids`, `.sire`, `.dam`, `.generation`, `.n` | internal order; parent indices, with `-1` for unknown |
| `.index_of(ids)` | internal indices of IDs |
| `.inbreeding()` | `F` (C++ kernel if available); `.inbreeding_kernel` names the kernel that ran |
| `.mendelian_d()` | `d_i` of `A = T D T'` |
| `.ainv()` | sparse `A⁻¹` |
| `.logdet_a()` | log-determinant of A, `log det A` |
| `.a_times(x)` | `A @ x` for a vector or matrix, without forming `A` |
| `.a_columns(idx)`, `.a_submatrix(idx)`, `.a_dense()` | dense parts of `A` (`a_dense` refuses more than 20,000 animals) |

## 2. Fixed effects: `abp.core.design`

```python
from abp.core.design import FixedTerm, build_fixed_design
fixed = build_fixed_design({"sex": ["M", "F", "F", "M", "M"]},
                           [FixedTerm("sex", "factor")], intercept=False, n=5)
fixed.X                    # full-column-rank sparse design
fixed.labels, fixed.kept   # all candidate columns and which were kept
fixed.constrained_labels   # columns set to zero for identifiability
```

## 3. Single-trait BLUP: `abp.solvers.blup`

```python
import numpy as np, scipy.sparse as sp
from abp.solvers.blup import RandomTerm, blup

rec_ids = ["4", "5", "6", "7", "8"]
y = np.array([4.5, 2.9, 3.9, 3.5, 5.0])
Z = sp.csr_matrix((np.ones(5), (np.arange(5), ped.index_of(rec_ids))), shape=(5, ped.n))
animal = RandomTerm("animal", Z, Ainv, ped.ids, genetic=True,
                    k_diag=1.0 + F, logdet_k=ped.logdet_a())
res = blup(y, fixed.X, [animal], {"animal": 20.0, "residual": 40.0})
t = res.terms["animal"]      # .solution, .pev, .reliability, .labels
res.fixed_solution           # kept fixed-effect columns
res.solve.method, res.solve.selection_reason, res.solve.rel_residual
```

`RandomTerm(name, Z, k_inv, labels, genetic, k_diag=None, logdet_k=None)`
takes the inverse structure `K⁻¹` *without* variance scaling. `blup(y, X,
terms, variances, method="auto", compute_pev=True, tol=1e-10,
max_iter=10000, memory_budget_bytes=4 GiB)` returns a `BLUPResult`.
`method` is one of `auto`, `dense`, `sparse_direct` or `pcg` (see
`abp.solvers.mme.choose_method`).

## 4. REML: `abp.solvers.reml`

```python
from abp.solvers.reml import reml_fit
fit = reml_fit(y2, X2, [term2], {"algorithm": "ai", "max_iter": 100, "tol": 1e-8, "start": None})
fit.status          # "converged" or "converged_boundary"
fit.variances       # {"animal": ..., "residual": ...}; boundary components are 0.0
fit.se, fit.heritability, fit.heritability_se, fit.history
```

Optional arguments: `checkpoint_path`, `resume`, `fingerprint` (checkpoint
and resume) and `memory_budget_bytes`. Raises `ABPError("REML_NOT_CONVERGED")`
when the budget is exhausted. `REMLEvaluator(y, X, terms, budget).evaluate(theta)`
exposes log L, the score, the AI matrix and the EM update at any `theta`.

## 5. Genomic: `abp.core.genomic`

```python
from abp.core.genomic import allele_frequencies, apply_g_policy, single_step, vanraden_g
p = allele_frequencies(M)                  # M: n x m dosages of the counted allele
G, d = vanraden_g(M, p)                    # G = WW'/d
Gstar, record = apply_g_policy(G, policy="blend", tuning="match_a22", A22=A22,
                               alpha=0.05, ridge=0.0)
ss = single_step(ped2, ped2.index_of(geno_ids), Gstar, a22=A22)
ss.h_inv, ss.h_diag, ss.logdet_h
```

`apply_g_policy` raises `RELATIONSHIP_SINGULAR` when the result is not
positive definite. `record` holds every adjustment and the eigenvalues before
and after. `minor_allele_frequency(p)` returns `min(p, 1 − p)`.

## 6. Multi-trait BLUP: `abp.solvers.multitrait`

```python
from abp.solvers.multitrait import MTData, build_and_solve
mt = build_and_solve(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), 1 + ped2.inbreeding(), G0, R0)
mt.ebv          # q x t
mt.pev_blocks   # q x t x t
mt.reliability  # q x t
```

In `MTData(Y, X_per_trait, animal_col)`, `Y` is `n_records × t` with `NaN`
for missing values. `X_per_trait[j]` has one row per record in which trait
`j` is observed. `animal_col` maps records to columns of `K`.

## 7. Selection indices: `abp.decision.selection_index`

```python
from abp.decision.selection_index import ebv_index, smith_hazel
r = smith_hazel(P=np.array([[4.0, 1.0], [1.0, 9.0]]), C=np.array([[2.0], [3.0]]),
                a=np.array([1.0]), G_H=np.array([[5.0]]), info_names=["own", "sibs"],
                trait_names=["H"], proportion_selected=0.1)
r.b, r.reliability, r.accuracy, r.response_objective
idx, idx_rel = ebv_index(mt.ebv, mt.pev_blocks, np.array([1.0, 2.0]), G0, 1 + ped2.inbreeding())
```

`smith_hazel(..., restrict=["trait"])` computes the restricted index.
`selection_intensity(p)` returns the truncation-selection intensity.

## 8. Unknown-parent groups: `abp.core.upg`

```python
from abp.core.upg import GroupAssignment, group_fractions, upg_structure_parts
ped3 = Pedigree.from_parent_ids(["s", "d", "x"], [None, None, "s"], [None, None, "d"])
groups = GroupAssignment(labels=("G_import",),
                         sire_group=np.array([0, -1, -1]),   # per animal, pedigree order
                         dam_group=np.array([0, -1, -1]))
group_fractions(ped3, groups)        # Q, animals x groups
k_inv, k_diag, logdet = upg_structure_parts(ped3, groups, "random", 1.0)
```

| Name | Returns |
|---|---|
| `GroupAssignment(labels, sire_group, dam_group)` | group index (or `-1`) of the unknown sire/dam of every animal, in `ped.ids` order |
| `ainv_with_groups(ped, groups)` | `A*⁻¹` of size `n + g` (Henderson rules with groups as parents) |
| `group_fractions(ped, groups)` | `Q` (`n × g`) |
| `upg_structure_parts(ped, groups, effect, ratio)` | `(K⁻¹, diag(K), log det K)` over `(u*, g)`; for `"fixed"` `diag(K)` is `NaN` and `log det K` is `None` |
| `check_fixed_groups_estimable(X, ZQ, labels)` | rank diagnostics, or `ABPError` `MODEL_NOT_IDENTIFIABLE` |
| `upg_structure(ped, groups, effect, ratio)` | a `GeneticStructure` (animals, then groups) for `abp.core.model.build_single_trait` |

`load_pedigree(..., group_prefix="UPG:")` in `abp.qc.pedigree` returns
`PedigreeData.groups` from group codes in the parent columns.

## 9. Optimal contributions and mating: `abp.decision.ocs`, `abp.decision.mating`

```python
from abp.decision.mating import allocate, forbidden_mask
from abp.decision.ocs import coancestry_target_from_delta_f, integer_matings, solve_ocs
cmax, ct = coancestry_target_from_delta_f(A_c, 0.02)
ocs = solve_ocs(g_c, A_c, male_c, np.zeros(cand.size), cap / (2.0 * n_mat), cmax)
ocs.c, ocs.merit, ocs.coancestry, ocs.status, ocs.kkt
counts = integer_matings(ocs.c, cap, male_c, n_mat)
forb, why = forbidden_mask(A_sd, 0.25, None, None, None)
plan = allocate(counts[s_i], counts[d_i], A_sd, forb)
plan.pairs, plan.offspring_inbreeding, plan.mean_inbreeding, plan.checks
```

| Name | Notes |
|---|---|
| `solve_ocs(g, A, male, lo, hi, max_coancestry)` | exact optimum with KKT certificate; `ABPError` if the ceiling is below the minimum achievable coancestry |
| `coancestry_target_from_delta_f(A, delta_f)` | `(C_max, C_t)` |
| `integer_matings(c, capacity, male, n_matings)` | largest-remainder rounding per sex |
| `repair_integer_plan(counts, g, A, male, capacity, n_matings, max_coancestry)` | `(counts, moves)`; enforces the ceiling on whole matings |
| `forbidden_mask(A_sd, max_pair_relationship, carrier_s, carrier_d, max_affected_risk, explicit=None)` | `(mask, counts per reason)` |
| `allocate(n_s, m_d, A_sd, forbidden, carrier_s=None, carrier_d=None)` | minimum-inbreeding plan (`MatingPlan`); `ABPError` naming unmatchable parents if infeasible |
| `abp.workflows.mating_plan.plan_matings(spec, out, force=False)` | the `abp mate` workflow; returns the published folder |

## 10. PLINK input: `abp.io.plink`

```python
from abp.io.plink import load_plink, write_bed
write_bed(Path(tmp) / "demo", ["a1", "a2", "a3"],
          [("snp1", "1", 1000, "A", "G"), ("snp2", "1", 2000, "C", "T")],
          np.array([[0.0, 2.0], [1.0, np.nan], [2.0, 1.0]]))
gd = load_plink(Path(tmp) / "demo", "DEMO-ASSEMBLY")  # reads demo.bed/.bim/.fam
gd.ids, gd.markers, gd.counted_allele                # counted allele = A1
np.where(gd.missing, np.nan, gd.dosage)              # dosages of A1
```

```python
gd8 = load_plink(Path(tmp) / "demo", "DEMO-ASSEMBLY", storage="int8")
```

`load_plink(prefix, assembly, storage="float64")`: with `storage="int8"` the
dosages are `int8` with −1 for a missing call (`abp.qc.genotype.to_int8(g)` converts
a loaded `GenotypeData`; it refuses fractional dosages).
`decode_bed(raw, n_samples, n_variants, dtype=...)` decodes SNP-major bytes;
`write_bed(prefix, ids, markers, dosage_a1)` writes a fileset (tests,
conversions).

## 11. Bayesian marker models: `abp.solvers.bayes`, `abp.solvers.mcmc_diagnostics`

```python
from abp.solvers.bayes import BayesConfig, run_bayes
from abp.solvers.mcmc_diagnostics import summarize
cfg = BayesConfig(method="BayesC", pi0=0.95, chains=4, iterations=1000, burn_in=200, thin=1,
                  seed=7, max_iterations=4000)
bres = run_bayes(yb, np.ones((300, 1)), Wm, Wm, cfg, float(2 * np.sum(pb * (1 - pb))))
if not bres.converged:                   # never use unconverged results
    raise SystemExit("MCMC diagnostics failed; results withheld")
bres.gebv_mean, bres.gebv_sd, bres.beta_mean, bres.inclusion_prob, bres.summaries
summarize(chains)                        # any (chains, draws) array
```

`run_bayes(y, X, W_train, W_all, cfg, sum2pq)`: `W_train` rows match `y`;
GEBVs are returned for the rows of `W_all`; `sum2pq = Σ 2pⱼ(1 − pⱼ)` scales
the default priors. `BayesConfig` fields mirror the `[bayes]` spec keys.
`run_bayes` does **not** raise on non-convergence (it returns
`converged = False`); the workflow turns that into `ABP-E405`. Diagnostics:
`rhat`, `bulk_ess`, `tail_ess`, `mcse_mean`, `summarize`, `passes`.

## 12. Whole workflow: `abp.workflows.evaluate.run_evaluation`

```python
from abp.workflows.evaluate import run_evaluation
out = run_evaluation("examples/01_textbook_mrode_3_1/analysis.toml", "runs/ex01",
                     force=False, resume=False, console=False)
out.status      # "passed"
out.out_dir     # Path of the published folder
out.results     # dict, identical to results.json
out.manifest    # dict, identical to manifest.json
```

On failure `run_evaluation` raises `ABPError` after writing
`<out>.failed-<run id>/manifest.json`. Related helpers:
`abp.workflows.validate.validate_inputs(spec)` (QC only),
`abp.core.spec.load_spec(path)` (validated spec object) and
`abp.examples.sheep.write_sheep_example(out, seed)` (synthetic data). With a
`[validation]` section the run also performs LR validation
(`abp.workflows.validation_lr.run_lr`); results are in `out.results["validation"]`.

## 13. Metafounders: `abp.core.metafounders`

```python
from abp.core.metafounders import MetafounderPedigree
mf_groups = GroupAssignment(labels=("MF:A",), sire_group=np.array([0, 0, -1]),
                            dam_group=np.array([0, 0, 0]))
mfp = MetafounderPedigree(ped3, mf_groups, gamma=np.array([[0.4]]))
mfp.diag_ext()                  # [1.2, 1.2, 1.2, 0.4]: animals, then the metafounder
mf_term = RandomTerm("animal", Z3, mfp.ainv_ext(), list(mfp.labels), genetic=True,
                     k_diag=mfp.diag_ext(), logdet_k=mfp.logdet_ext())
```

| Name | Returns |
|---|---|
| `MetafounderPedigree(ped, groups, gamma)` | validates `Γ` (symmetric, positive definite) and the assignment (every unknown parent has a metafounder, else `ABP-E205`); computes `Q`, `diag(A^Γ)` and the Mendelian factors `d` |
| `.ainv_ext()` | sparse inverse of the extended matrix over `(animals, metafounders)` |
| `.logdet_ext()`, `.diag_ext()` | `log|A_ext|`; `diag(A_ext)` (for reliabilities) |
| `.a_times(x)`, `.a_submatrix(idx)`, `.ext_columns(idx)`, `.cov_mf()` | `A^Γ x` without forming `A^Γ`; dense `A^Γ[idx, idx]`; columns of `A_ext`; `A_ext[:, metafounders]` |
| `.inbreeding()` | `F` relative to the metafounder base |
| `estimate_gamma_gls(ped, groups, geno_index, M, missing=None, sampling_correction=True)` | `GammaEstimate` with `gamma`, `gamma_uncorrected`, base frequencies, the GLS information inverse and notes (`.to_dict()` for manifests) |
| `read_gamma_file(path, labels)` | `Γ` from `metafounder_1, metafounder_2, gamma` |
| `single_step_mf(mfp, geno_index, Gstar, a22=None)` | `SingleStepMF` with `h_inv`, `h_diag`, `logdet_h`, `cov_mf` (`G*` must be on the `G05` scale) |
| `ml_general(sire, dam, c, e, fext)` | generalised Meuwissen–Luo trace `(diag A, d, kernel)` |

In a workflow, `[metafounders]` (user manual §5.17) builds these objects and
also reports EBVs against the reference metafounder (`ebv_vs_base`).

## 14. Sparse selected inversion: `abp.solvers.selinv`

```python
res_sparse = blup(y, fixed.X, [animal], {"animal": 20.0, "residual": 40.0},
                  method="sparse_direct")
si = res_sparse.solve.factor.selected_inverse()      # computed once, cached
si.diagonal(idx)          # C^-1[i, i] for equations idx
si.entries(rows, cols)    # C^-1[rows[k], cols[k]] on the factor pattern
si.trace_product(M, offset)
```

| Name | Returns |
|---|---|
| `SelectedInverse(lu, C, memory_budget_bytes)` | entries of `C⁻¹` on the pattern of the Cholesky factor of `C` (from a SuperLU factorization with symmetric permutation and no pivoting); `ABP-E500` if the factor exceeds the budget |
| `.diagonal(idx)`, `.entries(rows, cols)` | entries of `C⁻¹` in the original equation order; entries outside the pattern raise `ABPError` (`UNSUPPORTED_COMBINATION`) rather than being guessed |
| `.trace_product(M, offset)` | `Σ Mᵢⱼ C⁻¹[offset+i, offset+j]` for sparse `M` |
| `.nnz_factor`, `.kernel` | size of the symbolic factor; `"native_cpp/native_cpp"` or `"python/python"` |
| `SparseLU.selected_inverse()` | the cached `SelectedInverse` of a sparse factor |
| `symbolic_cholesky(B)`, `takahashi(colptr, rowidx, lval, d)` | the two phases (compiled kernel when available; `*_python` are the references) |

`pev_diagonal`, multi-trait PEV blocks and REML traces use this
automatically on the sparse path.

## 15. Multi-trait REML: `abp.solvers.multitrait_reml`

```python
from abp.solvers.multitrait_reml import mt_reml_fit
fit_mt = mt_reml_fit(MTData(Y_sim, Xs, rec_an), ped2.ainv(), ped2.logdet_a(),
                     {"tol": 1e-8, "max_iter": 200})
fit_mt.G0, fit_mt.R0, fit_mt.to_dict(["t1", "t2"])["genetic_correlations"]
```

| Name | Returns |
|---|---|
| `mt_reml_fit(data, k_inv, logdet_k, cfg, memory_budget_bytes, start=None)` | `MTREMLFit` with `G0`, `R0`, `loglik`, `iterations`, `se`, `trace_method`, `history`; `ABP-E403` if not converged, `ABP-E300` at a boundary |
| `MTREMLEvaluator(data, k_inv, logdet_k).evaluate(theta)` | log-likelihood, scores, AI matrix and EM update at `theta = (vech G0, vech R0)` |
| `default_start(data)` | the data-based starting matrices |

## 16. Threshold model: `abp.solvers.threshold`

```python
from abp.solvers.threshold import threshold_blup
thr = threshold_blup(cat, Xc, [RandomTerm("animal", Zc, ped2.ainv(), ped2.ids, True,
                                          k_diag=1 + ped2.inbreeding())],
                     {"animal": 0.3, "residual": 1.0}, intercept=True)
thr.thresholds, thr.terms["animal"].solution, thr.terms["animal"].pev
```

`threshold_blup(y, X, terms, variances, intercept, tol=1e-10, max_iter=100,
compute_pev=True)` returns `ThresholdResult` (`fixed_solution`, `thresholds`
with the first fixed at 0 when `intercept`, `categories`, `terms` of
`TermResult` on the liability scale with Laplace PEV, `iterations`,
`log_posterior`). `variances["residual"]` must be 1.

## 17. Sparse LDL': `abp.solvers.cholesky`

```python
from abp.solvers.cholesky import SparseLDL
fac_ldl = SparseLDL(C_small)          # minimum-degree ordering, symbolic, numeric
fac_ldl.solve(rhs); fac_ldl.logdet(); fac_ldl.selected_inverse().diagonal(idx)
```

| Name | Returns |
|---|---|
| `SparseLDL(C, memory_budget_bytes, dense_tail=True)` | factor with `solve`, `logdet`, `selected_inverse`, `inverse_block`, `nnz_factor`, `kernel`, `split` (first column of the dense trailing block factorized by LAPACK, round 12; `n` when none; `dense_tail=False` turns it off); `ABP-E404` for a non-positive pivot |
| `abp.solvers.cholesky.dense_tail_split(colptr, n, min_size, max_size, speed_ratio)` | the block-size rule (methods §21) |
| `mindegree_order(C)` | `(order, kernel)` |
| `abp.solvers.mme.make_sparse_factor(C, budget, factorization)` | `SparseLDL` or `SparseLU` (`"auto"`, `"ldl"`, `"superlu"`) |

`blup(..., factorization=...)` and `build_and_solve(..., factorization=...)`
pass the choice through; the workflow reads `solver.factorization`.

## 18. APY: `abp.core.genomic.apy_inverse`

```python
from abp.core.genomic import apy_inverse
apy = apy_inverse(G_apy_demo, np.arange(0, G.shape[0], 2))
apy.g_inv, apy.logdet, apy.g_apy, apy.m, apy.record
```

`apy_inverse(G, core)` returns `APYInverse`: the inverse and log-determinant
of `G_APY`, the implied `G_APY`, the conditional variances `m` of the
non-core animals and a record for the manifest. `single_step(...,
g_inverse=(apy.g_inv, apy.logdet))` uses it in `H⁻¹`; the workflow reads
`genomic.apy_core_size` and `genomic.apy_seed`.

## 19. PEV including REML uncertainty: `abp.solvers.vc_uncertainty`

```python
from abp.solvers.vc_uncertainty import kackar_harville_delta
res2 = blup(y2, X2, [term2], fit.variances)
delta = kackar_harville_delta(y2, X2, [term2], fit.variances, fit.cov, fit.cov_names, "animal")
pev_total = res2.terms["animal"].pev + delta
```

`reml_fit` now returns `REMLFit.cov` (the inverse average-information matrix,
`None` at a boundary) and `REMLFit.cov_names`. `kackar_harville_delta(y, X,
terms, variances, cov, names, genetic_term, method="auto", ...)` returns
`Δ_i = g_i' Σ g_i` for every equation of the genetic term (methods §23).

## 20. Threshold-model variance estimation: `abp.solvers.threshold`

```python
from abp.solvers.threshold import threshold_laplace_reml
tfit = threshold_laplace_reml(cat, Xc, [RandomTerm("animal", Zc, ped2.ainv(), ped2.ids, True,
                                                   k_diag=1 + ped2.inbreeding(),
                                                   logdet_k=ped2.logdet_a())], intercept=True)
tfit.variances, tfit.loglik, tfit.evaluations
```

| Name | Returns |
|---|---|
| `laplace_loglik(y, X, terms, variances, intercept, init=None)` | `(log p(y│s) up to a constant, ThresholdResult at the mode)` |
| `threshold_laplace_reml(y, X, terms, intercept, start=None, tol=1e-6, max_eval=400)` | `ThresholdVarianceFit` (`variances` incl. `residual = 1`, `loglik`, `evaluations`, `status`, `note`); `ABP-E300` at the search bound, `ABP-E403` if not converged |
| `threshold_blup(..., init=None, dense_limit=12000)` | as before; `init` warm-starts Newton's method, `ThresholdResult.logdet_neg_hessian` is `log|H|` at the mode |

The estimates are biased with little information per animal (methods §24).

## 21. Reduced-rank multi-trait REML: `abp.solvers.multitrait_reml`

```python
from abp.solvers.multitrait_reml import mt_reml_fit_reduced_rank
rr = mt_reml_fit_reduced_rank(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), ped2.logdet_a(),
                              1, {"tol": 1e-8, "max_iter": 200})
res_rr = build_and_solve(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), 1 + ped2.inbreeding(),
                         None, rr.R0, loadings=rr.loadings)
```

| Name | Returns |
|---|---|
| `mt_reml_fit_reduced_rank(data, k_inv, logdet_k, rank, cfg, memory_budget_bytes, start=None)` | `ReducedRankFit` (`loadings` Λ, `G0 = ΛΛ'`, `R0`, `rank`, `loglik`, `evaluations`, `gradient_norm`, `to_dict(traits)`) |
| `ReducedRankEvaluator(data, k_inv, logdet_k, rank).m2ll(Lam, R0)` | `(−2 log L, factor)` of the reduced-rank model |
| `assemble_multitrait(data, k_inv, G0, R0, loadings=None)`, `build_and_solve(..., loadings=None)` | with `loadings`, the latent-factor system; `build_and_solve` reports EBVs and PEV blocks on the trait scale |

## 22. Matrix-free single step: `abp.core.ssop`

```python
from abp.core.genomic import spd_inverse_and_logdet
from abp.core.ssop import A22InverseOperator, DenseInverseOperator, SingleStepHInverse
g_idx = ped2.index_of(geno_ids)
h_op = SingleStepHInverse(ped2.ainv().tocsr(), g_idx,
                          DenseInverseOperator(spd_inverse_and_logdet(Gstar, "G*")[0]),
                          A22InverseOperator(ped2.ainv(), g_idx))
ss_mf = blup(y2, X2, [RandomTerm("animal", Z2, h_op, ped2.ids, True)],
             {"animal": 2.0, "residual": 4.0}, method="pcg", compute_pev=False)
```

| Name | Role |
|---|---|
| `A22InverseOperator(ainv, geno_index)` | callable `v -> A22⁻¹ v` from sparse blocks of `A⁻¹` |
| `DenseInverseOperator(g_inv)` / `APYOperator(g_cc, g_cn, g_nn_diag, core, n2)` | callables `v -> G*⁻¹ v` (dense, or APY without forming `G_APY⁻¹`); `.diag()` |
| `apy_blocks_from_genotypes(Wc, d, core, policy, alpha, ridge, a22_core_cols, a22_diag)` | `G*_cc`, `G*_cn`, `diag(G*)_n` from centred genotypes |
| `SingleStepHInverse(a_inv, geno_index, g_op, a22_op)` | usable as `RandomTerm.k_inv`; `blup` then assembles `A⁻¹` and applies the correction inside PCG (`MixedModelSystem.extra`); dense/sparse direct solvers and PEV are refused |
| `DenseInverseOperator(g_inv, g_chol)`, `APYOperator.sample(rng)`, `SingleStepHInverse(..., ped=ped).sample(rng)` | draws from `N(0, G*)`, `N(0, G_APY)` and `N(0, H)` without forming `H` (round 6) |
| `apy_blocks_from_dosage(M_int8, p, core, policy, alpha, ridge, a22_core_cols, a22_diag)` | the APY blocks from `int8` dosages (−1 = missing), centred block by block |
| `a_block(ped, rows, cols)` | `A[rows][:, cols]` by Colleau products in column blocks |

## 23. Threshold-model Gibbs sampler: `abp.solvers.threshold_gibbs`

```python
from abp.solvers.threshold_gibbs import ThresholdGibbsConfig, threshold_gibbs
tg = threshold_gibbs(cat, Xc, [RandomTerm("animal", Zc, ped2.ainv(), ped2.ids, True,
                                          k_diag=1 + ped2.inbreeding())], True,
                     ThresholdGibbsConfig(chains=2, iterations=600, burn_in=200, thin=1,
                                          max_iterations=600, seed=1))
tg.variances["animal"]["mean"], tg.converged
```

`ThresholdGibbsConfig(chains, iterations, burn_in, thin, seed, rhat_max, ess_min,
max_iterations, nu=-2, s2=0, start=None, fix_variances=False)`;
`threshold_gibbs(y, X, terms, intercept, cfg, genetic_term=None)` returns
`ThresholdGibbsResult` (`terms` with posterior means and variances, `fixed_mean`,
`thresholds_mean`, `variances` with mean/median/SD/2.5%/97.5%, `summaries`,
`ebv_diagnostics`, `converged`, `iterations`, `traces`, `acceptance`). The caller
decides what to do with an unconverged result (the workflow withholds it).
Helpers: `rtruncnorm(rng, mean, lo, hi)`, `precision_root(k_inv)`,
`draw_location(problem, factor, liabilities, variances, rng)`.

## 24. Multi-trait PEV including REML uncertainty

```python
from abp.solvers.vc_uncertainty import kackar_harville_delta_multitrait
d_mt = kackar_harville_delta_multitrait(mt_d, ped2.ainv(), 1 + ped2.inbreeding(), fit_mt.G0,
                                        fit_mt.R0, fit_mt.cov)
```

`MTREMLFit.cov` is the inverse average-information matrix of
`(vech G0, vech R0)` (`None` when not positive definite). The function returns a
`q × t × t` array to add to `MTResult.pev_blocks`.

## 25. Sampled PEV: `abp.solvers.pev_sampling`

```python
smp = sampled_pev(ped2.n, X2, [RandomTerm("animal", Z2, h_smp, ped2.ids, True)],
                  {"animal": 2.0, "residual": 4.0}, "animal", n_samples=50, seed=3)
smp.pev, smp.reliability, smp.reliability_se
```

`sampled_pev(n, X, terms, variances, genetic_term, n_samples=200, seed=..., tol,
max_iter, estimator="orthogonal")`: terms must be a matrix-free single-step term
(with `ped` set) or iid terms. `estimator="orthogonal"` (default since 0.8):
reliability `mean h² / (mean h² + mean d²)` minus its second-order bias;
`"ratio"`: the 0.6 estimator `1 − mean d² / mean u*²` (kept for comparison).

## 26. Reduced-rank REML gradient

```python
ev_rr = MR.ReducedRankEvaluator(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), ped2.logdet_a(), 1)
m2ll, grad = ev_rr.value_and_gradient(MR._rr_pack(rr.loadings, rr.R0))
```

`value_and_gradient(x)` returns `−2 log L` and its analytic gradient in the
parameters `x` (lower-trapezoidal `Λ`, then `chol R0` with log diagonal);
`ReducedRankFit.newton_decrement` reports the convergence criterion.

## 27. Rank selection and reduced-rank Kackar–Harville PEV

```python
from abp.solvers.vc_uncertainty import kackar_harville_delta_reduced_rank
sel = MR.select_rank(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), ped2.logdet_a(),
                     {"tol": 1e-8, "max_iter": 200})
d_rr = kackar_harville_delta_reduced_rank(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(),
                                          1 + ped2.inbreeding(), rr.x, 2, 1, rr.cov_x)
```

`select_rank(data, k_inv, logdet_k, cfg, memory_budget_bytes, margin=AIC_MARGIN)`
fits ranks `1..t` and returns `{"table", "chosen_rank", "fits", "criterion",
"margin"}`; each table row has `rank`, `n_parameters`, and `loglik`, `aic`,
`delta_aic` or `error`. The choice starts at the highest fitted rank and moves to a
lower rank only if its AIC is smaller by at least `margin` (default 2).
`n_parameters(t, rank)` counts the covariance parameters.
`kackar_harville_delta_reduced_rank(data, k_inv, k_diag, x, t, r, cov_x)` returns
the `q × t × t` correction to add to the PEV blocks; `ReducedRankFit.x` and
`.cov_x` (`2 H⁻¹`) come from the fit.

## 28. Threshold Gibbs priors

`ThresholdGibbsConfig(nu=4, s2={"animal": 0.1, "pe": 0.1})` gives each variance a
scaled inverse-χ² prior (`s2` a number for all terms or a dict per term); the default
`nu=-2, s2=0` is the uniform prior. In a spec: `bayes.variance_prior`,
`bayes.nu`, `bayes.prior_variances`.

## 29. Multi-trait threshold model: `abp.solvers.mt_threshold_gibbs`

```python
from abp.solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs
mtt = mt_threshold_gibbs(Ymt, 1, Xmt, np.arange(ped2.n), ped2.ainv(), MTThresholdGibbsConfig(
    chains=2, iterations=300, burn_in=100, thin=1, max_iterations=300, seed=1, prior_nu=5.0,
    prior_G0=np.array([[1.0, 0.0], [0.0, 0.3]])))
```

`mt_threshold_gibbs(Y, cat, X, animal_col, k_inv, cfg)`: `Y` is `n × t` (NaN =
missing, column `cat` integer categories), `X` a list of per-trait fixed designs
(rows = records where the trait is observed; each with an intercept) or one design
shared by the traits, `animal_col` the column of each record's animal in `K`.
`MTThresholdGibbsConfig(chains, iterations, burn_in, thin, seed, rhat_max, ess_min,
max_iterations, start_G0, start_R0, fix_covariances, scale_move, shear_moves,
prior_nu, prior_G0)`; `prior_nu=None, prior_G0=None` is the flat prior on `G0`.
Returns `MTThresholdGibbsResult` (`ebv`, `pev` (`q × t`), `pev_blocks` (`q × t × t`),
`fixed_mean` per trait, `thresholds_mean`, `G0`/`R0` summaries (`mean`, `sd`,
`q025`, `q975`), `derived` (heritabilities `h2_j`, genetic correlations `rG_i_j`),
`summaries` (R-hat, ESS per scalar), `ebv_diagnostics`, `converged`, `traces`).
Building blocks (tested separately): `MTProblem`, `draw_location`, `draw_R0`,
`draw_G0`, `split_design`.

`SparseLDL.refactor_values(data)` (round 8) refactorises from values stored on
exactly the pattern of `SparseLDL.C` (no sparse re-assembly);
`Pedigree.a_times` uses the compiled Colleau kernel when available.

## 30. Bayesian multi-trait linear model, residual groups, inbreeding kernel (round 9)

```python
lin = mt_threshold_gibbs(Y, None, Xb, np.arange(ped2.n), ped2.ainv(), MTThresholdGibbsConfig(
    chains=4, iterations=4000, burn_in=1000, thin=4, max_iterations=16000, seed=1))
blk = mt_threshold_gibbs(Y, None, Xb, np.arange(ped2.n), ped2.ainv(), MTThresholdGibbsConfig(
    chains=4, iterations=4000, burn_in=1000, thin=4, seed=1, residual_groups=[[0], [1]]))
```

`cat=None` gives the linear model: all traits continuous, no thresholds
(`thresholds_mean` is empty), `R0 | E ~ IW(E′E, n − t − 1)`. `residual_groups` (a
partition of the trait indices `0 … t−1`, at least two groups) makes `R0` block
diagonal with exact zeros between groups; a `start_R0` must respect the zeros. In
the spec: `bayes.method = "multitrait"` and `bayes.residual_groups = { trait = group }`
(manual §5.14, §7.16). `draw_R0(E, c, rng, groups=None)` draws one `R0` (`c=None`:
no categorical trait).

`Pedigree.inbreeding()` uses the compiled depth kernel
(`inbreeding_kernel == "native_cpp_depth_ml_colleau"`; the same `F` as the
Meuwissen–Luo reference). `abp._native.inbreeding_depth(sire, dam, mode=0)` returns
`(bytes of F, depths, meuwissen_luo_pairs, colleau_columns, colleau_depths)`; `mode`
1 and 2 force the trace and the column path (tests).

## 31. Permanent environment, R0 prior and scale moves in the multi-trait Gibbs samplers (round 10)

```python
pe = mt_threshold_gibbs(Y, None, Xb, animal_col, ped2.ainv(), MTThresholdGibbsConfig(
    chains=4, iterations=4000, burn_in=1000, thin=4, max_iterations=16000, seed=1,
    prior_nu_pe=6.0, prior_P0=np.diag([0.5, 0.3]),          # optional proper P0 prior
    prior_nu_r=6.0, prior_R0=np.diag([2.0, 1.0])),          # optional proper R0 prior
    pe_col=animal_col)                                      # level of each record (0..m-1)
pe.P0["mean"], pe.pe_mean, pe.derived["c2_0"]
```

`pe_col` adds `pe ~ N(0, I_m ⊗ P0)` (repeated records: several records of an animal
share its level). `MTThresholdGibbsConfig` gains `start_P0`, `prior_nu_pe`/`prior_P0`
(IW prior of P0; default flat) and `prior_nu_r`/`prior_R0` (IW prior of every block of
continuous traits of R0; with a categorical trait that trait must form its own
residual group). `MTThresholdGibbsResult` gains `P0` (summaries) and `pe_mean`
(`m × t`); `derived` gains `c2_<j>`; heritabilities use `G + P + R`.
`draw_R0(E, c, rng, groups=None, nu=None, R_prior=None)`. `scale_move=True` (default)
now applies the scale move to every trait for `(u_j, G0)` and `(pe_j, P0)` (round 9:
the categorical trait only). In the spec: an iid term in `[model].random` and
`bayes.prior_covariance = { animal = ..., pe = ..., residual = ... }` (manual §5.14,
§7.16).

## 32. Maternal genetic effects in the multi-trait Gibbs samplers (round 11)

```python
mat = mt_threshold_gibbs(Y, None, Xb, animal_col, ped2.ainv(), MTThresholdGibbsConfig(
    chains=4, iterations=4000, burn_in=1000, thin=5, max_iterations=16000, seed=1),
    dam_col=dam_col,            # column in K of each record's dam, -1 = unknown
    pe_col=dam_level)           # optional maternal permanent environment (iid on the dam)
mat.ebv, mat.maternal_ebv, mat.maternal_pev, mat.genetic_blocks, mat.derived["m2_0"]
```

With `dam_col` every animal has `2t` genetic effects (direct traits, then maternal
traits); `G0`, `start_G0` and `prior_G0` are `2t × 2t`; `summaries` hold `G0_i_j`
over the `2t` columns and `rG_j_{t+j}` is the direct-maternal correlation of trait
`j`. One trait is allowed with `dam_col`. `MTProblem.incidence(k)` gives the
observations and animal indices that load genetic column `k`. In the spec: a random
term `{ name = "maternal", kind = "maternal" }` (dam from the pedigree; manual §7.16).

## 33. Maternal animal model by REML (round 13)

```python
from abp.solvers.maternal_reml import MaternalData, maternal_reml_fit, maternal_blup
from abp.solvers.vc_uncertainty import kackar_harville_delta_maternal
data = MaternalData(y, X, animal_idx, dam_idx,                 # dam_idx -1 = unknown
                    iid=[("mpe", dam_level_idx, n_dam_levels)])  # optional independent terms
fit = maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-8, "max_iter": 200})
fit.G0, fit.iid, fit.residual, fit.se, fit.derived["m2"], fit.derived["m2_se"], fit.status
beta, U, pev, iid_solutions, info = maternal_blup(data, ped.ainv(), fit.G0,
                                                  [fit.iid["mpe"]], fit.residual)
theta = [fit.G0[0, 0], fit.G0[0, 1], fit.G0[1, 1], fit.iid["mpe"], fit.residual]
delta = kackar_harville_delta_maternal(data, ped.ainv(), theta, fit.cov)   # q x 2 x 2
```

| Name | Meaning |
|---|---|
| `MaternalData(y, X, animal, dam, iid)` | one record per row; `animal`, `dam` index the relationship matrix; `iid` = `[(name, level index, n_levels)]` |
| `maternal_reml_fit(data, k_inv, logdet_k, cfg, memory_budget_bytes, start, dense)` | AI-REML with EM fallback; `status` `converged` or `converged_boundary` (an independent-term variance at 0, `start["boundary"]`); `ABP-E300` for a singular G0; `se`, `cov` (inverse AI, `names` order) withheld at a boundary; `derived` = h2, m2, r_AM (+ SE) |
| `MaternalREMLEvaluator(...).evaluate(theta)` | logL, score, AI, EM update at `theta = (s_A, s_AM, s_M, s_p..., s_e)` |
| `maternal_blup(data, k_inv, G0, iid_vars, residual)` | fixed effects, `U` (`q x 2`: direct, maternal), PEV blocks (`q x 2 x 2`), independent-term solutions, solver info (relative residual) |
| `kackar_harville_delta_maternal(data, k_inv, theta, cov)` | first-order PEV increase from the uncertainty of the REML estimates (`q x 2 x 2`) |

In the spec: `variances.mode = "reml"` (or `"known"` with `values = { animal = [[s_A,
s_AM], [s_AM, s_M]], mpe = ..., residual = ... }`) and a term `{ name = "maternal",
kind = "maternal" }`; one trait (manual §7.16).

## 34. Several categorical traits in the multi-trait threshold model (round 14)

```python
res = mt_threshold_gibbs(Y, [1, 2], Xb, animal_col, ped.ainv(), MTThresholdGibbsConfig(
    residual_groups=[[0], [1], [2]],            # every categorical trait in its own group
    prior_nu=5.0, prior_G0=np.diag([3.0, 0.25, 0.25]), seed=1))
res.thresholds_by_trait[1], res.categories_by_trait[2], res.traces["tau1_2"]
```

`cat` may be `None`, one index (unchanged results and names: `thresholds_mean`,
`categories`, `tau_<k>`) or a list of indices. With several, `residual_groups` must put
each categorical trait in a different group (`SPEC_INVALID` otherwise); the residual
covariance between two categorical traits is 0 by construction. `thresholds_mean` and
`categories` refer to the first categorical trait. In the spec: several
`type = "categorical"` traits with `bayes.method = "threshold"` and
`bayes.residual_groups` (manual §7.15).

