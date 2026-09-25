# ABP Python API

Version 0.2.0. Every snippet below is taken from `examples/api_example.py`,
which the test suite runs (`tests/test_examples.py::test_api_example_script_runs`).
The mathematics behind each function is in [`methods.md`](methods.md).

The public surface of this version is the set of modules and names listed here. Other
names (leading underscore, or not listed) may change without notice. All
arrays are NumPy `float64`, and sparse matrices are SciPy CSR.

## Layers

| Package | Role |
|---|---|
| `abp.core` | pedigree (`pedigree`), unknown-parent groups (`upg`), fixed-effect design (`design`), genomic relationships (`genomic`), model compiler (`model`), analysis spec (`spec`) |
| `abp.io` | delimited-table reader with provenance (`tables`), PLINK 1 binary reader (`plink`) |
| `abp.qc` | pedigree, phenotype and genotype QC with structured findings |
| `abp.solvers` | MME assembly and solvers (`mme`), single-trait BLUP (`blup`), REML (`reml`), multi-trait BLUP (`multitrait`), Bayesian marker models (`bayes`), MCMC diagnostics (`mcmc_diagnostics`) |
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

`decode_bed(raw, n_samples, n_variants)` decodes SNP-major bytes;
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
