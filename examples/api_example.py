#!/usr/bin/env python3
"""Using ABP as a Python library (every snippet of docs/api.md, runnable).

Run from the repository root:  python examples/api_example.py
"""

from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.genomic import allele_frequencies, apply_g_policy, single_step, vanraden_g
from abp.core.pedigree import Pedigree
from abp.decision.selection_index import ebv_index, smith_hazel
from abp.solvers.blup import RandomTerm, blup
from abp.solvers.multitrait import MTData, build_and_solve
from abp.solvers.reml import reml_fit
from abp.workflows.evaluate import run_evaluation

# --- 1. Pedigree: order, inbreeding, A-inverse, relationships ---------------
ped = Pedigree.from_parent_ids(
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
    sires=[None, None, None, "1", "3", "1", "4", "3"],
    dams=[None, None, None, None, "2", "2", "5", "6"])
F = ped.inbreeding()                         # internal order = ped.ids
Ainv = ped.ainv()                            # scipy.sparse CSR
print("F:", {a: round(float(f), 4) for a, f in zip(ped.ids, F)})
print("A[7, 8] =", ped.relationship(["7"], ["8"])[0, 0])

# --- 2. Single-trait BLUP with known variances (Mrode 2005, Example 3.1) ----
rec_ids = ["4", "5", "6", "7", "8"]
y = np.array([4.5, 2.9, 3.9, 3.5, 5.0])
fixed = build_fixed_design({"sex": ["M", "F", "F", "M", "M"]},
                           [FixedTerm("sex", "factor")], intercept=False, n=5)
Z = sp.csr_matrix((np.ones(5), (np.arange(5), ped.index_of(rec_ids))), shape=(5, ped.n))
animal = RandomTerm("animal", Z, Ainv, ped.ids, genetic=True,
                    k_diag=1.0 + F, logdet_k=ped.logdet_a())
res = blup(y, fixed.X, [animal], {"animal": 20.0, "residual": 40.0})
t = res.terms["animal"]
for a, e, r in zip(t.labels, t.solution, t.reliability):
    print(f"animal {a}: EBV {e:+.3f}  reliability {r:.3f}")
print("solver:", res.solve.method, "| relative residual", f"{res.solve.rel_residual:.1e}")

# --- 3. REML ---------------------------------------------------------------
rng = np.random.default_rng(1)
n_anim = 300
ids = [f"a{i}" for i in range(n_anim)]
sires = [None] * 30 + [ids[k] for k in rng.integers(0, 15, n_anim - 30)]
dams = [None] * 30 + [ids[k] for k in rng.integers(15, 30, n_anim - 30)]
ped2 = Pedigree.from_parent_ids(ids, sires, dams)
A = ped2.a_dense()
u_true = np.linalg.cholesky(A) @ rng.standard_normal(ped2.n) * np.sqrt(2.0)
y2 = 10 + u_true + rng.normal(0, 2.0, ped2.n)            # one record per animal
X2 = build_fixed_design({}, [], intercept=True, n=ped2.n).X
Z2 = sp.identity(ped2.n, format="csr")
term2 = RandomTerm("animal", Z2, ped2.ainv(), ped2.ids, True,
                   k_diag=1 + ped2.inbreeding(), logdet_k=ped2.logdet_a())
fit = reml_fit(y2, X2, [term2], {"algorithm": "ai", "max_iter": 100, "tol": 1e-8, "start": None})
print("REML:", fit.status, {k: round(v, 3) for k, v in fit.variances.items()},
      "h2 =", round(fit.heritability, 3), "+-", round(fit.heritability_se, 3))

# --- 4. Genomic relationships and single step --------------------------------
geno_ids = ped2.ids[-100:]
M = rng.integers(0, 3, (100, 1500)).astype(float)
p = allele_frequencies(M)
G, d = vanraden_g(M, p)
A22 = ped2.a_submatrix(ped2.index_of(geno_ids))
Gstar, record = apply_g_policy(G, policy="blend", tuning="match_a22", A22=A22,
                               alpha=0.05, ridge=0.0)
ss = single_step(ped2, ped2.index_of(geno_ids), Gstar, a22=A22)
print("G* policy:", record.singular_policy, "tuning b =", round(record.tuning_b, 4),
      "| H-inverse nnz:", ss.h_inv.nnz)

# --- 5. Multi-trait BLUP -----------------------------------------------------
Y = np.column_stack([y2, 0.5 * y2 + rng.normal(0, 1, ped2.n)])
Y[rng.random(ped2.n) < 0.4, 1] = np.nan                    # trait 2 partly missing
Xb = [build_fixed_design({}, [], True, int((~np.isnan(Y[:, j])).sum())).X for j in range(2)]
G0 = np.array([[2.0, 0.8], [0.8, 1.0]])
R0 = np.array([[4.0, 0.5], [0.5, 1.5]])
mt = build_and_solve(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), 1 + ped2.inbreeding(), G0, R0)
print("multi-trait EBV of", ped2.ids[0], mt.ebv[0].round(3), "reliabilities", mt.reliability[0].round(3))

# --- 6. Selection indices ----------------------------------------------------
r = smith_hazel(P=np.array([[4.0, 1.0], [1.0, 9.0]]), C=np.array([[2.0], [3.0]]),
                a=np.array([1.0]), G_H=np.array([[5.0]]), info_names=["own", "sibs"],
                trait_names=["H"], proportion_selected=0.1)
print("index b =", r.b.round(4), "accuracy", round(r.accuracy, 4), "response", round(r.response_objective, 3))
idx, idx_rel = ebv_index(mt.ebv, mt.pev_blocks, np.array([1.0, 2.0]), G0, 1 + ped2.inbreeding())
print("best animal by index:", ped2.ids[int(np.argmax(idx))], "reliability", round(float(idx_rel.max()), 3))

# --- 7. Unknown-parent groups (QP transformation) -----------------------------
from abp.core.upg import GroupAssignment, group_fractions, upg_structure_parts

ped3 = Pedigree.from_parent_ids(["s", "d", "x"], [None, None, "s"], [None, None, "d"])
groups = GroupAssignment(labels=("G_import",),
                         sire_group=np.array([0, -1, -1]),   # s: sire from group G_import
                         dam_group=np.array([0, -1, -1]))    # s: dam from group G_import
order3 = ped3.index_of(["s", "d", "x"])
print("gene fractions Q (s, d, x):", group_fractions(ped3, groups)[order3, 0])   # 1, 0, 0.5
k_inv, k_diag, logdet = upg_structure_parts(ped3, groups, "random", 1.0)
print("K-inverse with groups:", k_inv.shape, "| diag(K) for reliabilities:", k_diag.round(3))

# --- 8. Optimal contributions and a mating plan -------------------------------
from abp.decision.mating import allocate, forbidden_mask
from abp.decision.ocs import coancestry_target_from_delta_f, integer_matings, solve_ocs

cand = np.arange(ped2.n - 40, ped2.n)                     # last 40 animals as candidates
g_c = rng.normal(0, 1, cand.size)                          # merit (e.g. EBVs)
male_c = np.arange(cand.size) < 10                         # 10 males, 30 females
A_c = ped2.a_submatrix(cand)
n_mat = 30
cap = np.where(male_c, 10, 1)
cmax, ct = coancestry_target_from_delta_f(A_c, 0.02)
ocs = solve_ocs(g_c, A_c, male_c, np.zeros(cand.size), cap / (2.0 * n_mat), cmax)
print("OCS:", ocs.status, "coancestry", round(ocs.coancestry, 5), "<= C_max", round(cmax, 5),
      "| KKT stationarity", f"{ocs.kkt['stationarity_max_abs']:.1e}")
counts = integer_matings(ocs.c, cap, male_c, n_mat)       # (abp mate also repairs the ceiling)
s_i, d_i = np.flatnonzero(male_c & (counts > 0)), np.flatnonzero(~male_c & (counts > 0))
A_sd = A_c[np.ix_(s_i, d_i)]
forb, why = forbidden_mask(A_sd, 0.25, None, None, None)
plan = allocate(counts[s_i], counts[d_i], A_sd, forb)
print("mating plan:", len(plan.pairs), "pairs, mean progeny F", round(plan.mean_inbreeding, 5),
      "| forbidden:", why)

# --- 9. PLINK 1 binary genotypes ------------------------------------------------
from abp.io.plink import load_plink, write_bed

with tempfile.TemporaryDirectory() as tmp:
    write_bed(Path(tmp) / "demo", ["a1", "a2", "a3"],
              [("snp1", "1", 1000, "A", "G"), ("snp2", "1", 2000, "C", "T")],
              np.array([[0.0, 2.0], [1.0, np.nan], [2.0, 1.0]]))
    gd = load_plink(Path(tmp) / "demo", "DEMO-ASSEMBLY")
    print("PLINK:", gd.ids, gd.markers, "counted allele (A1):", gd.counted_allele,
          "dosages", np.where(gd.missing, np.nan, gd.dosage).tolist())

# --- 10. Bayesian marker regression and MCMC diagnostics ------------------------
from abp.solvers.bayes import BayesConfig, run_bayes
from abp.solvers.mcmc_diagnostics import summarize

Mb = rng.integers(0, 3, (300, 200)).astype(float)          # 300 animals x 200 markers
pb = allele_frequencies(Mb)
Wm = Mb - 2 * pb                                            # centred dosages, as for G
beta_true = np.zeros(200)
beta_true[rng.choice(200, 10, replace=False)] = rng.normal(0, 0.4, 10)
yb = Wm @ beta_true + rng.normal(0, 1, 300)
cfg = BayesConfig(method="BayesC", pi0=0.95, chains=4, iterations=1000, burn_in=200, thin=1,
                  seed=7, max_iterations=4000)
bres = run_bayes(yb, np.ones((300, 1)), Wm, Wm, cfg, float(2 * np.sum(pb * (1 - pb))))
if not bres.converged:                                      # never use unconverged results
    raise SystemExit("MCMC diagnostics failed; results withheld")
print("BayesC: converged after", bres.iterations, "iterations | h2",
      round(bres.summaries["h2"]["mean"], 3), "R-hat", round(bres.summaries["h2"]["rhat"], 4),
      "| worst GEBV R-hat", round(bres.gebv_diagnostics["max_rhat"], 4),
      "| markers with the highest inclusion probability:",
      sorted(int(j) for j in np.argsort(-bres.inclusion_prob)[:3]))
chains = rng.normal(size=(4, 1000))                         # any (chains, draws) array
print("diagnostics of iid draws:", {k: round(v, 3) for k, v in summarize(chains).items()
                                    if k in ("rhat", "ess_bulk", "ess_tail")})

# --- 11. Whole workflow from an analysis spec ----------------------------------
root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    out = run_evaluation(root / "01_textbook_mrode_3_1" / "analysis.toml", Path(tmp) / "run",
                         console=False)
    print("workflow:", out.status, "->", sorted(p.name for p in out.out_dir.iterdir()))
