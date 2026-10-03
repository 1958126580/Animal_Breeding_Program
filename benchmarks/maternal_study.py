#!/usr/bin/env python3
"""Calibration study of the Bayesian maternal animal model (round 11).

Per replicate: a random sex-consistent pedigree (``--animals``, 10% founders, dams
drawn among the earlier females, so dams have several offspring); one record per
non-founder animal: ``y = mu + a_animal + m_dam + c_dam + e`` with
``(a, m) ~ N(0, A (x) G0)``, ``G0 = [[4, -1], [-1, 2]]`` (direct-maternal correlation
-0.35), maternal permanent environment ``c`` (variance 1.5), residual 8 (the values
of example 18).  Fitted with ``mt_threshold_gibbs`` (one trait, ``dam_col``, iid term
on the dam), flat priors, 4 chains x ``--iterations`` (burn-in 1/4, thin 4); whether
R-hat/ESS passed is recorded and every fit is scored.

Scores (truth used only for scoring): posterior means of the variances; for the
direct and the maternal EBVs of all animals: MSE / mean PEV, coverage of nominal 95%
intervals, realized accuracy.  Means +- Monte-Carlo SE over replicates.  The posterior
medians of the (co)variances are recorded too (round 12).

``--prior equal`` (round 12, finding F19): weak proper inverse-Wishart priors on G0
(2 x 2), the maternal pe variance and the residual variance, each centred on an equal
share of the observed phenotypic variance Vp (G_prior = diag(Vp/4, Vp/4), P_prior = Vp/4,
R_prior = Vp/4; nu = dimension + 2, the smallest integer nu with a finite prior mean).
Uses the data only, never the true values; the shares are deliberately not tuned (the
maternal share Vp/4 is above the true value).

``--method reml`` (round 13): the same simulated data fitted by AI-REML
(:mod:`abp.solvers.maternal_reml`) and BLUP at the estimates (PEV from ``C^-1``, not
including the uncertainty of the estimates); records the estimates, their asymptotic SE,
whether ``estimate +- 1.96 SE`` covers the true value, and fits that stopped (singular
G0, ``ABP-E300``) or ended on the boundary; EBV scores with the plug-in PEV and with the
Kackar-Harville PEV (``*_kh``; interior fits).  REML takes seconds per replicate, so
``--replicates`` can be much larger.

Usage: python benchmarks/maternal_study.py [--replicates 20] [--workers 3]
       [--animals 1500] [--iterations 4000] [--prior flat|equal] [--method gibbs|reml]
       [--out docs/validation/maternal_study.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.core.pedigree import Pedigree  # noqa: E402
from abp.solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs  # noqa: E402

G0 = np.array([[4.0, -1.0], [-1.0, 2.0]])
VAR_C, VAR_E = 1.5, 8.0
N_ANIMALS, ITER = 1500, 4000
PRIOR = "flat"                        # --prior; inherited by forked workers
METHOD = "gibbs"                      # --method


def _prior_kwargs(y: np.ndarray) -> dict:
    if PRIOR == "flat":
        return {}
    vp = float(np.var(y, ddof=1))
    return {"prior_nu": 4.0, "prior_G0": np.diag([vp / 4, vp / 4]),
            "prior_nu_pe": 3.0, "prior_P0": np.array([[vp / 4]]),
            "prior_nu_r": 3.0, "prior_R0": np.array([[vp / 4]])}


def simulate(seed: int, n: int):
    from scipy.sparse.linalg import spsolve_triangular
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    nf = n // 10
    sires, dams = [], []
    for i in range(n):
        if i < nf:
            sires.append(None)
            dams.append(None)
        else:
            sires.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            dams.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    AM = F @ np.linalg.cholesky(G0).T
    dam = np.asarray(ped.dam, dtype=np.int64)
    rec = np.flatnonzero(dam >= 0)
    c = rng.normal(0, np.sqrt(VAR_C), n)
    y = 30 + AM[rec, 0] + AM[dam[rec], 1] + c[dam[rec]] + rng.normal(0, np.sqrt(VAR_E), rec.size)
    return ped, rec, dam, y, AM


def _score(true, ebv, pev):
    err = true - ebv
    return {"pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
            "realized_accuracy": float(np.corrcoef(true, ebv)[0, 1])}


def replicate_reml(seed: int) -> dict:
    from abp.errors import ABPError
    from abp.solvers.maternal_reml import MaternalData, maternal_blup, maternal_reml_fit
    ped, rec, dam, y, AM = simulate(seed, N_ANIMALS)
    damrec = dam[rec]
    levels, pe_col = np.unique(damrec, return_inverse=True)
    data = MaternalData(y, sp.csr_matrix(np.ones((rec.size, 1))), rec, damrec,
                        [("mpe", pe_col, levels.size)])
    t0 = time.time()
    out = {"seed": seed}
    try:
        fit = maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-8, "max_iter": 300})
    except ABPError as exc:
        out.update({"status": exc.code, "message": exc.message[:200],
                    "wall_s": time.time() - t0})
        return out
    th = {"direct": fit.G0[0, 0], "covariance": fit.G0[0, 1], "maternal": fit.G0[1, 1],
          "mpe": fit.iid["mpe"], "residual": fit.residual}
    truth = {"direct": G0[0, 0], "covariance": G0[0, 1], "maternal": G0[1, 1],
             "mpe": VAR_C, "residual": VAR_E}
    se_key = {"direct": "direct", "covariance": "direct-maternal covariance",
              "maternal": "maternal", "mpe": "mpe", "residual": "residual"}
    out.update({"status": fit.status, "iterations": fit.iterations,
                "estimates": {k: float(v) for k, v in th.items()},
                "se": None if fit.se is None else {k: fit.se[se_key[k]] for k in th},
                "cover95": None if fit.se is None else {
                    k: bool(abs(th[k] - truth[k]) <= 1.96 * fit.se[se_key[k]]) for k in th}})
    keep = [] if fit.iid["mpe"] == 0.0 else data.iid
    beta, U, pev, _, _ = maternal_blup(MaternalData(y, data.X, rec, damrec, keep), ped.ainv(),
                                       fit.G0, [fit.iid["mpe"]] if keep else [], fit.residual)
    out["direct"] = _score(AM[:, 0], U[:, 0], pev[:, 0, 0])
    out["maternal"] = _score(AM[:, 1], U[:, 1], pev[:, 1, 1])
    if fit.cov is not None:            # PEV including the uncertainty of the estimates
        from abp.solvers.vc_uncertainty import kackar_harville_delta_maternal
        th = np.array([fit.G0[0, 0], fit.G0[0, 1], fit.G0[1, 1], fit.iid["mpe"], fit.residual])
        D = kackar_harville_delta_maternal(data, ped.ainv(), th, fit.cov)
        out["direct_kh"] = _score(AM[:, 0], U[:, 0], pev[:, 0, 0] + D[:, 0, 0])
        out["maternal_kh"] = _score(AM[:, 1], U[:, 1], pev[:, 1, 1] + D[:, 1, 1])
    out["wall_s"] = time.time() - t0
    return out


def main_reml(a, seeds):
    t0 = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            reps = list(pool.imap(replicate_reml, seeds))
    else:
        reps = [replicate_reml(s) for s in seeds]
    ok = [r for r in reps if "estimates" in r]
    truth = {"direct": 4.0, "covariance": -1.0, "maternal": 2.0, "mpe": VAR_C,
             "residual": VAR_E}
    s = {"n": len(reps), "status": {}, "estimates": {}, "rmse": {}, "cover95": {}}
    for r in reps:
        s["status"][r["status"]] = s["status"].get(r["status"], 0) + 1
    for k, tv in truth.items():
        v = np.array([r["estimates"][k] for r in ok])
        s["estimates"][k] = {**_ms(v), "true": tv}
        s["rmse"][k] = float(np.sqrt(np.mean((v - tv) ** 2)))
        cv = [r["cover95"][k] for r in ok if r["cover95"] is not None]
        s["cover95"][k] = _ms(cv) if len(cv) > 1 else None
    for eff in ("direct", "maternal", "direct_kh", "maternal_kh"):
        rr = [r for r in ok if eff in r]
        s[eff] = {k: _ms([r[eff][k] for r in rr]) for k in rr[0][eff]}
        s[eff]["pev_ratio_median"] = float(np.median([r[eff]["pev_ratio"] for r in rr]))
    s["wall_s_per_replicate"] = _ms([r["wall_s"] for r in reps])
    doc = {"study": "maternal animal model fitted by AI-REML (round 13; F19 reference)",
           "replicates": len(reps), "seeds": f"{seeds[0]}..{seeds[-1]}", "animals": N_ANIMALS,
           "method": "reml", "true": {"G0": G0.tolist(), "var_c": VAR_C, "var_e": VAR_E},
           "wall_seconds": time.time() - t0, "summary": s, "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"wrote {a.out}")


def replicate(seed: int) -> dict:
    ped, rec, dam, y, AM = simulate(seed, N_ANIMALS)
    damrec = dam[rec]
    levels, pe_col = np.unique(damrec, return_inverse=True)
    t0 = time.time()
    g = mt_threshold_gibbs(y[:, None], None, sp.csr_matrix(np.ones((rec.size, 1))), rec,
                           ped.ainv(), MTThresholdGibbsConfig(
                               chains=4, iterations=ITER, burn_in=ITER // 4, thin=4,
                               max_iterations=ITER, seed=seed, **_prior_kwargs(y)),
                           pe_col=pe_col, dam_col=damrec)
    return {"seed": seed, "converged": g.converged, "wall_s": time.time() - t0,
            "max_rhat": float(max(v["rhat"] for v in g.summaries.values())),
            "min_ess_bulk": float(min(v["ess_bulk"] for v in g.summaries.values())),
            "G0": g.G0["mean"], "P0": g.P0["mean"][0][0], "R0": g.R0["mean"][0][0],
            "G0_median": g.G0["median"], "P0_median": g.P0["median"][0][0],
            "R0_median": g.R0["median"][0][0],
            "G0_cover95": [[bool(lo <= tv <= hi) for lo, hi, tv in zip(rl, rh, rt)]
                           for rl, rh, rt in zip(g.G0["q025"], g.G0["q975"], G0.tolist())],
            "direct": _score(AM[:, 0], g.ebv[:, 0], g.pev[:, 0]),
            "maternal": _score(AM[:, 1], g.maternal_ebv[:, 0], g.maternal_pev[:, 0])}


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size)}


def main():
    global N_ANIMALS, ITER, PRIOR, METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=1500)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--prior", choices=("flat", "equal"), default="flat")
    ap.add_argument("--method", choices=("gibbs", "reml"), default="gibbs")
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "maternal_study.json"))
    a = ap.parse_args()
    N_ANIMALS, ITER, PRIOR, METHOD = a.animals, a.iterations, a.prior, a.method
    t0 = time.time()
    seeds = list(range(a.first_seed, a.first_seed + a.replicates))
    if METHOD == "reml":
        return main_reml(a, seeds)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            reps = list(pool.imap(replicate, seeds))
    else:
        reps = [replicate(s) for s in seeds]
    s = {"converged": sum(r["converged"] for r in reps),
         "direct_variance": {**_ms([r["G0"][0][0] for r in reps]), "true": 4.0},
         "maternal_variance": {**_ms([r["G0"][1][1] for r in reps]), "true": 2.0},
         "direct_maternal_covariance": {**_ms([r["G0"][0][1] for r in reps]), "true": -1.0},
         "maternal_pe_variance": {**_ms([r["P0"] for r in reps]), "true": VAR_C},
         "residual_variance": {**_ms([r["R0"] for r in reps]), "true": VAR_E},
         "posterior_median": {
             "direct_variance": _ms([r["G0_median"][0][0] for r in reps]),
             "maternal_variance": _ms([r["G0_median"][1][1] for r in reps]),
             "direct_maternal_covariance": _ms([r["G0_median"][0][1] for r in reps]),
             "maternal_pe_variance": _ms([r["P0_median"] for r in reps]),
             "residual_variance": _ms([r["R0_median"] for r in reps])},
         "interval_coverage95": {
             "direct_variance": _ms([r["G0_cover95"][0][0] for r in reps]),
             "maternal_variance": _ms([r["G0_cover95"][1][1] for r in reps]),
             "direct_maternal_covariance": _ms([r["G0_cover95"][0][1] for r in reps])},
         "max_rhat_median": float(np.median([r["max_rhat"] for r in reps])),
         "wall_s_per_replicate": _ms([r["wall_s"] for r in reps])}
    for eff in ("direct", "maternal"):
        s[eff] = {k: _ms([r[eff][k] for r in reps]) for k in reps[0][eff]}
    doc = {"study": "Bayesian maternal animal model (direct + maternal genetic + maternal pe)",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "iterations": ITER, "prior": PRIOR, "true": {"G0": G0.tolist(), "var_c": VAR_C, "var_e": VAR_E},
           "wall_seconds": time.time() - t0, "summary": s, "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
