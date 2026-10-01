#!/usr/bin/env python3
"""Finding F16 follow-up: Bayesian posterior PEV vs REML + Kackar-Harville (round 9).

Round 8 found that with full-rank multi-trait REML on 1,000 animals the EBVs of the
low-heritability traits stay optimistic even with the first-order Kackar-Harville PEV
(MSE/PEV 1.14-1.17).  This study fits the same simulated data (the ``t2_full`` and
``t3_full`` scenarios of ``rank_selection_study.py``: random sex-consistent pedigree,
one record per animal, traits after the first missing at random for 30% of records,
one mean per trait) with

* ``reml_plugin``  full-rank AI-REML, BLUP with the estimates plugged in;
* ``reml_kh``      the same EBVs, PEV + Kackar-Harville correction;
* ``bayes``        Bayesian multi-trait linear model (``mt_threshold_gibbs`` with no
                   categorical trait), flat priors on G0 and R0, EBV = posterior mean,
                   PEV = posterior variance (integrates over the (co)variances);
* ``true_parameters``  BLUP with the true G0 and R0 (reference).

Scores per trait over all animals: MSE/mean PEV, coverage of nominal 95% intervals,
realized accuracy.  Chains: 4 x ``--iterations`` (burn-in 1/4, thin 4); whether the
R-hat/ESS criteria passed is recorded and the results are scored either way.  Truth is
used only for scoring.  Means +- Monte-Carlo SE over replicates.

Usage: python benchmarks/bayes_vs_reml_study.py [--scenarios t2_full t3_full]
       [--replicates 50] [--workers 4] [--animals 1000] [--iterations 4000]
       [--out docs/validation/bayes_vs_reml_study.json]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from rank_selection_study import SCENARIOS, simulate  # noqa: E402

from abp.errors import ABPError  # noqa: E402
from abp.solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs  # noqa: E402
from abp.solvers.multitrait import build_and_solve  # noqa: E402
from abp.solvers.multitrait_reml import mt_reml_fit  # noqa: E402
from abp.solvers.vc_uncertainty import kackar_harville_delta_multitrait  # noqa: E402

N_ANIMALS = 1000
ITER = 4000
SCENARIO = "t2_full"                  # set by main(); inherited by forked workers


def _scores(U, ebv, pev):
    out = {}
    for j in range(U.shape[1]):
        err = U[:, j] - ebv[:, j]
        out[f"trait{j + 1}"] = {"pev_ratio": float(np.mean(err ** 2) / np.mean(pev[:, j])),
                                "coverage95": float(np.mean(np.abs(err)
                                                            <= 1.96 * np.sqrt(pev[:, j]))),
                                "realized_accuracy": float(np.corrcoef(U[:, j],
                                                                       ebv[:, j])[0, 1])}
    return out


def _diag(blocks):
    return np.einsum("ijj->ij", blocks)


def replicate(seed: int) -> dict:
    L, R0 = SCENARIOS[SCENARIO]
    G0 = L @ L.T
    ped, data, U = simulate(seed, N_ANIMALS, L, R0)
    ainv, kd = ped.ainv(), 1.0 + ped.inbreeding()
    out = {"seed": seed}
    t0 = time.time()
    try:
        fit = mt_reml_fit(data, ainv, ped.logdet_a(), {"tol": 1e-8, "max_iter": 200})
        res = build_and_solve(data, ainv, kd, fit.G0, fit.R0, method="dense")
        out["reml_G0"] = np.asarray(fit.G0).tolist()
        out["reml_plugin"] = _scores(U, res.ebv, _diag(res.pev_blocks))
        if fit.cov is not None:
            delta = kackar_harville_delta_multitrait(data, ainv, kd, fit.G0, fit.R0, fit.cov,
                                                     method="dense")
            out["reml_kh"] = _scores(U, res.ebv, _diag(res.pev_blocks) + _diag(delta))
        out["reml_wall_s"] = time.time() - t0
    except ABPError as exc:
        out["reml_status"] = exc.code
    t0 = time.time()
    g = mt_threshold_gibbs(data.Y, None, data.X_per_trait, np.asarray(data.animal_col), ainv,
                           MTThresholdGibbsConfig(chains=4, iterations=ITER,
                                                  burn_in=ITER // 4, thin=4,
                                                  max_iterations=ITER, seed=seed))
    out["bayes_converged"] = g.converged
    out["bayes_wall_s"] = time.time() - t0
    out["bayes_G0"] = g.G0["mean"]
    out["bayes"] = _scores(U, g.ebv, g.pev)
    res_t = build_and_solve(data, ainv, kd, G0, R0, method="dense")
    out["true_parameters"] = _scores(U, res_t.ebv, _diag(res_t.pev_blocks))
    return out


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()),
            "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
            "n": int(v.size)}


def summarize(reps, t):
    s = {"bayes_converged": sum(r["bayes_converged"] for r in reps),
         "reml_failed": sum("reml_status" in r for r in reps)}
    for sc in ("reml_plugin", "reml_kh", "bayes", "true_parameters"):
        rr = [r[sc] for r in reps if sc in r]
        s[sc] = {f"trait{j + 1}": {k: _ms([x[f"trait{j + 1}"][k] for x in rr])
                                   for k in ("pev_ratio", "coverage95", "realized_accuracy")}
                 for j in range(t)}
    for est in ("reml_G0", "bayes_G0"):
        Gs = np.array([r[est] for r in reps if est in r])
        s[est] = {"mean": Gs.mean(axis=0).tolist(),
                  "mc_se": (Gs.std(axis=0, ddof=1) / np.sqrt(len(Gs))).tolist()}
    return s


def main():
    global N_ANIMALS, ITER, SCENARIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=("t2_full", "t3_full"),
                    default=["t2_full", "t3_full"])
    ap.add_argument("--replicates", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=1000)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "bayes_vs_reml_study.json"))
    a = ap.parse_args()
    N_ANIMALS, ITER = a.animals, a.iterations
    doc = {"study": "Bayesian posterior PEV vs REML + Kackar-Harville (finding F16)",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "iterations": ITER, "scenarios": {}}
    for name in a.scenarios:
        SCENARIO = name
        L, R0 = SCENARIOS[name]
        t0 = time.time()
        seeds = list(range(1, a.replicates + 1))
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as pool:
                reps = list(pool.imap(replicate, seeds))
        else:
            reps = [replicate(s) for s in seeds]
        summ = summarize(reps, L.shape[0])
        doc["scenarios"][name] = {"true": {"G0": (L @ L.T).tolist(), "R0": R0.tolist()},
                                  "wall_seconds": time.time() - t0, "summary": summ,
                                  "replicate_results": reps}
        print(name, f"bayes converged {summ['bayes_converged']}/{len(reps)}, "
                    f"REML failed {summ['reml_failed']}", flush=True)
        for sc in ("reml_plugin", "reml_kh", "bayes", "true_parameters"):
            print(f"  {sc:>15}", " ".join(
                f"{k}: MSE/PEV {v['pev_ratio']['mean']:.3f}+-{v['pev_ratio']['mc_se']:.3f} "
                f"cov {v['coverage95']['mean']:.3f} acc {v['realized_accuracy']['mean']:.3f}"
                for k, v in summ[sc].items()), flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
