#!/usr/bin/env python3
"""Prior simulation-based calibration (SBC) of ABP's Bayesian marker samplers.

SBC (Talts, Betancourt, Simpson, Vehtari & Gelman 2018) checks that a
sampler computes the posterior of the model it claims to implement: draw
parameters from the prior, simulate data from the model, run the sampler,
and record the rank of each true value among the posterior draws.  If the
computation is correct, every rank is uniformly distributed.

Design (all hyper-parameters fixed and passed explicitly to the sampler, so
the simulation prior and the sampler prior are identical):

* 150 records, 40 markers, centred dosages of a fixed genotype matrix;
* ``sigma_e^2 ~ nu_e S_e^2 / chi^2(nu_e)`` with ``nu_e = 5``, ``S_e^2 = 0.6``;
* marker prior per method with ``nu = 5``, ``S^2 = 0.02``: BRR (common
  variance), BayesA (marker-specific variances), BayesB (``pi0 = 0.8``,
  marker-specific variances), BayesC (``pi0 = 0.8``), BayesCpi
  (``pi0 ~ Beta(1, 1)``) and BayesR (``gamma = (0, 0.01, 0.1, 1)``,
  mixture weights ``~ Dirichlet(1, 1, 1, 1)``);
* ``y = 10 + W beta + e``.  The sampler's prior on the intercept is flat
  (improper), so the intercept cannot be drawn from it; the posterior of
  every other quantity depends on ``y`` only through the residuals after
  projecting out the intercept, so any fixed value is valid for SBC of
  those quantities.

Monitored quantities: ``sigma_e2``, the marker-variance parameter
(``sigma_b2`` or the mean of the marker-specific variances), the in-sample
genetic variance ``var(W beta)``, the first marker effect, the genomic
value of the first animal and, where it is estimated, ``pi0``.  Each replicate runs 2 chains x 2,200
iterations (burn-in 200, thin 20: 200 draws).  Ties (point masses of
BayesC) are broken uniformly at random.  Uniformity is tested with a
chi-square test on 10 bins per quantity; with 32 tests a p-value below 0.001
is treated as a failure (about 3% family-wise false alarm rate).  SBC has
limited power: passing does not prove correctness, failing proves a problem.

Usage: python benchmarks/sbc_bayes.py [--replicates 300] [--out docs/validation/sbc_bayes.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.solvers.bayes import BayesConfig, BayesPriors, run_bayes  # noqa: E402

N, M = 150, 40
NU, S2, NU_E, S2_E = 5.0, 0.02, 5.0, 0.6
PI0_FIXED = 0.8                      # BayesB and BayesC
GAMMA_R = (0.0, 0.01, 0.1, 1.0)      # BayesR component variance ratios
METHODS = ("BRR", "BayesA", "BayesB", "BayesC", "BayesCpi", "BayesR")
BASE_QUANTITIES = ("sigma_e2", "marker_variance", "genetic_variance", "beta_0", "gebv_0")


def quantities(method: str) -> tuple[str, ...]:
    return BASE_QUANTITIES + (("pi0",) if method in ("BayesCpi", "BayesR") else ())
N_BINS = 10


def design(seed: int = 20260925) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, M)
    geno = rng.binomial(2, p, size=(N, M)).astype(float)
    W = geno - geno.mean(axis=0)
    assert np.all(W.std(axis=0) > 0)
    return W


def priors(method: str) -> BayesPriors:
    if method == "BayesR":
        return BayesPriors(nu=NU, s2=S2, nu_e=NU_E, s2_e=S2_E, pi0=0.25, gamma=GAMMA_R,
                           dirichlet=(1.0,) * len(GAMMA_R),
                           derivation={"source": "fixed SBC hyper-parameters"})
    pi0 = {"BayesB": PI0_FIXED, "BayesC": PI0_FIXED, "BayesCpi": 0.5}.get(method, 0.0)
    return BayesPriors(nu=NU, s2=S2, nu_e=NU_E, s2_e=S2_E, pi0=pi0, gamma=(0.0, 1.0),
                       dirichlet=(1.0, 1.0), derivation={"source": "fixed SBC hyper-parameters"})


def draw_truth(method: str, W: np.ndarray, rng) -> dict:
    sigma_e2 = NU_E * S2_E / rng.chisquare(NU_E)
    pi0 = None
    if method in ("BayesA", "BayesB"):
        v = NU * S2 / rng.chisquare(NU, size=M)
        beta = rng.normal(0.0, np.sqrt(v))
        if method == "BayesB":
            beta = beta * (rng.random(M) >= PI0_FIXED)
        marker_variance = float(v.mean())
    elif method == "BayesR":
        sigma_b2 = NU * S2 / rng.chisquare(NU)
        pi = rng.dirichlet(np.ones(len(GAMMA_R)))
        comp = rng.choice(len(GAMMA_R), size=M, p=pi)
        beta = rng.normal(0.0, 1.0, size=M) * np.sqrt(np.array(GAMMA_R)[comp] * sigma_b2)
        marker_variance, pi0 = float(sigma_b2), float(pi[0])
    else:
        sigma_b2 = NU * S2 / rng.chisquare(NU)
        beta = rng.normal(0.0, np.sqrt(sigma_b2), size=M)
        if method == "BayesC":
            beta = beta * (rng.random(M) >= PI0_FIXED)
        elif method == "BayesCpi":
            pi0 = float(rng.beta(1.0, 1.0))
            beta = beta * (rng.random(M) >= pi0)
        marker_variance = float(sigma_b2)
    g = W @ beta
    y = 10.0 + g + rng.normal(0.0, np.sqrt(sigma_e2), size=N)
    return {"y": y, "sigma_e2": float(sigma_e2), "marker_variance": marker_variance,
            "genetic_variance": float(np.var(g)), "beta_0": float(beta[0]),
            "gebv_0": float(g[0]), "pi0": pi0}


def rank(draws: np.ndarray, truth: float, rng) -> int:
    d = np.ravel(draws)
    less = int(np.sum(d < truth))
    ties = int(np.sum(d == truth))
    return less + int(rng.integers(0, ties + 1))


def run_method(method: str, W: np.ndarray, replicates: int, seed: int) -> dict:
    ss = np.random.SeedSequence([seed, METHODS.index(method)])
    qs = quantities(method)
    ranks = {q: [] for q in qs}
    n_draws = None
    t0 = time.perf_counter()
    for r, child in enumerate(ss.spawn(replicates)):
        rng = np.random.default_rng(child)
        truth = draw_truth(method, W, rng)
        cfg = BayesConfig(method=method, chains=2, iterations=2200, burn_in=200, thin=20,
                          seed=int(rng.integers(2**31)), max_iterations=2200,
                          priors=priors(method), return_draws=True)
        res = run_bayes(truth["y"], np.ones((N, 1)), W, W, cfg, 1.0)
        mv = res.traces["mean_marker_variance" if method in ("BayesA", "BayesB") else "sigma_b2"]
        draws = {"sigma_e2": res.traces["sigma_e2"], "marker_variance": mv,
                 "genetic_variance": res.traces["genetic_variance"],
                 "beta_0": res.beta_draws[:, :, 0], "gebv_0": res.gebv_draws[:, :, 0],
                 "pi0": res.traces.get("pi0")}
        n_draws = int(np.size(draws["sigma_e2"]))
        for q in qs:
            ranks[q].append(rank(draws[q], truth[q], rng))
    edges = np.linspace(0, n_draws + 1, N_BINS + 1)
    ints_per_bin = np.histogram(np.arange(n_draws + 1), bins=edges)[0]
    expected = ints_per_bin / (n_draws + 1) * replicates
    out = {}
    for q in qs:
        obs = np.histogram(ranks[q], bins=edges)[0]
        stat, pval = chisquare(obs, expected)
        out[q] = {"histogram": obs.tolist(), "expected": expected.tolist(),
                  "chi2": float(stat), "p_value": float(pval), "passed": bool(pval >= 0.001),
                  "mean_rank_over_draws": float(np.mean(ranks[q]) / n_draws)}
    return {"draws_per_replicate": n_draws, "wall_seconds": time.perf_counter() - t0,
            "quantities": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "sbc_bayes.json"))
    args = ap.parse_args()
    W = design()
    results = {}
    for method in METHODS:
        results[method] = run_method(method, W, args.replicates, args.seed)
        q = results[method]["quantities"]
        print(f"{method:7s} ({results[method]['wall_seconds']:.0f} s): " + "  ".join(
            f"{k} p={v['p_value']:.3f}{'' if v['passed'] else ' FAIL'}" for k, v in q.items()),
            flush=True)
    all_passed = all(v["passed"] for r in results.values() for v in r["quantities"].values())
    doc = {"study": "prior simulation-based calibration of the Gibbs samplers",
           "reference": "Talts et al. 2018, arXiv:1804.06788",
           "replicates_per_method": args.replicates, "seed": args.seed,
           "design": {"records": N, "markers": M, "nu": NU, "S2": S2, "nu_e": NU_E,
                      "S2_e": S2_E, "pi0_BayesB_BayesC": PI0_FIXED, "pi0_BayesCpi": "Beta(1, 1)",
                      "gamma_BayesR": list(GAMMA_R), "dirichlet_BayesR": [1.0] * 4, "chains": 2,
                      "iterations": 2200, "burn_in": 200, "thin": 20, "bins": N_BINS},
           "criterion": "chi-square uniformity test per quantity; failure if p < 0.001",
           "all_passed": all_passed, "results": results}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("all passed:", all_passed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
