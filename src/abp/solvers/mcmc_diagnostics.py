"""MCMC convergence diagnostics (module M20).

Implements the rank-normalized split-R-hat, bulk-ESS and tail-ESS of
Vehtari, Gelman, Simpson, Carpenter & Buerkner (2021, Bayesian Analysis
16:667-718), with the autocorrelation-based effective sample size and Geyer's
initial positive and monotone sequence estimator as used in Stan.  ArviZ
(Apache-2.0) serves only as an external test oracle and is not a dependency.  The same functions are
applied to every monitored quantity (variance components, mixture weights,
genetic variance and, where stored, individual GEBVs) - diagnosing only the
variance components is not enough.

Default acceptance (project spec M20, starting targets, not proofs of
correctness): at least 4 chains, R-hat < 1.01, bulk-ESS >= 400 and
tail-ESS >= 400.  These are Gibbs-sampler diagnostics; HMC-specific checks
(divergences, energy) do not apply.

Arrays are ``(chains, draws)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def _split(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected (chains, draws)")
    n = x.shape[1] // 2
    return np.vstack([x[:, :n], x[:, x.shape[1] - n:]])


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Pooled fractional ranks -> normal scores (Blom offset 3/8)."""
    from scipy.stats import rankdata
    flat = x.ravel()
    r = rankdata(flat, method="average").reshape(x.shape)
    return ndtri((r - 0.375) / (flat.size + 0.25))


def _rhat_basic(x: np.ndarray) -> float:
    m, n = x.shape
    if n < 2:
        return float("nan")
    means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1)
    if W <= 0:
        return float("nan") if B > 0 else 1.0
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def rhat(x: np.ndarray) -> float:
    """Rank-normalized split-R-hat: max of bulk and folded versions."""
    s = _split(x)
    if np.all(s == s.flat[0]):
        return 1.0
    bulk = _rhat_basic(_rank_normalize(s))
    folded = _rhat_basic(_rank_normalize(np.abs(s - np.median(s))))
    return float(max(bulk, folded))


def _autocov(x: np.ndarray) -> np.ndarray:
    """Per-chain autocovariance for all lags via FFT (biased estimator)."""
    m, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, n=size, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=size, axis=1)[:, :n] / n
    return ac


def ess(x: np.ndarray) -> float:
    """Effective sample size of ``x`` (chains, draws), Stan's estimator.

    Multi-chain autocorrelations ``rho_t = 1 - (W - mean_chains acov_t) / var+``
    (``W`` the mean within-chain variance, ``var+`` the pooled variance
    estimate) are truncated by Geyer's *initial positive sequence*: pairs
    ``(rho_{2k}, rho_{2k+1})`` are kept while their sum is positive (the pair
    at lags 0 and 1 always).  The last even autocorrelation is kept if it is
    positive, the kept pairs are made monotonically non-increasing (*initial
    monotone sequence*), and ``tau = -1 + 2 sum_{t <= T} rho_t + rho_{T+1}``,
    ``ESS = m n / max(tau, 1/log10(m n))``.  This follows the reference
    algorithm of Vehtari et al. (2021) as implemented in Stan; agreement
    with the independent ArviZ implementation is recorded in
    ``docs/validation/mcmc_diagnostics_crosscheck.json``.
    """
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    if n < 4:
        return float("nan")
    if np.all(x == x.flat[0]):
        return float("nan")
    acov = _autocov(x)
    W = float(np.mean(acov[:, 0])) * n / (n - 1.0)
    var_plus = W * (n - 1.0) / n
    if m > 1:
        var_plus += float(np.var(x.mean(axis=1), ddof=1))
    if var_plus <= 0:
        return float("nan")
    mean_acov = acov.mean(axis=0)

    def rho_at(t):
        return 1.0 - (W - mean_acov[t]) / var_plus

    rho = np.zeros(n)
    rho[0], rho[1] = 1.0, rho_at(1)
    even, odd = rho[0], rho[1]
    t = 1
    # initial positive sequence over the pairs (t+1, t+2) = (2, 3), (4, 5), ...
    while t < n - 3 and even + odd > 0.0:
        even, odd = rho_at(t + 1), rho_at(t + 2)
        if even + odd >= 0.0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    last = t - 2                        # index of the last kept odd lag
    if even > 0.0:
        rho[last + 1] = even            # keep the last positive even autocorrelation
    # initial monotone sequence: no pair may exceed the pair before it
    t = 1
    while t <= last - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = 0.5 * (rho[t - 1] + rho[t])
        t += 2
    tau = -1.0 + 2.0 * float(np.sum(rho[:last + 1])) + float(np.sum(rho[last + 1:last + 2]))
    tau = max(tau, 1.0 / np.log10(m * n))   # Stan's safeguard against tau -> 0
    return float(m * n / tau)


def bulk_ess(x: np.ndarray) -> float:
    return ess(_rank_normalize(_split(x)))


def tail_ess(x: np.ndarray) -> float:
    s = _split(x)
    q05, q95 = np.quantile(s, [0.05, 0.95])
    e1 = ess((s <= q05).astype(float))
    e2 = ess((s <= q95).astype(float))
    return float(np.nanmin([e1, e2]))


def mcse_mean(x: np.ndarray) -> float:
    """Monte-Carlo standard error of the posterior mean: ``sd / sqrt(ESS)``, with
    the SD of all draws and the ESS of the split chains."""
    x = np.asarray(x, dtype=np.float64)
    e = ess(_split(x))
    return float(x.std(ddof=1) / np.sqrt(e)) if e and np.isfinite(e) else float("nan")


def summarize(x: np.ndarray) -> dict:
    """Posterior summary and diagnostics of one scalar quantity."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    return {"mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
            "q05": float(np.quantile(flat, 0.05)), "q50": float(np.quantile(flat, 0.5)),
            "q95": float(np.quantile(flat, 0.95)), "rhat": rhat(x), "ess_bulk": bulk_ess(x),
            "ess_tail": tail_ess(x), "mcse_mean": mcse_mean(x)}


def passes(s: dict, rhat_max: float = 1.01, ess_min: float = 400.0) -> bool:
    return bool(np.isfinite(s["rhat"]) and s["rhat"] < rhat_max and s["ess_bulk"] >= ess_min
                and s["ess_tail"] >= ess_min)
