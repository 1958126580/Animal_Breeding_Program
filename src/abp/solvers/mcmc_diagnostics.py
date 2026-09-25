"""MCMC convergence diagnostics (module M20).

Implements the rank-normalized split-R-hat, bulk-ESS and tail-ESS of
Vehtari, Gelman, Simpson, Carpenter & Buerkner (2021, Bayesian Analysis
16:667-718), with the autocorrelation-based effective sample size and Geyer's
initial monotone sequence estimator as used in Stan.  The same functions are
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
    """Effective sample size of ``x`` (chains, draws) by Geyer's initial
    monotone sequence on the multi-chain autocorrelation (Stan's estimator)."""
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    if n < 4:
        return float("nan")
    if np.all(x == x.flat[0]):
        return float("nan")
    acov = _autocov(x)
    chain_mean = x.mean(axis=1)
    W = (acov[:, 0] * n / (n - 1)).mean()
    var_plus = W * (n - 1) / n + (chain_mean.var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sums of adjacent pairs while positive, made monotone
    t = 0
    pair_sums = []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        pair_sums.append(p)
        t += 2
    pair_sums = np.minimum.accumulate(np.array(pair_sums)) if pair_sums else np.array([1.0])
    tau = -1.0 + 2.0 * pair_sums.sum()
    tau = max(tau, 1.0 / np.log10(m * n))  # Stan's safeguard against tau -> 0
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
    """Monte-Carlo standard error of the posterior mean."""
    s = _split(x)
    e = ess(s)
    return float(s.std(ddof=1) / np.sqrt(e)) if e and np.isfinite(e) else float("nan")


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
