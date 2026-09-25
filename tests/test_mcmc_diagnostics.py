"""Diagnostics against known behaviour: iid draws, AR(1) chains with a known
integrated autocorrelation time, and chains stuck in different places."""

import numpy as np
import pytest

from abp.solvers.mcmc_diagnostics import bulk_ess, ess, mcse_mean, rhat, summarize, tail_ess


def _ar1(phi, chains, n, seed):
    rng = np.random.default_rng(seed)
    x = np.empty((chains, n))
    x[:, 0] = rng.normal(size=chains) / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + rng.normal(size=chains)
    return x


def test_iid_draws():
    x = np.random.default_rng(0).normal(size=(4, 2000))
    assert rhat(x) < 1.01
    assert ess(x) == pytest.approx(8000, rel=0.1)
    assert bulk_ess(x) == pytest.approx(8000, rel=0.1)
    assert tail_ess(x) == pytest.approx(8000, rel=0.15)
    assert mcse_mean(x) == pytest.approx(1 / np.sqrt(8000), rel=0.1)


def test_ar1_effective_sample_size():
    """For AR(1), tau = (1 + phi) / (1 - phi); phi = 0.9 -> ESS = N / 19."""
    x = _ar1(0.9, 4, 20000, 1)
    expected = x.size * (1 - 0.9) / (1 + 0.9)
    assert ess(x) == pytest.approx(expected, rel=0.15)
    assert bulk_ess(x) == pytest.approx(expected, rel=0.2)
    assert rhat(x) < 1.01


def test_nonmixing_chains_are_flagged():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 1000)) + np.array([[0.0], [0.0], [0.0], [1.0]])
    assert rhat(x) > 1.05
    # scale difference only: bulk R-hat misses it, folded R-hat catches it
    y = rng.normal(size=(4, 1000)) * np.array([[1.0], [1.0], [1.0], [3.0]])
    assert rhat(y) > 1.05


def test_summary_fields():
    s = summarize(np.random.default_rng(3).normal(2.0, 1.0, size=(4, 500)))
    assert set(s) >= {"mean", "sd", "q05", "q95", "rhat", "ess_bulk", "ess_tail", "mcse_mean"}
    assert s["mean"] == pytest.approx(2.0, abs=0.1)


def _det_chains(m, n, phi, shift=0.0):
    """Deterministic (RNG-free) chains: Weyl sequence plus a sine, AR(1)-filtered."""
    t = np.arange(n)
    x = np.zeros((m, n))
    for c in range(m):
        e = ((t * 0.6180339887498949 + (c + 1) * 0.41421356237309515) % 1.0) - 0.5
        e = e + 0.3 * np.sin(1.7 * t + c)
        v = 0.0
        for k in range(n):
            v = phi * v + e[k]
            x[c, k] = v
        x[c] += shift * c
    return x


# Reference values from ArviZ 0.23.4 (independent implementation, Apache-2.0; not an
# ABP dependency): az.rhat(method="rank"), az.ess(method="bulk"|"tail"),
# az.mcse(method="mean"). See benchmarks/mcmc_diagnostics_crosscheck.py.
ARVIZ_REFERENCE = [
    ((4, 400, 0.0, 0.0), (0.9978807438844917, 3811.716474366458, 1806.2889589250117,
                          0.0054700793274743194)),
    ((4, 401, 0.7, 0.0), (0.9977880860052293, 1507.7710457404758, 1557.687953082816,
                          0.007693585283813083)),
    ((3, 300, 0.5, 0.4), (1.4985641145868853, 6.7914893422866, 115.27984174196739,
                          0.1754562666114481)),
]


@pytest.mark.parametrize("args, ref", ARVIZ_REFERENCE)
def test_agreement_with_arviz_reference_values(args, ref):
    x = _det_chains(*args)
    got = (rhat(x), bulk_ess(x), tail_ess(x), mcse_mean(x))
    np.testing.assert_allclose(got, ref, rtol=1e-10)
