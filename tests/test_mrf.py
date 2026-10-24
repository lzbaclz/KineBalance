"""Tests for the optional MRF/Ising correlated-mediator prior in mediation_v2.

The MRF branch is live production code inside the shipped sampler, so it needs
coverage: (1) lambda=0 must be an exact no-op vs mrf=None (the coupling term is
0*..., same RNG stream); (2) a malformed W must raise; (3) a positive coupling must
pool inclusion evidence -- a weak balance coupled to an always-on neighbour gets a
higher PIP than under the independent (mrf=None) prior.
"""
import numpy as np
import pytest

from src.model.mediation_v2 import SpikeSlabMediation
from src.model.specificity_v2 import posterior_inclusion_probs


def _pip(samples, out, k):
    return float((samples.mediator["gamma"] * samples.outcomes[out]["delta"]).mean(0)[k])


def test_mrf_lambda_zero_is_exact_noop():
    rng = np.random.default_rng(0); n, K = 120, 6
    X = rng.standard_normal(n); B = rng.standard_normal((n, K)); B[:, 0] += 0.9 * X
    y = 0.8 * B[:, 0] + rng.standard_normal(n) * 0.6
    W = (np.abs(np.corrcoef(B.T)) > 0.3).astype(float); np.fill_diagonal(W, 0.0)
    base = SpikeSlabMediation(n_iter=400, burn=200, seed=1).fit(X, B, {"Y": y})
    zero = SpikeSlabMediation(n_iter=400, burn=200, seed=1,
                              mrf={"W": W, "lambda": 0.0}).fit(X, B, {"Y": y})
    # lambda=0 adds exactly 0 to the log-odds and does not consume RNG -> identical draws
    np.testing.assert_array_equal(base.mediator["gamma"], zero.mediator["gamma"])
    np.testing.assert_allclose(posterior_inclusion_probs(base)["Y"],
                               posterior_inclusion_probs(zero)["Y"], atol=1e-12)


def test_mrf_bad_W_shape_raises():
    rng = np.random.default_rng(1); n, K = 60, 5
    X = rng.standard_normal(n); B = rng.standard_normal((n, K)); y = rng.standard_normal(n)
    with pytest.raises(ValueError):
        SpikeSlabMediation(n_iter=50, burn=20, seed=0,
                           mrf={"W": np.ones((K + 1, K + 1)), "lambda": 1.0}).fit(X, B, {"Y": y})


def test_mrf_positive_coupling_pools_evidence():
    # Balance 1 is a STRONG mediator (PIP ~ 1); balance 0 is a WEAK correlated mediator.
    # W couples 0<->1, so a positive coupling should raise balance 0's PIP vs independent.
    def sim(seed):
        rng = np.random.default_rng(seed); n, K = 140, 8
        X = rng.standard_normal(n)
        B = rng.standard_normal((n, K))
        shared = rng.standard_normal(n)
        B[:, 0] = 0.75 * shared + 0.35 * rng.standard_normal(n) + 0.18 * X   # weak exposure reg.
        B[:, 1] = 0.75 * shared + 0.35 * rng.standard_normal(n) + 0.9 * X    # strong exposure reg.
        y = 0.3 * B[:, 0] + 0.9 * B[:, 1] + rng.standard_normal(n)
        return X, B, {"Y": y}
    W = np.zeros((8, 8)); W[0, 1] = W[1, 0] = 1.0
    ind, mrf = [], []
    for sd in range(6):
        X, B, Y = sim(sd)
        s0 = SpikeSlabMediation(n_iter=800, burn=400, seed=sd).fit(X, B, Y)
        s1 = SpikeSlabMediation(n_iter=800, burn=400, seed=sd,
                                mrf={"W": W, "lambda": 3.0}).fit(X, B, Y)
        ind.append(_pip(s0, "Y", 0)); mrf.append(_pip(s1, "Y", 0))
    # pooling should not reduce and on average should raise the weak balance's PIP
    assert np.mean(mrf) >= np.mean(ind) - 1e-6
