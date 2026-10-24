"""Tests for MCMC convergence diagnostics (R-hat, ESS)."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.model.diagnostics import gelman_rubin_rhat, effective_sample_size, ess_per_chain


def test_rhat_near_one_for_converged_chains():
    rng = np.random.default_rng(0)
    chains = rng.normal(0, 1, size=(4, 2000))  # same target, independent
    r = gelman_rubin_rhat(chains)
    assert np.isscalar(r) or r.ndim == 0
    assert abs(float(r) - 1.0) < 0.05


def test_rhat_large_for_divergent_chains():
    rng = np.random.default_rng(1)
    # Four chains centred at very different means -> not converged.
    chains = np.stack([rng.normal(mu, 0.1, 500) for mu in (-5, -1, 1, 5)])
    r = gelman_rubin_rhat(chains)
    assert float(r) > 1.5


def test_rhat_multiparam_shape_and_degenerate():
    rng = np.random.default_rng(2)
    chains = rng.normal(0, 1, size=(3, 400, 5))
    r = gelman_rubin_rhat(chains)
    assert r.shape == (5,)
    # A constant parameter (zero within-variance) returns 1.0, not nan/inf.
    const = np.ones((3, 400, 1))
    assert float(gelman_rubin_rhat(const)[0]) == 1.0


def test_rhat_validation():
    with pytest.raises(ValueError):
        gelman_rubin_rhat(np.zeros((1, 100)))   # need >= 2 chains
    with pytest.raises(ValueError):
        gelman_rubin_rhat(np.zeros((3, 1)))      # need length >= 2


def test_ess_bounds_and_ordering():
    rng = np.random.default_rng(3)
    n = 4000
    white = rng.normal(0, 1, n)
    # AR(1) with strong autocorrelation -> much lower ESS.
    phi = 0.9
    ar = np.empty(n); ar[0] = rng.normal()
    eps = rng.normal(0, 1, n)
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + eps[t]
    ess_w = effective_sample_size(white)
    ess_a = effective_sample_size(ar)
    assert 1.0 <= ess_a <= ess_w <= n
    assert ess_w > 0.5 * n          # white noise ~ n
    assert ess_a < 0.25 * n         # autocorrelated << n


def test_ess_constant_chain():
    assert effective_sample_size(np.full(100, 3.0)) == 1.0
    assert effective_sample_size(np.array([2.0])) == 1.0


def test_ess_per_chain_shapes():
    rng = np.random.default_rng(4)
    assert ess_per_chain(rng.normal(0, 1, (3, 500))).shape == (3,)
    assert ess_per_chain(rng.normal(0, 1, (3, 500, 4))).shape == (3, 4)


def test_diagnostics_on_real_mediation_fit():
    """R-hat/ESS behave sensibly on an ACTUAL posterior (not just synthetic
    arrays): 4 independent Gibbs chains on a small cell should agree (R-hat~1)
    with non-trivial ESS."""
    from src.simulate.generator import simulate_dataset
    from src.model.mediation import KineBalanceModel

    data = simulate_dataset(n=48, n_balances=10, sparsity=0.2, outcomes="F",
                            batch_strength="none", missing="none", seed=11)
    chains = []
    for c in range(4):
        m = KineBalanceModel(n_iter=800, burn=400, thin=2, ridge_sd=10.0, seed=200 + c)
        m.fit(np.asarray(data["X"], float), np.asarray(data["B"], float), data["Y"],
              C=np.asarray(data["C"], float), batch=np.asarray(data["batch"]))
        chains.append(m.samples.mediator["aX"])      # (S, K)
    stacked = np.stack(chains)                        # (4, S, K)
    rhat = gelman_rubin_rhat(stacked)                 # (K,)
    ess = ess_per_chain(stacked)                      # (4, K)
    assert np.all(np.isfinite(rhat))
    assert float(np.max(rhat)) < 1.3                  # independent chains agree
    assert float(np.min(ess)) > 20.0                  # mixing is non-degenerate


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))


def test_ess_matches_ar1_theory():
    """Geyer's initial monotone positive sequence estimate on a long AR(1) chain is
    close to the analytic value n (1 - phi) / (1 + phi)."""
    rng = np.random.default_rng(7)
    n, phi = 200_000, 0.8
    eps = rng.normal(0, 1, n)
    ar = np.empty(n); ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + eps[t]
    ess = effective_sample_size(ar)
    theory = n * (1 - phi) / (1 + phi)
    assert abs(ess / theory - 1.0) < 0.08


def test_rhat_infinite_for_chains_stuck_at_different_constants():
    # Binary indicators: two chains always 0, two always 1 -> no within-chain
    # variance but between-chain disagreement -> not converged.
    chains = np.stack([np.zeros(300), np.zeros(300), np.ones(300), np.ones(300)])
    assert np.isinf(gelman_rubin_rhat(chains))
