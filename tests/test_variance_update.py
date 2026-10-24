"""The variance and coefficient full conditionals, checked against an independent sampler.

Second review round: the exact enumeration pins the variances, so it does not test the
conjugate variance updates. Here every indicator is forced on (force_include), which
turns each stage into a Bayesian linear regression with independent priors,
    coefficients ~ N(0, v) (fixed slab, not scaled by the residual variance),
    nuisance block ~ N(0, 1/ridge_prec),   sigma^2 ~ InvGamma(0.5, 0.5),
on the standardized data the production sampler uses. An independent BLOCK Gibbs
sampler for that posterior (joint Gaussian draw of all coefficients, then the
inverse-gamma draw of sigma^2) must give the same posterior moments as the production
sampler's single-site updates: a wrong shape/rate, or a slab that silently scaled with
sigma^2, would shift the variance and the coefficient spread.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.model.mediation_v2 import SpikeSlabMediation

RIDGE_PREC = 1.0 / 10.0 ** 2          # nuisance prior precision used by the sampler
SLAB_V = 2.0 ** 2                     # default slab_scale = 2


def _reference_gibbs(Z, y, prior_prec, n_iter=24000, burn=4000, seed=0):
    """Block Gibbs for y = Z theta + e, theta ~ N(0, diag(1/prior_prec)),
    e ~ N(0, s2), s2 ~ InvGamma(0.5, 0.5)."""
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    ZtZ, Zty = Z.T @ Z, Z.T @ y
    s2 = 1.0
    th_draws, s2_draws = [], []
    for it in range(n_iter):
        Q = ZtZ / s2 + np.diag(prior_prec)
        L = np.linalg.cholesky(Q)
        mean = np.linalg.solve(L.T, np.linalg.solve(L, Zty / s2))
        theta = mean + np.linalg.solve(L.T, rng.standard_normal(p))
        rss = float(np.sum((y - Z @ theta) ** 2))
        s2 = 1.0 / rng.gamma(0.5 + n / 2.0, 1.0 / (0.5 + rss / 2.0))
        if it >= burn:
            th_draws.append(theta)
            s2_draws.append(s2)
    return np.asarray(th_draws), np.asarray(s2_draws)


def _data(seed=3, n=150, K=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K))
    B[:, 0] += 0.8 * X
    y = 0.7 * B[:, 0] - 0.4 * B[:, 1] + 0.3 * X + rng.normal(0.0, 1.2, n)
    return X, B, y


def test_stage2_variance_and_coefficients_match_reference():
    X, B, y = _data()
    s = SpikeSlabMediation(n_iter=24000, burn=4000, thin=1, seed=4,
                           force_include=True).fit(X, B, {"Y": y})
    Bs = (B - B.mean(0)) / B.std(0)
    ys = (y - y.mean()) / y.std()
    Z = np.column_stack([np.ones(len(y)), X, Bs])             # [1, X | coordinates]
    prior = np.r_[np.full(2, RIDGE_PREC), np.full(B.shape[1], 1.0 / SLAB_V)]
    th, s2 = _reference_gibbs(Z, ys, prior)
    beta = s.outcomes["Y"]["beta"]
    sigma2 = (s.outcomes["Y"]["sigma"] / y.std()) ** 2        # back to the standardized scale
    assert abs(sigma2.mean() - s2.mean()) < 0.01, (sigma2.mean(), s2.mean())
    assert abs(sigma2.std() - s2.std()) < 0.01, (sigma2.std(), s2.std())
    assert np.allclose(beta.mean(0), th[:, 2:].mean(0), atol=0.01)
    assert np.allclose(beta.std(0), th[:, 2:].std(0), atol=0.01)


def test_stage1_variance_enters_the_exposure_coefficient_correctly():
    X, B, y = _data()
    s = SpikeSlabMediation(n_iter=24000, burn=4000, thin=1, seed=5,
                           force_include=True).fit(X, B, {"Y": y})
    Bs = (B - B.mean(0)) / B.std(0)
    alpha = s.mediator["alpha"]
    for k in range(B.shape[1]):
        Z = np.column_stack([np.ones(len(X)), X])            # [1 | exposure]
        prior = np.array([RIDGE_PREC, 1.0 / SLAB_V])
        th, _ = _reference_gibbs(Z, Bs[:, k], prior, seed=10 + k)
        assert abs(alpha[:, k].mean() - th[:, 1].mean()) < 0.01
        assert abs(alpha[:, k].std() - th[:, 1].std()) < 0.005
