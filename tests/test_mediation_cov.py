"""Unit tests for the B4 residual-covariance (SUR) mediation sampler."""
import numpy as np

from src.model.mediation_cov import SpikeSlabMediationCov
from src.model.specificity_v2 import posterior_inclusion_probs


def _sim(seed, rho, n=200, K=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K)); B[:, 0] += 0.9 * X      # balance 0 exposure-regulated
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    E = rng.standard_normal((n, 2)) @ L.T
    Y1 = 0.8 * B[:, 0] + E[:, 0]; Y2 = 0.8 * B[:, 0] + E[:, 1]
    return X, B, {"Y1": Y1, "Y2": Y2}


def test_sur_recovers_residual_correlation_and_mediator():
    X, B, Y = _sim(0, rho=0.6)
    s = SpikeSlabMediationCov(n_iter=1200, burn=600, thin=2, seed=1).fit(X, B, Y)
    # residual correlation recovered (positive, in a sensible band around 0.6)
    assert 0.35 < s.meta["resid_corr_mean"] < 0.85
    # the true mediator (balance 0) is recovered on both outcomes
    pip = posterior_inclusion_probs(s)
    assert pip["Y1"][0] > 0.8 and pip["Y2"][0] > 0.8


def test_sur_zero_correlation_is_near_zero():
    X, B, Y = _sim(2, rho=0.0)
    s = SpikeSlabMediationCov(n_iter=1200, burn=600, thin=2, seed=3).fit(X, B, Y)
    assert abs(s.meta["resid_corr_mean"]) < 0.25


def _sim_R(seed, rho12, R=3, n=200, K=6):
    """R-outcome sim: balance 0 mediates every outcome; outcomes 1,2 share resid corr rho12."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K)); B[:, 0] += 0.9 * X
    S = np.eye(R)
    S[0, 1] = S[1, 0] = rho12
    L = np.linalg.cholesky(S)
    E = rng.standard_normal((n, R)) @ L.T
    Y = {f"Y{r+1}": 0.8 * B[:, 0] + E[:, r] for r in range(R)}
    return X, B, Y


def test_sur_R1_reduces_to_conference_sampler():
    # The docstring claims R=1 (single outcome) reduces to the conference sampler. Fit both on
    # identical data and check the selection posteriors agree (not byte-identical -- the SUR
    # sampler draws a 1x1 Sigma -- but the PIPs must track closely).
    from src.model.mediation_v2 import SpikeSlabMediation
    rng = np.random.default_rng(11)
    X = rng.standard_normal(200)
    B = rng.standard_normal((200, 6)); B[:, 0] += 0.9 * X
    Y = {"Y1": 0.8 * B[:, 0] + rng.standard_normal(200)}
    cov = SpikeSlabMediationCov(n_iter=1500, burn=750, thin=2, seed=5).fit(X, B, Y)
    conf = SpikeSlabMediation(n_iter=1500, burn=750, thin=2, seed=5).fit(X, B, Y)
    pc, pf = posterior_inclusion_probs(cov)["Y1"], posterior_inclusion_probs(conf)["Y1"]
    assert pc[0] > 0.8 and pf[0] > 0.8                       # both recover the true mediator
    assert np.max(np.abs(pc - pf)) < 0.15                    # and the whole PIP vector tracks
    assert abs(cov.meta["resid_corr_mean"]) < 1e-9           # no off-diagonal to estimate at R=1


def test_sur_R3_inverse_wishart_update_recovers():
    # Exercises the inverse-Wishart Sigma update for R>2 (not covered by the R=2 tests): the
    # true mediator is recovered on all three outcomes and the residual-correlation summary is
    # a finite, valid correlation.
    X, B, Y = _sim_R(7, rho12=0.6, R=3)
    s = SpikeSlabMediationCov(n_iter=1200, burn=600, thin=2, seed=9).fit(X, B, Y)
    pip = posterior_inclusion_probs(s)
    assert all(pip[o][0] > 0.8 for o in Y)                   # mediator recovered on all 3 outcomes
    rc = s.meta["resid_corr_mean"]
    assert np.isfinite(rc) and -1.0 <= rc <= 1.0


def _sim_endpoint_only(seed, n=200, K=6):
    """Balance 0 mediates both outcomes; balance 1 acts on Y1 WITHOUT exposure
    regulation (the case where the coupled and decoupled priors disagree); balance 2 is a
    weak mediator whose indicators have intermediate posterior probabilities."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K)); B[:, 0] += 0.9 * X; B[:, 2] += 0.25 * X
    Y1 = 0.8 * B[:, 0] + 0.6 * B[:, 1] + 0.18 * B[:, 2] + rng.standard_normal(n)
    Y2 = 0.8 * B[:, 0] + 0.18 * B[:, 2] + rng.standard_normal(n)
    return X, B, {"Y1": Y1, "Y2": Y2}


def test_sur_diagonal_sigma_matches_independent_sampler_under_both_priors():
    # With Sigma held diagonal under the conference model's InvGamma(1/2, 1/2) residual
    # prior, the SUR sampler targets exactly the independent-residual model, so under each
    # prior coupling its selection posterior must match SpikeSlabMediation's (same model,
    # different scan order and RNG stream: agreement up to Monte-Carlo error). The data
    # carry an endpoint-only coordinate, on which the two priors disagree, so the check
    # also verifies that the SUR sampler's decoupled branch is the decoupled prior.
    from src.model.mediation_v2 import SpikeSlabMediation
    from src.model.specificity_v2 import pip_gamma
    X, B, Y = _sim_endpoint_only(23)
    pips = {}
    for sub in (True, False):
        cov = SpikeSlabMediationCov(n_iter=12000, burn=2000, thin=2, seed=3, subsetting=sub,
                                    sigma_diag=True).fit(X, B, Y)
        ind = SpikeSlabMediation(n_iter=12000, burn=2000, thin=2, seed=4,
                                 subsetting=sub).fit(X, B, Y)
        pc, pi = posterior_inclusion_probs(cov), posterior_inclusion_probs(ind)
        for o in Y:
            assert np.max(np.abs(pc[o] - pi[o])) < 0.04, (sub, o, pc[o], pi[o])
        assert np.max(np.abs(pip_gamma(cov) - pip_gamma(ind))) < 0.04
        assert cov.meta["resid_corr_mean"] == 0.0              # diagonal by construction
        assert 0.05 < pc["Y2"][2] < 0.4                         # an intermediate PIP is compared
        pips[sub] = pc["Y1"][1]
    # the endpoint-only coordinate: pulled in by the coupling, not by the decoupled prior
    assert pips[True] > 0.5 and pips[False] < 0.1, pips


def test_sur_decoupled_full_sigma_recovers_mediator_not_endpoint_only():
    # Full inverse-Wishart Sigma under the decoupled prior: the mediator is found on both
    # outcomes, the endpoint-only coordinate is not called a mediator, and the residual
    # correlation is estimated.
    X, B, Y = _sim_endpoint_only(22)
    s = SpikeSlabMediationCov(n_iter=2000, burn=1000, thin=2, seed=5,
                              subsetting=False).fit(X, B, Y)
    pip = posterior_inclusion_probs(s)
    assert pip["Y1"][0] > 0.8 and pip["Y2"][0] > 0.8
    assert pip["Y1"][1] < 0.15
    assert np.isfinite(s.meta["resid_corr_mean"])
