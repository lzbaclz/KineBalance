"""Gold-standard validation: Gibbs PIPs (fixed-scale mode) must match the exact
enumerated posterior on a tiny case. This proves the discrete-selection sampler
targets the correct posterior."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.model.mediation_v2 import SpikeSlabMediation
from src.model.specificity_v2 import posterior_inclusion_probs, pip_gamma
from src.model.validation import enumerate_posterior


def _tiny_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B0 = 0.8 * X + rng.normal(0, 1.0, n)     # balance 0: regulated by X
    B1 = rng.normal(0, 1.0, n)               # balance 1: null (no X association)
    B = np.column_stack([B0, B1])
    y = 0.9 * B0 + rng.normal(0, 1.0, n)     # only B0 mediates
    # standardized copies the sampler will reproduce internally
    Bs = (B - B.mean(0)) / B.std(0)
    ys = (y - y.mean()) / y.std()
    return X, B, y, Bs, ys


FIXED = {"sM2": 1.0, "so2": 1.0, "va": 4.0, "vb": 4.0, "pi_a": 0.3, "pi_b": 0.3}


def test_gibbs_pip_matches_exact_enumeration():
    X, B, y, Bs, ys = _tiny_data(seed=0)
    enum = enumerate_posterior(X, Bs, ys, va=FIXED["va"], vb=FIXED["vb"],
                               sM2=FIXED["sM2"], so2=FIXED["so2"],
                               pi_a=FIXED["pi_a"], pi_b=FIXED["pi_b"])
    model = SpikeSlabMediation(n_iter=9000, burn=3000, thin=2, seed=1, fixed=FIXED)
    s = model.fit(X, B, {"Y": y})
    gibbs_pip = posterior_inclusion_probs(s)["Y"]
    gibbs_pg = pip_gamma(s)

    # the true mediator (balance 0) should be confidently recovered by BOTH
    assert enum["pip"][0] > 0.6
    # Gibbs matches the exact posterior within Monte-Carlo error
    assert np.allclose(gibbs_pip, enum["pip"], atol=0.06), \
        f"PIP mismatch: gibbs={gibbs_pip}, enum={enum['pip']}"
    assert np.allclose(gibbs_pg, enum["pip_gamma"], atol=0.06), \
        f"PIP_gamma mismatch: gibbs={gibbs_pg}, enum={enum['pip_gamma']}"


def test_enumeration_self_consistency():
    """PIPs are bounded, and the null balance has lower inclusion than the mediator."""
    X, B, y, Bs, ys = _tiny_data(seed=3)
    enum = enumerate_posterior(X, Bs, ys, va=4.0, vb=4.0, sM2=1.0, so2=1.0,
                               pi_a=0.3, pi_b=0.3)
    assert np.all((enum["pip"] >= 0) & (enum["pip"] <= 1))
    assert np.all(enum["pip"] <= enum["pip_gamma"] + 1e-9)   # joint <= gamma marginal
    assert enum["pip"][0] > enum["pip"][1]                    # mediator > null


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))


def test_gibbs_pip_matches_enumeration_K5():
    """Exact-enumeration check at K=5 (243 configs) -- beyond the K=2 toy."""
    rng = np.random.default_rng(4)
    n = 80
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, 5))
    B[:, 0] = 0.8 * X + rng.normal(0, 1, n)        # mediator
    B[:, 2] = 0.7 * X + rng.normal(0, 1, n)        # mediator
    y = 0.9 * B[:, 0] - 0.8 * B[:, 2] + rng.normal(0, 1, n)
    Bs = (B - B.mean(0)) / B.std(0); ys = (y - y.mean()) / y.std()
    enum = enumerate_posterior(X, Bs, ys, va=FIXED["va"], vb=FIXED["vb"],
                               sM2=FIXED["sM2"], so2=FIXED["so2"],
                               pi_a=FIXED["pi_a"], pi_b=FIXED["pi_b"])
    s = SpikeSlabMediation(n_iter=12000, burn=4000, thin=2, seed=2, fixed=FIXED).fit(X, B, {"Y": y})
    gibbs = posterior_inclusion_probs(s)["Y"]
    assert np.allclose(gibbs, enum["pip"], atol=0.07), f"K=5 mismatch: {gibbs} vs {enum['pip']}"


def test_gibbs_pip_matches_enumeration_two_outcome():
    """Exact two-outcome enumeration (5**K configs) vs Gibbs, K=3."""
    from src.model.validation import enumerate_posterior_2outcome
    rng = np.random.default_rng(7)
    n = 70
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, 3))
    B[:, 0] = 0.8 * X + rng.normal(0, 1, n)        # joint (both outcomes)
    B[:, 1] = 0.7 * X + rng.normal(0, 1, n)        # F-specific
    yF = 0.9 * B[:, 0] + 0.9 * B[:, 1] + rng.normal(0, 1, n)
    yD = 0.9 * B[:, 0] + rng.normal(0, 1, n)
    Bs = (B - B.mean(0)) / B.std(0)
    yFs = (yF - yF.mean()) / yF.std(); yDs = (yD - yD.mean()) / yD.std()
    enum = enumerate_posterior_2outcome(X, Bs, yFs, yDs, va=FIXED["va"], vb=FIXED["vb"],
                                        sM2=FIXED["sM2"], soF2=FIXED["so2"], soD2=FIXED["so2"],
                                        pi_a=FIXED["pi_a"], pi_bF=FIXED["pi_b"], pi_bD=FIXED["pi_b"])
    s = SpikeSlabMediation(n_iter=12000, burn=4000, thin=2, seed=3, fixed=FIXED).fit(
        X, B, {"F": yF, "D": yD})
    pip = posterior_inclusion_probs(s)
    assert np.allclose(pip["F"], enum["pip_F"], atol=0.08), f"F: {pip['F']} vs {enum['pip_F']}"
    assert np.allclose(pip["D"], enum["pip_D"], atol=0.08), f"D: {pip['D']} vs {enum['pip_D']}"


def test_gibbs_matches_enumeration_high_pi_b():
    """Regression for the gamma-update prior-mass bug: at high pi_b the gamma update
    must pay the sum_o log(1-pi_b_o) penalty, else null PIPs are inflated."""
    rng = np.random.default_rng(0); n = 60
    X = rng.standard_normal(n)
    B = np.column_stack([0.8 * X + rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    y = 0.9 * B[:, 0] + rng.normal(0, 1, n)
    Bs = (B - B.mean(0)) / B.std(0); ys = (y - y.mean()) / y.std()
    FX = {"sM2": 1.0, "so2": 1.0, "va": 4.0, "vb": 4.0, "pi_a": 0.3, "pi_b": 0.8}
    enum = enumerate_posterior(X, Bs, ys, va=4.0, vb=4.0, sM2=1.0, so2=1.0, pi_a=0.3, pi_b=0.8)
    s = SpikeSlabMediation(n_iter=12000, burn=4000, thin=2, seed=1, fixed=FX).fit(X, B, {"Y": y})
    g = posterior_inclusion_probs(s)["Y"]
    assert np.allclose(g, enum["pip"], atol=0.04), f"high-pi_b: {g} vs {enum['pip']}"


def _outcome_only_data(seed=11, n=80):
    """K=3: B0 joint mediator, B1 outcome-only (no exposure path), B2 null."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, 3))
    B[:, 0] = 0.8 * X + rng.normal(0, 1, n)
    yF = 0.9 * B[:, 0] + 0.9 * B[:, 1] + rng.normal(0, 1, n)   # B1 acts on F only
    yD = 0.9 * B[:, 0] + rng.normal(0, 1, n)
    Bs = (B - B.mean(0)) / B.std(0)
    zs = lambda v: (v - v.mean()) / v.std()
    return X, B, yF, yD, Bs, zs(yF), zs(yD)


def test_gibbs_matches_enumeration_two_outcome_decoupled():
    """Decoupled variant (no subsetting): Gibbs vs exact 8**K enumeration, K=3."""
    from src.model.validation import enumerate_posterior_2outcome
    X, B, yF, yD, Bs, yFs, yDs = _outcome_only_data()
    enum = enumerate_posterior_2outcome(X, Bs, yFs, yDs, va=4.0, vb=4.0, sM2=1.0,
                                        soF2=1.0, soD2=1.0, pi_a=0.3, pi_bF=0.3, pi_bD=0.3,
                                        subsetting=False)
    s = SpikeSlabMediation(n_iter=12000, burn=4000, thin=2, seed=5, fixed=FIXED,
                           subsetting=False).fit(X, B, {"F": yF, "D": yD})
    pip = posterior_inclusion_probs(s)
    assert np.allclose(pip["F"], enum["pip_F"], atol=0.08), f"F: {pip['F']} vs {enum['pip_F']}"
    assert np.allclose(pip["D"], enum["pip_D"], atol=0.08), f"D: {pip['D']} vs {enum['pip_D']}"


def test_outcome_only_balance_inflates_coupled_pip_only():
    """Exact posteriors: an outcome-only balance (a=0, beta!=0) gets a high mediation
    PIP under the subsetting prior but not under the decoupled variant."""
    from src.model.validation import enumerate_posterior_2outcome
    X, B, yF, yD, Bs, yFs, yDs = _outcome_only_data()
    kw = dict(va=4.0, vb=4.0, sM2=1.0, soF2=1.0, soD2=1.0, pi_a=0.3, pi_bF=0.3, pi_bD=0.3)
    coupled = enumerate_posterior_2outcome(X, Bs, yFs, yDs, subsetting=True, **kw)
    decoupled = enumerate_posterior_2outcome(X, Bs, yFs, yDs, subsetting=False, **kw)
    # the endpoint-only balance (a=0, beta!=0): its F-mediation probability is inflated
    # several-fold by the coupling (exact values here: ~0.47 vs ~0.06)
    assert coupled["pip_F"][1] > 0.3, coupled["pip_F"]
    assert coupled["pip_F"][1] > 5 * decoupled["pip_F"][1], (coupled["pip_F"], decoupled["pip_F"])
    assert decoupled["pip_F"][1] < 0.15, decoupled["pip_F"]     # composite null respected
    assert decoupled["pip_F"][0] > 0.9 and coupled["pip_F"][0] > 0.9   # true mediator kept
