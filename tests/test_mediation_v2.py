"""Tests for the v2 joint spike-and-slab mediation sampler + specificity_v2."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.simulate.generator import simulate_dataset
from src.model.mediation_v2 import SpikeSlabMediation
from src.model.specificity_v2 import (
    posterior_inclusion_probs, pip_gamma, four_way_specificity,
    bayes_fdr_qvalues_from_pip, classify_balances_v2, nie_summary,
)


def _fit(n=100, K=12, sparsity=0.25, n_iter=800, burn=400, seed=0):
    data = simulate_dataset(n=n, n_balances=K, sparsity=sparsity, outcomes="FD",
                            zero_inflation="low", batch_strength="none",
                            missing="none", seed=seed)
    model = SpikeSlabMediation(n_iter=n_iter, burn=burn, thin=2, seed=seed)
    s = model.fit(np.asarray(data["X"], float), np.asarray(data["B"], float), data["Y"])
    return s, data


# ---- fast smoke fixture (shapes / invariants) ----
def _fit_smoke():
    data = simulate_dataset(n=40, n_balances=6, sparsity=0.34, outcomes="FD",
                            batch_strength="none", missing="none", seed=1)
    s = SpikeSlabMediation(n_iter=200, burn=100, thin=2, seed=1).fit(
        np.asarray(data["X"], float), np.asarray(data["B"], float), data["Y"])
    return s, data


def test_shapes_and_finiteness():
    s, data = _fit_smoke()
    K = data["B"].shape[1]
    S = s.meta["n_samples"]
    assert s.mediator["alpha"].shape == (S, K)
    assert s.mediator["gamma"].shape == (S, K)
    assert set(s.outcomes) == set(data["Y"])
    for o in data["Y"]:
        for key in ("beta", "delta", "nie", "sigma"):
            assert key in s.outcomes[o]
        assert s.outcomes[o]["beta"].shape == (S, K)
        assert np.all(np.isfinite(s.outcomes[o]["nie"]))
    assert np.all(np.isfinite(s.mediator["alpha"]))


def test_subsetting_invariant():
    """delta_{k,o}=1 must imply gamma_k=1 in EVERY draw (subsetting prior)."""
    s, data = _fit_smoke()
    g = s.mediator["gamma"]
    for o in data["Y"]:
        d = s.outcomes[o]["delta"]
        assert np.all(g[d == 1] == 1), f"subsetting violated for {o}"
    # indicators are 0/1
    assert set(np.unique(g)).issubset({0.0, 1.0})


def test_indicators_when_gamma_zero_force_beta_zero():
    s, data = _fit_smoke()
    g = s.mediator["gamma"]
    for o in data["Y"]:
        beta = s.outcomes[o]["beta"]
        # where gamma==0, beta must be exactly 0 (no mediation through unregulated balance)
        assert np.all(beta[g == 0] == 0.0)


def test_pip_recovers_true_mediators():
    s, data = _fit(seed=0)
    pip = posterior_inclusion_probs(s)
    K = data["B"].shape[1]
    for o in ("F", "D_Mw"):
        true = np.zeros(K, bool)
        true[np.asarray(data["true_mediators"][o], dtype=int)] = True
        if true.sum() == 0 or (~true).sum() == 0:
            continue
        mean_true = pip[o][true].mean()
        mean_null = pip[o][~true].mean()
        assert mean_true > mean_null, f"{o}: PIP does not rank true>null"
        assert mean_true - mean_null > 0.15, f"{o}: weak separation"
        assert pip[o][true].max() > 0.4, f"{o}: no confidently-recovered mediator"


def test_four_way_sums_to_one_and_bounds():
    s, _ = _fit_smoke()
    fw = four_way_specificity(s, outcome_F="F", outcome_D="D_Mw")
    tot = fw["joint"] + fw["feeding"] + fw["depoly"] + fw["none"]
    assert np.allclose(tot, 1.0, atol=1e-9)
    for v in fw.values():
        assert np.all((v >= -1e-9) & (v <= 1 + 1e-9))


def test_bayes_fdr_monotone_and_bounded():
    rng = np.random.default_rng(3)
    pip = rng.uniform(0, 1, 30)
    q = bayes_fdr_qvalues_from_pip(pip)
    assert q.shape == pip.shape
    assert np.all((q >= 0) & (q <= 1))
    # higher PIP -> smaller (or equal) q: q is monotone non-decreasing as PIP decreases
    order = np.argsort(-pip)            # descending PIP
    assert np.all(np.diff(q[order]) >= -1e-9)
    # a PIP=1 (pnull=0) gets q=0
    pip2 = np.array([1.0, 0.5, 0.0])
    assert bayes_fdr_qvalues_from_pip(pip2)[0] == pytest.approx(0.0)


def test_classify_v2_validation_and_labels():
    s, _ = _fit_smoke()
    with pytest.raises(ValueError):
        classify_balances_v2(s, alpha=0.0, outcome_F="F", outcome_D="D_Mw")
    res = classify_balances_v2(s, alpha=0.1, outcome_F="F", outcome_D="D_Mw")
    assert set(res["labels"]).issubset(set(CLASS := {"joint", "feeding", "depoly", "none"}))
    assert res["pip_F"].shape == res["q_F"].shape


def test_classify_v2_recovers_structure():
    """At FDR 0.1 the flagged balances should be enriched for true mediators and
    the deoxycholate-style F/D specificity should not be wildly wrong."""
    s, data = _fit(seed=0)
    K = data["B"].shape[1]
    res = classify_balances_v2(s, alpha=0.10, outcome_F="F", outcome_D="D_Mw")
    trueF = set(int(i) for i in data["true_mediators"]["F"])
    trueD = set(int(i) for i in data["true_mediators"]["D_Mw"])
    flagged = [k for k in range(K) if res["labels"][k] != "none"]
    # most flagged balances are genuinely active on at least one of F/D_Mw
    if flagged:
        hit = sum(1 for k in flagged if k in trueF or k in trueD)
        assert hit >= 0.5 * len(flagged), "too many false discoveries at FDR 0.1"


def test_nie_summary_and_lfsr():
    s, data = _fit_smoke()
    df = nie_summary(s, "F")
    assert len(df) == data["B"].shape[1]
    assert {"balance", "mean", "lo95", "hi95", "lfsr"}.issubset(df.columns)
    assert np.all((df["lfsr"] >= 0) & (df["lfsr"] <= 0.5 + 1e-9))
    assert np.all(df["lo95"] <= df["hi95"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))


def test_pi_beta_posterior_counts_trials_per_prior():
    """Coupled prior: delta ~ Bern(pi) only where gamma=1 (n_g trials); decoupled prior:
    every balance is a trial (K trials)."""
    from src.model.mediation_v2 import pi_beta_posterior
    assert pi_beta_posterior((1.0, 9.0), n_d=3, n_g=5, K=40, subsetting=True) == (4.0, 11.0)
    assert pi_beta_posterior((1.0, 9.0), n_d=3, n_g=5, K=40, subsetting=False) == (4.0, 46.0)
    # decoupled: delta can exceed gamma count; failures are still K - n_d
    assert pi_beta_posterior((1.0, 9.0), n_d=7, n_g=5, K=40, subsetting=False) == (8.0, 42.0)


def test_standardize_false_matches_on_prestandardized_data():
    """standardize=False (the SBC path) fits the data on their own scale. On data that are
    already centred and scaled, both paths see the same numbers, so the chains coincide;
    on raw data the NIE scale is the outcome's own (y_scale = 1)."""
    data = simulate_dataset(n=80, n_balances=6, sparsity=0.34, outcomes="FD",
                            batch_strength="none", missing="none", seed=3)
    X = np.asarray(data["X"], float)
    B = np.asarray(data["B"], float)
    B = (B - B.mean(0)) / B.std(0)
    Y = {o: (np.asarray(v, float) - np.mean(v)) / np.std(v) for o, v in data["Y"].items()}
    a = SpikeSlabMediation(n_iter=300, burn=100, thin=2, seed=7).fit(X, B, Y)
    b = SpikeSlabMediation(n_iter=300, burn=100, thin=2, seed=7, standardize=False).fit(X, B, Y)
    assert np.array_equal(a.mediator["gamma"], b.mediator["gamma"])
    for o in Y:
        assert np.array_equal(a.outcomes[o]["delta"], b.outcomes[o]["delta"])
        assert np.allclose(a.outcomes[o]["beta"], b.outcomes[o]["beta"], atol=1e-8)
    raw = {o: 5.0 + 3.0 * v for o, v in Y.items()}
    c = SpikeSlabMediation(n_iter=200, burn=100, thin=2, seed=7, standardize=False).fit(X, B, raw)
    assert all(c.meta["y_scale"][o] == 1.0 for o in raw)
