"""Tests for the SOTA-benchmark count plasmode + single-outcome FDR selection."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.simulate.count_plasmode import simulate_count_plasmode, clr
from src.model.mediation_v2 import SpikeSlabMediation
from src.model.specificity_v2 import select_mediators_fdr


def _real(seed=0, N0=120, K=30):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.3, 1.0, size=(N0, K))
    return rng.poisson(base * 200)


@pytest.mark.parametrize("model", ["clr", "ra"])
def test_plasmode_shapes_and_truth(model):
    d = simulate_count_plasmode(_real(), n=80, m_med=4, seed=1, outcome_model=model)
    assert d["counts"].shape == (80, 30)
    assert d["counts"].dtype.kind in "iu" and np.all(d["counts"] >= 0)
    assert d["X"].shape == (80,) and d["Y"].shape == (80,)
    assert int(d["true_mediators"].sum()) == 4
    assert d["med_idx"].size == 4
    assert np.all(np.isfinite(d["Y"]))


def test_clr_centered():
    c = _real(seed=2, N0=20, K=10)
    z = clr(c)
    assert np.allclose(z.sum(axis=1), 0.0, atol=1e-8)   # CLR rows sum to 0


def test_m_med_must_be_valid():
    with pytest.raises(ValueError):
        simulate_count_plasmode(_real(N0=20, K=5), n=20, m_med=5, seed=0)


def test_select_mediators_fdr_recovers_signal():
    """On a plasmode with strong mediators, v2 selection flags true > null."""
    d = simulate_count_plasmode(_real(seed=3), n=160, m_med=4, seed=3,
                                alpha_eff=1.2, beta_eff=1.5)
    s = SpikeSlabMediation(n_iter=1500, burn=700, thin=2, seed=3).fit(
        d["X"].astype(float), d["clr"], {"Y": d["Y"]})
    out = select_mediators_fdr(s, "Y", alpha=0.10)
    assert out["selected"].shape == (30,)
    assert np.all((out["qval"] >= 0) & (out["qval"] <= 1))
    truth = d["true_mediators"].astype(bool)
    # mean PIP on true mediators exceeds mean PIP on nulls
    assert out["pip"][truth].mean() > out["pip"][~truth].mean() + 0.1


def test_warm_start_high_dim_no_collapse():
    """Regression: in the K>n regime the warm-started sampler must not collapse
    (stage-2 beta -> 0 / stage-1 gamma -> all-1). Plain init can degenerate there."""
    rng = np.random.default_rng(0)
    n, K = 160, 100                            # high-dimensional (K<n): the real-data regime
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K))
    B[:, 3] += 1.3 * X                          # balance 3: X-regulated
    y = 1.4 * B[:, 3] + rng.normal(0, 1.0, n)   # ...and outcome-associated -> mediator
    s = SpikeSlabMediation(n_iter=1500, burn=700, thin=2, seed=1,
                           warm_start=True).fit(X, B, {"Y": y})
    beta = s.outcomes["Y"]["beta"]
    gamma_mean = s.mediator["gamma"].mean(0)
    pip = select_mediators_fdr(s, "Y", alpha=0.10)["pip"]
    assert np.abs(beta).max() > 0.1            # beta did NOT collapse to ~0
    assert gamma_mean.mean() < 0.9            # gamma is selective, not all-1
    assert pip[3] > pip.mean()                 # the true mediator stands out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
