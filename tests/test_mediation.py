"""Tests for KineBalance mediation recovery.

On a small synthetic dataset with known true mediators (from
:func:`src.simulate.generator.simulate_dataset`), a short Gibbs chain of
:class:`src.model.mediation.KineBalanceModel` is fitted and we assert that the
posterior magnitude of the mediator weight ``gamma`` (and of the natural
indirect effect ``IE`` from :func:`src.model.indirect.indirect_effect`) is, on
average, LARGER for the true mediators than for the null balances. This is a
tolerance-based recovery check, not an exact-value check; all outputs must be
finite.

The seeds are pinned to a configuration verified to recover under a deliberately
short chain (~250 iterations) so the test stays fast and deterministic.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.model.indirect import indirect_effect, summarize_ie
from src.model.mediation import KineBalanceModel, _mvn_from_precision
from src.simulate.generator import simulate_dataset


# Pinned, recovery-verified configuration (small + fast).
_SIM_SEED = 0
_MODEL_SEED = 0
_N = 60
_N_BALANCES = 12
_SPARSITY = 0.25
_N_ITER = 250
_BURN = 125
_THIN = 2


def _fit_single_outcome():
    """Simulate, fit a short KineBalance chain, and return draws + truth."""
    data = simulate_dataset(
        n=_N,
        n_balances=_N_BALANCES,
        sparsity=_SPARSITY,
        zero_inflation="low",
        batch_strength="none",
        outcomes="F",
        missing="none",
        seed=_SIM_SEED,
    )
    B = data["B"]
    X = data["X"]
    Y = {"F": data["Y"]["F"]}
    true_mediators = data["true_mediators"]["F"]

    model = KineBalanceModel(
        n_iter=_N_ITER,
        burn=_BURN,
        thin=_THIN,
        ridge_sd=10.0,
        seed=_MODEL_SEED,
    )
    samples = model.fit(X=X, B=B, Y=Y)
    return samples, B.shape[1], true_mediators


def _true_null_masks(n_balances: int, true_mediators: np.ndarray):
    """Boolean masks selecting the true-mediator and null-balance columns."""
    mask_true = np.zeros(n_balances, dtype=bool)
    mask_true[true_mediators] = True
    mask_null = ~mask_true
    return mask_true, mask_null


def test_fit_shapes_and_finiteness():
    """Posterior draws have the documented shapes and are all finite."""
    samples, K, _ = _fit_single_outcome()

    n_keep = samples.meta["n_samples"]
    assert samples.meta["K"] == K

    gamma = samples.outcomes["F"]["gamma"]
    aX = samples.mediator["aX"]
    assert gamma.shape == (n_keep, K)
    assert aX.shape == (n_keep, K)
    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(aX))
    assert np.all(np.isfinite(samples.outcomes["F"]["gX"]))
    assert np.all(np.isfinite(samples.outcomes["F"]["sigma"]))
    # Variance scales must be strictly positive.
    assert np.all(samples.outcomes["F"]["sigma"] > 0.0)


def test_gamma_larger_for_true_mediators():
    """Posterior mean |gamma| is larger for true mediators than null balances."""
    samples, K, true_mediators = _fit_single_outcome()
    mask_true, mask_null = _true_null_masks(K, true_mediators)

    gamma = samples.outcomes["F"]["gamma"]  # (S, K)
    abs_gamma_mean = np.abs(gamma).mean(axis=0)  # (K,)

    mean_true = abs_gamma_mean[mask_true].mean()
    mean_null = abs_gamma_mean[mask_null].mean()

    assert np.isfinite(mean_true) and np.isfinite(mean_null)
    # Recovery: true mediators carry clearly more posterior mass. The margin is
    # large for this pinned config; a generous factor keeps the test robust.
    assert mean_true > mean_null
    assert mean_true > 2.0 * mean_null
    # Magnitude floor (not just a ratio): guards against everything shrinking to
    # ~0 uniformly, which would satisfy the ratio test vacuously. For this pinned
    # config mean_true~0.50, mean_null~0.09; these floors keep wide margin.
    assert mean_true > 0.25
    assert (mean_true - mean_null) > 0.15


def test_indirect_effect_larger_for_true_mediators():
    """Posterior |IE| (Eq. 6) is larger for true mediators than null balances."""
    samples, K, true_mediators = _fit_single_outcome()
    mask_true, mask_null = _true_null_masks(K, true_mediators)

    # Any nonzero exposure contrast suffices here.
    ie = indirect_effect(samples, x_a=0.0, x_b=1.0)["F"]  # (S, K)
    assert ie.shape == samples.outcomes["F"]["gamma"].shape
    assert np.all(np.isfinite(ie))

    abs_ie_mean = np.abs(ie).mean(axis=0)
    mean_true = abs_ie_mean[mask_true].mean()
    mean_null = abs_ie_mean[mask_null].mean()

    assert np.isfinite(mean_true) and np.isfinite(mean_null)
    assert mean_true > mean_null
    assert mean_true > 2.0 * mean_null


def test_top_balance_is_a_true_mediator():
    """The single largest posterior |IE| balance is a genuine mediator."""
    samples, K, true_mediators = _fit_single_outcome()
    ie = indirect_effect(samples, x_a=0.0, x_b=1.0)["F"]
    abs_ie_mean = np.abs(ie).mean(axis=0)

    top = int(np.argmax(abs_ie_mean))
    assert top in set(int(t) for t in true_mediators)


def test_summarize_ie_outputs_finite():
    """summarize_ie returns one finite row per balance with valid stabilities."""
    samples, K, _ = _fit_single_outcome()
    ie = indirect_effect(samples, x_a=0.0, x_b=1.0)["F"]

    df = summarize_ie(ie)
    assert len(df) == K
    assert np.all(np.isfinite(df["mean"].to_numpy()))
    # Direction stability lives in [0.5, 1] by construction.
    stab = df["direction_stability"].to_numpy()
    assert np.all(stab >= 0.5 - 1e-9)
    assert np.all(stab <= 1.0 + 1e-9)
    p_pos = df["prob_positive"].to_numpy()
    assert np.all((p_pos >= 0.0) & (p_pos <= 1.0))


def test_mvn_from_precision_rank_deficient():
    """The adaptive-jitter/pinv path returns a finite draw for a singular
    precision (the p>>n regime that motivated the fallback)."""
    rng = np.random.default_rng(0)
    p = 6
    # Rank-1 (singular) precision: a fixed tiny jitter would not suffice.
    v = rng.standard_normal(p)
    prec = np.outer(v, v)
    b = rng.standard_normal(p)
    draw = _mvn_from_precision(rng, prec, b)
    assert draw.shape == (p,)
    assert np.all(np.isfinite(draw))
    # A well-conditioned SPD precision still yields a finite draw of right shape.
    spd = np.eye(p) * 3.0 + 0.1 * np.outer(v, v)
    draw2 = _mvn_from_precision(rng, spd, b)
    assert draw2.shape == (p,) and np.all(np.isfinite(draw2))


def test_fit_stable_in_realistic_p_gg_n_regime():
    """End-to-end fit at K=100, n=48 (rank-deficient precision) yields finite
    posterior draws -- exercises the adaptive-jitter/pinv path inside fit(),
    not just the unit-level _mvn_from_precision toy."""
    data = simulate_dataset(n=48, n_balances=100, sparsity=0.05, outcomes="FD",
                            batch_strength="none", missing="none", seed=21)
    model = KineBalanceModel(n_iter=300, burn=150, thin=2, ridge_sd=10.0, seed=0)
    model.fit(np.asarray(data["X"], float), np.asarray(data["B"], float), data["Y"],
              C=np.asarray(data["C"], float), batch=np.asarray(data["batch"]))
    assert model.samples.mediator["aX"].shape[1] == 100
    assert np.all(np.isfinite(model.samples.mediator["aX"]))
    for nm in data["Y"]:
        assert np.all(np.isfinite(model.samples.outcomes[nm]["gamma"]))


def test_y_scale_stored_and_used():
    """The model stores a per-outcome y_scale; indirect_effect uses it silently,
    and warns only when it is missing (guards the silent-1.0 fallback)."""
    import warnings
    data = simulate_dataset(n=48, n_balances=12, sparsity=0.2, outcomes="FD",
                            batch_strength="none", missing="none", seed=3)
    model = KineBalanceModel(n_iter=200, burn=100, thin=2, ridge_sd=10.0, seed=0)
    model.fit(np.asarray(data["X"], float), np.asarray(data["B"], float), data["Y"],
              C=np.asarray(data["C"], float), batch=np.asarray(data["batch"]))
    y_scale = model.samples.meta.get("y_scale", {})
    assert set(data["Y"]).issubset(set(y_scale)) and all(v > 0 for v in y_scale.values())
    # Present y_scale -> no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        indirect_effect(model.samples, x_a=0.0, x_b=1.0)
    # Stripped y_scale -> RuntimeWarning (no silent unit-scale IEs).
    model.samples.meta["y_scale"] = {}
    with pytest.warns(RuntimeWarning):
        indirect_effect(model.samples, x_a=0.0, x_b=1.0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
