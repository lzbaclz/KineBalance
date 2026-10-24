"""Tests for the Step-3 plasmode / misspecification generators.

Checks that every mode returns the ``simulate_dataset`` dict schema (so the main
sweep's scorer can consume it), that the structural ground-truth indirect effect
is ``gamma * alphaX * delta_x`` in every mode, and that plasmode actually uses
the supplied real covariance.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.simulate.plasmode import MODES, simulate_step3

_N, _K = 40, 30
_DELTA = float(np.log(600.0) - np.log(200.0))  # low-to-high exposure contrast (Eq. 4)


def _real_B(rows=120, k=_K, seed=1):
    """A correlated stand-in for real balances (block covariance, heavy-ish)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((rows, k))
    z[:, 1:] += 0.7 * z[:, :-1]  # induce cross-balance correlation
    return z


@pytest.mark.parametrize("mode", MODES)
def test_schema_and_shapes(mode):
    rB = _real_B() if mode == "plasmode" else None
    d = simulate_step3(mode, _N, _K, sparsity=0.1, seed=3, outcomes="FD", real_B=rB)
    for key in ("X", "C", "batch", "B", "Y", "true_mediators",
                "true_gamma", "true_alphaX", "true_IE", "counts"):
        assert key in d, f"{mode}: missing key {key}"
    assert d["X"].shape == (_N,)
    assert d["B"].shape == (_N, _K)
    assert set(d["Y"]) == {"F", "D_Mw", "D_Mn"}
    assert d["true_alphaX"].shape == (_K,)
    for nm in ("F", "D_Mw", "D_Mn"):
        assert d["Y"][nm].shape == (_N,)
        assert d["true_IE"][nm].shape == (_K,)
        assert np.all(np.isfinite(d["Y"][nm]))
    assert np.all(np.isfinite(d["B"]))


@pytest.mark.parametrize("mode", MODES)
def test_true_ie_is_structural(mode):
    """true_IE must equal gamma * alphaX * delta_x exactly, in every mode."""
    rB = _real_B() if mode == "plasmode" else None
    d = simulate_step3(mode, _N, _K, sparsity=0.1, seed=5, outcomes="FD", real_B=rB)
    for nm in ("F", "D_Mw", "D_Mn"):
        expected = d["true_gamma"][nm] * d["true_alphaX"] * _DELTA
        np.testing.assert_allclose(d["true_IE"][nm], expected, rtol=1e-10, atol=1e-12)


def test_partial_overlap_active_sets():
    d = simulate_step3("wellspec", _N, _K, sparsity=0.1, seed=7, outcomes="FD")
    f = set(int(i) for i in d["true_mediators"]["F"])
    dd = set(int(i) for i in d["true_mediators"]["D_Mw"])
    assert f and dd
    assert f & dd            # some overlap (joint balances exist)
    assert (f - dd) or (dd - f)  # but not identical


def test_plasmode_requires_real_B():
    with pytest.raises(ValueError):
        simulate_step3("plasmode", _N, _K, sparsity=0.1, seed=0, outcomes="FD")


def test_plasmode_uses_real_covariance():
    """The plasmode residual (B minus the injected X term) should reflect the
    supplied real covariance, not an i.i.d. Gaussian."""
    rB = _real_B(seed=2)
    d = simulate_step3("plasmode", _N, _K, sparsity=0.1, seed=4, outcomes="FD", real_B=rB)
    # Neighbour-column correlation in B should be clearly positive (the real_B
    # block structure), unlike the ~0 of an i.i.d. design.
    B = d["B"]
    corrs = [np.corrcoef(B[:, j], B[:, j + 1])[0, 1] for j in range(_K - 1)]
    assert np.mean(corrs) > 0.2


def test_invalid_mode():
    with pytest.raises(ValueError):
        simulate_step3("bogus", _N, _K, sparsity=0.1, seed=0)


def test_determinism():
    a = simulate_step3("wellspec", _N, _K, sparsity=0.1, seed=11, outcomes="FD")
    b = simulate_step3("wellspec", _N, _K, sparsity=0.1, seed=11, outcomes="FD")
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["B"], b["B"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
