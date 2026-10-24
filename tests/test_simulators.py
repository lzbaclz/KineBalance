"""Unit tests for the simulators and balance options."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.preprocess.balances import philr_balances, zero_replace
from src.simulate.clade_plasmode import (ilr_loading, match_balances, sbp_loadings,
                                         simulate_clade_counts)
from src.simulate.plasmode import simulate_specificity_design


def _counts(seed=0, n=60, p=12):
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(p) * 2.0, size=n)
    return np.vstack([rng.multinomial(5000, b) for b in base]) + 1


def test_ilr_loadings_are_orthonormal_contrasts_matching_philr():
    C = _counts()
    B, sbp, _ = philr_balances(C, tree=None, seed=0)
    L = sbp_loadings(sbp, C.shape[1])
    assert np.allclose(L @ L.T, np.eye(len(sbp)), atol=1e-10)      # orthonormal
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-12)              # contrasts
    comp = zero_replace(C)
    clr = np.log(comp) - np.log(comp).mean(axis=1, keepdims=True)
    assert np.allclose(clr @ L.T, B, atol=1e-10)                    # balance = clr . v


def test_clade_shift_moves_only_its_own_coordinate():
    C = _counts(1)
    _, sbp, _ = philr_balances(C, tree=None, seed=0)
    L = sbp_loadings(sbp, C.shape[1])
    b = 3
    shift = 0.7 * L[b]                          # the injected perturbation in CLR space
    delta = L @ shift
    assert delta[b] == pytest.approx(0.7)
    assert np.allclose(np.delete(delta, b), 0.0, atol=1e-12)


def test_simulate_clade_counts_shapes_and_classes():
    C = _counts(2, n=80, p=24)
    _, sbp, _ = philr_balances(C, tree=None, seed=0)
    d = simulate_clade_counts(C, 50, 7, sbp, min_size=3, max_size=11)
    assert d["counts"].shape == (50, C.shape[1])
    assert list(d["target_class"]).count("joint") == 2
    assert all(3 <= s <= 11 for s in d["target_sizes"])
    m = match_balances(sbp_loadings(sbp, C.shape[1]), d["target_loadings"])
    assert m["matched"].sum() == len(d["targets"])      # the reference basis contains them


def test_specificity_design_classes_and_default_stream():
    B = np.random.default_rng(0).normal(size=(120, 20))
    d = simulate_specificity_design(100, 15, 3, B, 1, 3, 2, effect_scale=2.0)
    labs = list(d["labels"])
    assert (labs.count("joint"), labs.count("feeding"), labs.count("depoly")) == (1, 3, 2)
    e = simulate_specificity_design(100, 15, 3, B, 1, 3, 2, effect_scale=2.0,
                                    n_yonly_f=0, n_yonly_d=0)
    assert np.array_equal(d["Y"]["F"], e["Y"]["F"])     # new options default to no-ops
    f = simulate_specificity_design(100, 15, 3, B, 1, 3, 2, n_yonly_f=1, n_yonly_d=1)
    assert np.all(f["true_alphaX"][f["outcome_only"]] == 0.0)


def test_residual_correlation_is_the_requested_one():
    B = np.random.default_rng(0).normal(size=(300, 20))
    r = []
    for s in range(30):
        d = simulate_specificity_design(300, 15, s, B, 2, 2, 2, resid_rho=0.6)
        eF = d["Y"]["F"] - d["B"] @ d["true_gamma"]["F"]
        eD = d["Y"]["D_Mw"] - d["B"] @ d["true_gamma"]["D_Mw"]
        r.append(np.corrcoef(eF, eD)[0, 1])
    assert np.mean(r) == pytest.approx(0.6, abs=0.03)


def test_balance_options_default_is_unchanged_and_variants_run():
    C = _counts(4)
    B0, _, _ = philr_balances(C, tree=None, seed=0)
    B1, _, _ = philr_balances(C, tree=None, seed=0, ward_metric="corr", pseudo_counts=None)
    assert np.array_equal(B0, B1)
    Bv, sbp_v, _ = philr_balances(C, tree=None, seed=0, ward_metric="variation")
    Bp, _, _ = philr_balances(C, tree=None, seed=0, pseudo_counts=0.5)
    assert Bv.shape == B0.shape and Bp.shape == B0.shape
    with pytest.raises(ValueError):
        philr_balances(C, tree=None, seed=0, ward_metric="bogus")


def test_clade_coordinate_truth_is_exact_on_the_reference_basis():
    """On the fixed reference tree the coordinate-level non-null cells
    (a_k beta_{k,r} != 0 computed from the simulator's CLR-space effects) are exactly the
    injected clades; on a rebuilt tree an injected clade spreads over several balances."""
    C = _counts(2, n=80, p=24)
    _, sbp, _ = philr_balances(C, tree=None, seed=0)
    d = simulate_clade_counts(C, 60, 7, sbp, min_size=3, max_size=11)
    L = sbp_loadings(sbp, C.shape[1])
    a = L @ d["exposure_clr_effect"]
    for r, keep in (("F", ("joint", "feeding")), ("D", ("joint", "depoly"))):
        b = L @ d["outcome_clr_coef"][r]
        nonnull = np.nonzero((np.abs(a) > 1e-9) & (np.abs(b) > 1e-9))[0]
        expect = sorted(int(t) for t, c in zip(d["targets"], d["target_class"]) if c in keep)
        assert sorted(nonnull.tolist()) == expect
    _, sbp2, _ = philr_balances(d["counts"], tree=None, seed=0)
    a2 = sbp_loadings(sbp2, C.shape[1]) @ d["exposure_clr_effect"]
    assert (np.abs(a2) > 1e-9).sum() >= (np.abs(a) > 1e-9).sum()


def test_count_plasmode_returns_effect_bookkeeping():
    from src.simulate.count_plasmode import simulate_count_plasmode
    C = _counts(3, n=80, p=20)
    d = simulate_count_plasmode(C, n=50, m_med=4, seed=11)
    assert len(d["a_signs"]) == len(d["b_signs"]) == len(d["col_sd"]) == 4
    assert sorted(d["med_order"].tolist()) == d["med_idx"].tolist()
    assert np.all(d["col_sd"] > 0)
