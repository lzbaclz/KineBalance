"""Tests for the specificity decomposition and 4-class assignment.

Constructs synthetic posterior indirect-effect (IE) draws for four balances --
one D-specific (``depoly``), one F-specific (``feeding``), one joint,
one null -- and asserts that :func:`src.model.specificity.specificity_score`,
:func:`feeding_score`, and the dominant-probability rule of
:func:`classify_balances` label each balance correctly.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.model.specificity import (
    CLASS_LABELS,
    bayes_fdr_qvalues,
    classify_balances,
    classify_balances_fdr,
    feeding_score,
    local_false_sign_rate,
    none_score,
    posterior_class_probabilities,
    specificity_score,
)


# Index convention for the constructed draws.
_FEED, _DEPOLY, _JOINT, _NONE = 0, 1, 2, 3
_TAU_F = 0.5
_TAU_D = 0.5


def _make_draws(seed: int = 0, S: int = 400):
    """Build (S, 4) IE draws with one balance per specificity class.

    * balance 0 (feeding): ``|IE_F|`` large, ``|IE_D|`` ~ 0
    * balance 1 (depoly):  ``|IE_D|`` large, ``|IE_F|`` ~ 0
    * balance 2 (joint):   both channels large
    * balance 3 (none):    both channels ~ 0
    """
    rng = np.random.default_rng(seed)
    K = 4
    ie_F = np.zeros((S, K))
    ie_D = np.zeros((S, K))

    big, small = 2.0, 0.05

    ie_F[:, _FEED] = rng.normal(big, 0.2, S)
    ie_D[:, _FEED] = rng.normal(0.0, small, S)

    ie_F[:, _DEPOLY] = rng.normal(0.0, small, S)
    ie_D[:, _DEPOLY] = rng.normal(big, 0.2, S)

    ie_F[:, _JOINT] = rng.normal(big, 0.2, S)
    ie_D[:, _JOINT] = rng.normal(big, 0.2, S)

    ie_F[:, _NONE] = rng.normal(0.0, small, S)
    ie_D[:, _NONE] = rng.normal(0.0, small, S)
    return ie_F, ie_D


def test_specificity_score_is_depoly_specific():
    """specificity_score peaks on the depoly balance and is ~0 elsewhere."""
    ie_F, ie_D = _make_draws()
    spec = specificity_score(ie_F, ie_D, _TAU_F, _TAU_D)

    assert spec.shape == (4,)
    assert spec[_DEPOLY] > 0.95
    # Feeding / joint / none should carry essentially no chemistry-specific mass.
    assert spec[_FEED] < 0.05
    assert spec[_JOINT] < 0.05
    assert spec[_NONE] < 0.05


def test_feeding_score_is_physical_specific():
    """feeding_score peaks on the feeding balance and is ~0 elsewhere."""
    ie_F, ie_D = _make_draws()
    feed = feeding_score(ie_F, ie_D, _TAU_F, _TAU_D)

    assert feed.shape == (4,)
    assert feed[_FEED] > 0.95
    assert feed[_DEPOLY] < 0.05
    assert feed[_JOINT] < 0.05
    assert feed[_NONE] < 0.05


def test_classify_balances_labels_each_class():
    """The dominant-probability rule assigns the correct label per balance."""
    ie_F, ie_D = _make_draws()
    labels = classify_balances(ie_F, ie_D, _TAU_F, _TAU_D, decision=0.5)

    assert labels.shape == (4,)
    assert labels[_FEED] == "feeding"
    assert labels[_DEPOLY] == "depoly"
    assert labels[_JOINT] == "joint"
    assert labels[_NONE] == "none"


def test_class_probabilities_partition_unity():
    """The four posterior class probabilities sum to one for every balance."""
    ie_F, ie_D = _make_draws()
    probs = posterior_class_probabilities(ie_F, ie_D, _TAU_F, _TAU_D)

    assert set(probs.keys()) == set(CLASS_LABELS)
    stacked = np.column_stack([probs[k] for k in CLASS_LABELS])
    assert stacked.shape == (4, 4)
    assert np.allclose(stacked.sum(axis=1), 1.0)
    assert np.all((stacked >= 0.0) & (stacked <= 1.0))

    # The none-balance is dominated by the "none" event.
    none = none_score(ie_F, ie_D, _TAU_F, _TAU_D)
    assert none[_NONE] > 0.95


def test_classify_indecisive_falls_back_to_none():
    """A balance with no dominant class above ``decision`` is labelled none."""
    rng = np.random.default_rng(1)
    S = 600
    # Split the depoly-active mass so no single class clears a high threshold:
    # half the draws are depoly-specific, half are null.
    ie_F = np.zeros((S, 1))
    ie_D = np.zeros((S, 1))
    half = S // 2
    ie_D[:half, 0] = rng.normal(2.0, 0.1, half)  # depoly-active draws
    ie_D[half:, 0] = rng.normal(0.0, 0.02, S - half)  # null draws

    # With decision=0.6 neither "depoly" (~0.5) nor "none" (~0.5) is decisive.
    labels = classify_balances(ie_F, ie_D, _TAU_F, _TAU_D, decision=0.6)
    assert labels[0] == "none"


def test_threshold_validation():
    """Negative thresholds and out-of-range decisions are rejected."""
    ie_F, ie_D = _make_draws()
    with pytest.raises(ValueError):
        posterior_class_probabilities(ie_F, ie_D, -0.1, _TAU_D)
    with pytest.raises(ValueError):
        classify_balances(ie_F, ie_D, _TAU_F, _TAU_D, decision=1.5)


# --------------------------------------------------------------------------- #
# Threshold-free lfsr / Bayesian-FDR rule (Step 6)
# --------------------------------------------------------------------------- #
def test_lfsr_bounds_and_sign_certainty():
    """lfsr lies in [0, 0.5]; certain-sign -> ~0, symmetric-around-0 -> ~0.5."""
    rng = np.random.default_rng(0)
    S = 4000
    draws = np.column_stack([
        rng.normal(2.0, 0.2, S),    # clearly positive -> lfsr ~ 0
        rng.normal(-2.0, 0.2, S),   # clearly negative -> lfsr ~ 0
        rng.normal(0.0, 0.5, S),    # symmetric around 0 -> lfsr ~ 0.5
    ])
    lfsr = local_false_sign_rate(draws)
    assert lfsr.shape == (3,)
    assert np.all(lfsr >= 0.0) and np.all(lfsr <= 0.5 + 1e-12)
    assert lfsr[0] < 0.01 and lfsr[1] < 0.01
    assert lfsr[2] > 0.4


def test_lfsr_requires_2d():
    with pytest.raises(ValueError):
        local_false_sign_rate(np.zeros(10))


def test_bayes_fdr_qvalues_monotone_and_bounded():
    """q-values are non-decreasing in significance order and bounded by 0.5."""
    rng = np.random.default_rng(2)
    lfsr = rng.uniform(0.0, 0.5, size=25)
    q = bayes_fdr_qvalues(lfsr)
    assert q.shape == lfsr.shape
    assert np.all(q >= 0.0) and np.all(q <= 0.5 + 1e-12)
    # In ascending-lfsr order the q-values must be monotone non-decreasing.
    order = np.argsort(lfsr, kind="mergesort")
    q_sorted = q[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)
    # The smallest-lfsr balance has q == its own lfsr (cumulative mean of one).
    assert q_sorted[0] == pytest.approx(np.sort(lfsr)[0])


def test_classify_fdr_recovers_four_classes():
    """The lfsr rule labels feeding/depoly/joint/none balances correctly."""
    ie_F, ie_D = _make_draws(seed=3, S=2000)
    res = classify_balances_fdr(ie_F, ie_D, alpha=0.05, control="lfsr")
    labels = res["labels"]
    assert labels[_FEED] == "feeding"
    assert labels[_DEPOLY] == "depoly"
    assert labels[_JOINT] == "joint"
    assert labels[_NONE] == "none"
    # Active masks must agree with the labels.
    assert res["active_F"][_FEED] and not res["active_D"][_FEED]
    assert res["active_D"][_DEPOLY] and not res["active_F"][_DEPOLY]
    assert res["active_F"][_JOINT] and res["active_D"][_JOINT]
    assert not res["active_F"][_NONE] and not res["active_D"][_NONE]
    # lfsr of an active, sign-certain channel is tiny; of a null channel ~0.5.
    assert res["lfsr_F"][_FEED] < 0.05 and res["lfsr_D"][_FEED] > 0.3


def test_classify_fdr_validation():
    ie_F, ie_D = _make_draws()
    with pytest.raises(ValueError):
        classify_balances_fdr(ie_F, ie_D, alpha=0.0)
    with pytest.raises(ValueError):
        classify_balances_fdr(ie_F, ie_D, alpha=0.6)
    with pytest.raises(ValueError):
        classify_balances_fdr(ie_F, ie_D, control="bogus")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
