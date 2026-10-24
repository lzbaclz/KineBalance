"""Unit tests for the B2 label-level class-FDR machinery (src/model/classfdr)."""
import numpy as np
import pytest

from src.model.classfdr import class_fdr_qvalues, classfdr_labels, _CLASSES
from src.model.specificity_v2 import four_way_specificity


def test_class_fdr_qvalues_direct_bayes_fdr():
    # P_c = [0.9, 0.8, 0.2] -> pnull = [0.1, 0.2, 0.8]; sorted cumulative means are
    # [0.1, 0.15, 0.367], monotone-guarded, returned in input order.
    q = class_fdr_qvalues(np.array([0.9, 0.8, 0.2]))
    np.testing.assert_allclose(q, [0.1, 0.15, (0.1 + 0.2 + 0.8) / 3], atol=1e-9)


def test_class_fdr_qvalues_monotone_and_bounded():
    rng = np.random.default_rng(0)
    q = class_fdr_qvalues(rng.random(50))
    assert q.min() >= 0.0 and q.max() <= 1.0
    # sorting by pnull ascending, the q-values are nondecreasing along that order
    p = 1.0 - rng.random(50)  # arbitrary; check the guarantee on a known vector
    p = np.array([0.05, 0.1, 0.9, 0.2])
    qq = class_fdr_qvalues(p)          # pnull = [0.95,0.9,0.1,0.8]
    order = np.argsort(1.0 - p)
    assert np.all(np.diff(qq[order]) >= -1e-12)


# --- classfdr_labels: the argmax-restriction Lemma and the discovery gate -----------
# Build a fake ``samples`` directly from binary posterior draws. With gamma == 1 the
# per-outcome activation aF = gamma*delta_F equals delta_F, so setting the (joint /
# F-only / D-only / none) draw counts per balance realizes exact class posteriors.

class _FakeSamples:
    def __init__(self, aF, aD):
        S, K = aF.shape
        self.mediator = {"gamma": np.ones((S, K))}
        self.outcomes = {"F": {"delta": aF.astype(float)},
                         "D": {"delta": aD.astype(float)}}
        self.meta = {"K": K, "names": ["F", "D"]}


def _fake_from_targets(targets, S=1000):
    """targets[k] = (P_joint, P_feed, P_dep); the remainder is P_none."""
    K = len(targets)
    aF = np.zeros((S, K), int)
    aD = np.zeros((S, K), int)
    for k, (pj, pf, pd) in enumerate(targets):
        nj, nf, nd = round(pj * S), round(pf * S), round(pd * S)
        i = 0
        aF[i:i + nj, k] = 1; aD[i:i + nj, k] = 1; i += nj     # joint
        aF[i:i + nf, k] = 1; aD[i:i + nf, k] = 0; i += nf     # feeding (F only)
        aF[i:i + nd, k] = 0; aD[i:i + nd, k] = 1; i += nd     # depoly (D only)
        # rest: none (already 0/0)
    return _FakeSamples(aF, aD)


# k0..k4 strong joint; k5 borderline joint(.45)/feed(.55) -> lands in the joint prefix
# but its argmax is feeding; k6 strong feeding; k7 pure none; k8 strong depoly; k9
# none-dominant (P_none=.70) yet its argmax over the three active classes is joint.
_TARGETS = ([(.995, 0, 0)] * 5
            + [(.45, .55, 0), (0, .92, 0), (0, 0, 0), (0, 0, .92), (.30, 0, 0)])


def test_classfdr_labels_lemma_subset_and_argmax_restriction():
    a = 0.10
    fake = _fake_from_targets(_TARGETS)
    res = classfdr_labels(fake, alpha=a, outcome_F="F", outcome_D="D",
                          require_discovery=False)
    labels, qbc = res["labels"], res["q_by_class"]
    P = four_way_specificity(fake, "F", "D")
    Pmat = np.vstack([P[c] for c in _CLASSES])
    argmax = Pmat.argmax(0)

    removed_any = False
    for ci, c in enumerate(_CLASSES):
        Sc = set(np.where(qbc[c] <= a + 1e-12)[0])                   # full-prefix set
        Dc = {k for k in range(len(labels)) if str(labels[k]) == c}  # declared set
        assert Dc <= Sc                                             # (1) D_c subseteq S_c
        for k in Sc - Dc:
            assert argmax[k] != ci                                  # (2) removed => non-argmax
            removed_any = True
        if Dc:                                                      # (3) posterior FDP <= alpha
            assert np.mean([1 - Pmat[ci, k] for k in Dc]) <= a + 1e-9

    assert removed_any                                              # the restriction actually fired
    # balance 5 sits in the joint prefix but is NOT declared joint (its argmax is feeding)
    assert 5 in set(np.where(qbc["joint"] <= a + 1e-12)[0])
    assert str(labels[5]) != "joint"
    # argmax is taken over the three active classes only: none-dominant balance 9 still
    # points at joint, exactly the subtlety a refactor breaking mutual-exclusivity would hit
    assert argmax[9] == _CLASSES.index("joint")


def test_classfdr_discovery_gate_only_shrinks_declarations():
    fake = _fake_from_targets(_TARGETS)
    on = classfdr_labels(fake, alpha=0.10, outcome_F="F", outcome_D="D",
                         require_discovery=True)
    off = classfdr_labels(fake, alpha=0.10, outcome_F="F", outcome_D="D",
                          require_discovery=False)
    d_on = {k for k in range(len(on["labels"])) if str(on["labels"][k]) in _CLASSES}
    d_off = {k for k in range(len(off["labels"])) if str(off["labels"][k]) in _CLASSES}
    assert d_on <= d_off                                            # gate can only remove


def test_classfdr_alpha_guard_requires_alpha_le_half():
    fake = _fake_from_targets([(.9, 0, 0), (0, .9, 0)])
    classfdr_labels(fake, alpha=0.5, require_discovery=False)        # boundary alpha=1/2 allowed
    for bad in (0.0, 0.7, 1.0):                                      # <=0 or >1/2 must raise
        with pytest.raises(ValueError):
            classfdr_labels(fake, alpha=bad, require_discovery=False)


def test_classfdr_margin_gate_demotes_low_margin():
    # abstention widening: a balance whose winning class barely edges the runner-up
    # (margin ~0.04) is declared at gate 0 but demoted to 'uncertain' at a gate above its
    # margin, while high-margin balances are untouched. The gate can only remove.
    fake = _fake_from_targets([(.995, 0, 0)] * 6 + [(.48, .44, 0)])
    lo = classfdr_labels(fake, alpha=0.10, outcome_F="F", outcome_D="D",
                         require_discovery=False, margin_gate=0.0)
    hi = classfdr_labels(fake, alpha=0.10, outcome_F="F", outcome_D="D",
                         require_discovery=False, margin_gate=0.5)
    d_lo = {k for k in range(len(lo["labels"])) if str(lo["labels"][k]) in _CLASSES}
    d_hi = {k for k in range(len(hi["labels"])) if str(hi["labels"][k]) in _CLASSES}
    assert d_hi <= d_lo                                    # the gate only removes declarations
    assert lo["class_margin"][6] < 0.5 and 6 in d_lo       # bal 6 declared at gate 0 ...
    assert 6 not in d_hi                                   # ... and gated out at margin 0.5
