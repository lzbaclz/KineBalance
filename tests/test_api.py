"""Tests for the high-level API (kinebalance.balances / kinebalance.screen) and the CLI."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

import kinebalance as kb

LINEAGES = [
    "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Blautia",
    "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Roseburia",
    "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Faecalibacterium",
    "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus",
    "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Enterococcaceae;g__Enterococcus",
    "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides",
    "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola",
    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
]


def _counts(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, n).astype(float)
    logits = rng.normal(0, 1, (n, len(LINEAGES)))
    logits[:, 3] += 1.2 * X                                    # Streptococcus up with exposure
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    C = np.vstack([rng.multinomial(5000, p) for p in P])
    ids = [f"s{i}" for i in range(n)]
    return pd.DataFrame(C, index=ids, columns=LINEAGES), X, P


def test_balances_bases_are_complete_ilr():
    counts, _, _ = _counts()
    for basis in ("ward", "taxonomy-ladder", "taxonomy-balanced"):
        B, info = kb.balances(counts, basis=basis, top_n=8)
        assert B.shape == (counts.shape[0], len(LINEAGES) - 1)
        part = info["partition"]
        assert len(part) == B.shape[1]
        assert all(set(a.split(";")).isdisjoint(b.split(";"))
                   for a, b in zip(part["numerator"], part["denominator"]))
    B, info = kb.balances(counts, basis="clr", top_n=8)
    assert B.shape == counts.shape and np.allclose(B.sum(axis=1), 0)
    with pytest.raises(ValueError):
        kb.balances(counts, basis="nope")


def test_screen_two_endpoints_finds_joint_mediator():
    rng = np.random.default_rng(3)
    n, K = 200, 6
    X = rng.standard_normal(n)
    B = rng.standard_normal((n, K)); B[:, 0] += 0.9 * X
    Y = {"F": 0.8 * B[:, 0] + rng.standard_normal(n), "D": 0.8 * B[:, 0] + rng.standard_normal(n)}
    res = kb.screen(X, pd.DataFrame(B, columns=[f"c{j}" for j in range(K)]), Y,
                    prior="decoupled", chains=2, n_iter=1200, burn=600)
    t = res.table
    assert list(t["coordinate"]) == [f"c{j}" for j in range(K)]
    assert t.loc[0, "label"] == "joint" and t.loc[0, "class_post"] > 0.8
    assert (t.loc[1:, "label"] == "none").all()
    assert {"pip_F", "pip_D", "q_F", "q_D", "P_joint", "ess_min", "rhat_max"} <= set(t.columns)
    assert "joint" in res.summary()


def test_screen_rejects_missing_values():
    X = np.zeros(30); X[:15] = 1
    B = np.random.default_rng(0).standard_normal((30, 3)); B[0, 0] = np.nan
    with pytest.raises(ValueError):
        kb.screen(X, B, {"F": np.ones(30)}, chains=1, n_iter=50, burn=10)


def test_cli_smoke(tmp_path):
    from kinebalance.cli import main
    counts, X, P = _counts(seed=5)
    rng = np.random.default_rng(5)
    samples = pd.DataFrame({"exposure": X,
                            "dca": np.exp(2 + 2 * np.log(P[:, 3] / P[:, 0]).clip(-5, 5) / 5
                                          + rng.normal(0, 0.3, len(X))),
                            "prop": np.exp(1 + rng.normal(0, 0.3, len(X))),
                            "age": rng.normal(50, 10, len(X))}, index=counts.index)
    counts.to_csv(tmp_path / "counts.tsv", sep="\t")
    samples.to_csv(tmp_path / "samples.csv")
    out = tmp_path / "res" / "kb"
    rc = main(["--counts", str(tmp_path / "counts.tsv"), "--samples", str(tmp_path / "samples.csv"),
               "--exposure", "exposure", "--endpoints", "dca", "prop", "--covariates", "age",
               "--log1p", "--basis", "taxonomy-balanced", "--top-n", "8", "--prior", "both",
               "--chains", "2", "--n-iter", "300", "--burn", "100", "--out", str(out)])
    assert rc == 0
    for f in ("kb_decoupled.csv", "kb_coupled.csv", "kb_partition.csv"):
        assert (tmp_path / "res" / f).exists()
    t = pd.read_csv(tmp_path / "res" / "kb_decoupled.csv")
    assert len(t) == len(LINEAGES) - 1 and "label" in t.columns
