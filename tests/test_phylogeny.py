"""Tests for the fixed trees: GTDB rank hierarchy (taxonomy_tree) and the GTDB phylogeny."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.preprocess.balances import philr_balances, taxonomy_tree
from src.preprocess.phylogeny import GTDB_TAXONOMY, GTDB_TREE, gtdb_genus_tree, parse_newick, to_newick
from src.simulate.clade_plasmode import sbp_loadings

TAXA = [
    "d__Bacteria;p__A;c__A1;o__A11;f__A111;g__x",
    "d__Bacteria;p__A;c__A1;o__A11;f__A111;g__y",
    "d__Bacteria;p__A;c__A1;o__A11;f__A112;g__z",
    "d__Bacteria;p__A;c__A1;o__A11;f__A113;g__w",
    "d__Bacteria;p__B;c__B1;o__B11;f__B111;g__v",       # single-child chain: collapsed
    "d__Bacteria;p__C;c__C1;o__C11;f__C111;g__u",
    "d__Bacteria;p__C;c__C1;o__C11;f__C111;g__t",
]


@pytest.mark.parametrize("rule", ["ladder", "balanced"])
def test_taxonomy_tree_is_a_complete_binary_basis(rule):
    g = taxonomy_tree(TAXA, resolve=rule)
    internal = [v for v in g if g.out_degree(v) > 0]
    assert all(g.out_degree(v) == 2 for v in internal)
    assert len(internal) == len(TAXA) - 1
    assert sorted(v for v in g if g.out_degree(v) == 0) == sorted(TAXA)
    counts = np.random.default_rng(0).integers(1, 50, (20, len(TAXA))).astype(float)
    B, sbp, _ = philr_balances(counts, tree=g, taxa_ids=TAXA)
    L = sbp_loadings(sbp, len(TAXA))
    assert B.shape == (20, len(TAXA) - 1) and np.allclose(L @ L.T, np.eye(len(TAXA) - 1))
    # taxon nodes keep their lineage prefix: family A111 splits x | y
    a111 = "d__Bacteria;p__A;c__A1;o__A11;f__A111"
    assert a111 in g and set(g.successors(a111)) == {TAXA[0], TAXA[1]}


def test_taxonomy_tree_rules_differ_only_inside_polytomies():
    lad, bal = taxonomy_tree(TAXA, "ladder"), taxonomy_tree(TAXA, "balanced")
    fam = "d__Bacteria;p__C;c__C1;o__C11;f__C111"          # a binary taxon: same in both
    assert set(lad.successors(fam)) == set(bal.successors(fam))
    with pytest.raises(ValueError):
        taxonomy_tree(TAXA, "random")
    with pytest.raises(ValueError):
        taxonomy_tree(TAXA[:1])


def test_parse_newick_quoted_labels_and_lengths():
    root, names, parent, children = parse_newick(
        "((A:0.1,B:0.2)'100.0:g__X':0.3,(C:1,D:2)95:0.5)'root lab';")
    leaves = sorted(names[k] for k in range(len(names)) if not children[k])
    assert leaves == ["A", "B", "C", "D"]
    assert names[root] == "root lab" and parent[root] == -1
    labs = {names[k] for k in range(len(names)) if children[k]}
    assert "100.0:g__X" in labs and "95" in labs


@pytest.mark.skipif(not (os.path.isfile(GTDB_TREE) and os.path.isfile(GTDB_TAXONOMY)),
                    reason="GTDB r207 files not downloaded (bash scripts/get_gtdb.sh)")
def test_gtdb_genus_tree_on_real_genera():
    gen = ["d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus",
           "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Enterococcaceae;g__Enterococcus_B",
           "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Blautia",
           "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides",
           "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia"]
    g = gtdb_genus_tree(gen)
    internal = [v for v in g if g.out_degree(v) > 0]
    assert len(internal) == len(gen) - 1 and all(g.out_degree(v) == 2 for v in internal)
    nwk = to_newick(g)
    # the two Lactobacillales genera are sisters in the reference tree
    assert "('g__Streptococcus','g__Enterococcus_B')" in nwk or "('g__Enterococcus_B','g__Streptococcus')" in nwk
    with pytest.raises(KeyError):
        gtdb_genus_tree(gen[:2] + ["d__Bacteria;p__X;c__X;o__X;f__X;g__NotAGenus"])
