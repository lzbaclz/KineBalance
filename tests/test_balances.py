"""Tests for the PhILR balance transform.

Validates :mod:`src.preprocess.balances`:

* :func:`philr_balances` returns an ``(n, K-1)`` ILR embedding for ``K`` taxa;
* the per-node balance contrasts are (weighted) orthonormal -- their Gram
  matrix in clr space is diagonal up to the leaf-weighting, i.e. the identity
  for a balanced binary tree -- so the balances form a near-orthogonal
  coordinate system on the simplex;
* :func:`multiplicative_replacement` removes all zeros while keeping every row
  closed (summing to 1).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import networkx as nx
import pytest

from src.preprocess.balances import (
    build_sbp_from_tree,
    closure,
    multiplicative_replacement,
    philr_balances,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _balanced_binary_tree_4() -> "nx.DiGraph":
    """A rooted balanced binary tree on 4 leaves: ((0,1),(2,3))."""
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("root", "a"),
            ("root", "b"),
            ("a", 0),
            ("a", 1),
            ("b", 2),
            ("b", 3),
        ]
    )
    return g


def _contrast_matrix(sbp, n_taxa: int) -> np.ndarray:
    """Reconstruct the (K-1, K) clr-space contrast matrix Psi from an SBP.

    Row ``j`` is the PhILR balance vector of node ``j``: weight ``+w/r`` on the
    ``r`` numerator leaves and ``-w/s`` on the ``s`` denominator leaves, with
    ``w = sqrt(r s / (r + s))``. For a balanced binary tree these
    rows are orthonormal.
    """
    psi = np.zeros((len(sbp), n_taxa), dtype=float)
    for j, node in enumerate(sbp):
        r = len(node.left)
        s = len(node.right)
        w = np.sqrt(r * s / (r + s))
        psi[j, list(node.left)] = w / r
        psi[j, list(node.right)] = -w / s
    return psi


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #
def test_philr_balances_shape_explicit_tree():
    """For K taxa philr_balances returns an (n, K-1) embedding."""
    rng = np.random.default_rng(0)
    n, K = 20, 4
    counts = rng.integers(1, 100, size=(n, K)).astype(float)
    tree = _balanced_binary_tree_4()

    B, sbp, labels = philr_balances(counts, tree=tree, taxa_ids=[0, 1, 2, 3])

    assert B.shape == (n, K - 1)
    assert len(sbp) == K - 1
    assert len(labels) == K - 1
    assert np.all(np.isfinite(B))


def test_philr_balances_shape_ward_fallback():
    """The Ward-clustering fallback (tree=None) also yields (n, K-1)."""
    rng = np.random.default_rng(1)
    n, K = 16, 8
    counts = rng.integers(1, 50, size=(n, K)).astype(float)

    B, sbp, labels = philr_balances(counts, tree=None, seed=3)

    assert B.shape == (n, K - 1)
    assert len(sbp) == K - 1
    assert np.all(np.isfinite(B))


# --------------------------------------------------------------------------- #
# Orthonormality of the balance contrasts
# --------------------------------------------------------------------------- #
def test_balance_contrasts_orthonormal_gram_identity():
    """Balanced-binary-tree contrasts are orthonormal: Gram(Psi) == I.

    The ``K-1`` PhILR balance vectors form an isometric log-ratio basis, so in
    clr space their Gram matrix is diagonal (orthogonal contrasts) and, for the
    equal-weighting of a balanced binary tree, exactly the identity.
    """
    taxa = [0, 1, 2, 3]
    sbp = build_sbp_from_tree(_balanced_binary_tree_4(), taxa)
    psi = _contrast_matrix(sbp, n_taxa=len(taxa))

    gram = psi @ psi.T
    off_diag = gram - np.diag(np.diag(gram))

    # Diagonal ~ 1 (unit-norm contrasts), off-diagonal ~ 0 (mutually orthogonal).
    assert np.allclose(np.diag(gram), 1.0, atol=1e-10)
    assert np.max(np.abs(off_diag)) < 1e-10


def test_balances_equal_clr_projection():
    """B equals the clr-transformed composition projected onto Psi^T.

    This ties the public output of :func:`philr_balances` to the orthonormal
    contrast basis, confirming the balances really are the ILR coordinates.
    """
    rng = np.random.default_rng(2)
    n, K = 25, 4
    counts = rng.integers(1, 200, size=(n, K)).astype(float)
    tree = _balanced_binary_tree_4()

    B, sbp, _ = philr_balances(counts, tree=tree, taxa_ids=[0, 1, 2, 3])
    psi = _contrast_matrix(sbp, n_taxa=K)

    comp = multiplicative_replacement(counts)
    log_comp = np.log(comp)
    clr = log_comp - log_comp.mean(axis=1, keepdims=True)

    assert np.allclose(B, clr @ psi.T, atol=1e-10)


def test_gram_near_diagonal_general_tree():
    """A caterpillar (unbalanced) tree still gives orthogonal contrasts.

    For an unbalanced tree the diagonal entries need not all equal 1, but the
    Gram matrix is still (near-)diagonal: the off-diagonal mass is negligible
    relative to the diagonal, i.e. the contrasts remain mutually orthogonal.
    """
    # Caterpillar: (((0,1),2),3)
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("root", "c"),
            ("root", 3),
            ("c", "b"),
            ("c", 2),
            ("b", 0),
            ("b", 1),
        ]
    )
    taxa = [0, 1, 2, 3]
    sbp = build_sbp_from_tree(g, taxa)
    psi = _contrast_matrix(sbp, n_taxa=len(taxa))

    gram = psi @ psi.T
    off = np.abs(gram - np.diag(np.diag(gram)))
    diag = np.abs(np.diag(gram))

    # Contrasts are unit norm and mutually orthogonal regardless of balance.
    assert np.allclose(diag, 1.0, atol=1e-10)
    assert np.max(off) < 1e-10


# --------------------------------------------------------------------------- #
# Multiplicative replacement
# --------------------------------------------------------------------------- #
def test_multiplicative_replacement_removes_zeros_and_closes():
    """Zeros are removed and every row still sums to 1."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 20, size=(15, 6)).astype(float)
    # Force structural zeros, including an all-but-one-zero row.
    counts[0, :3] = 0.0
    counts[5, 1:] = 0.0

    out = multiplicative_replacement(counts)

    assert not np.any(out == 0.0)
    assert np.all(out > 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out.shape == counts.shape


def test_multiplicative_replacement_1d_input():
    """A single composition (1-D input) is handled and re-closed."""
    x = np.array([0.0, 3.0, 0.0, 7.0])
    out = multiplicative_replacement(x)

    assert out.shape == x.shape
    assert np.all(out > 0.0)
    assert np.isclose(out.sum(), 1.0)


def test_closure_rejects_negative_and_zero_rows():
    """closure() guards against negative values and empty rows."""
    with pytest.raises(ValueError):
        closure(np.array([[1.0, -1.0, 2.0]]))
    with pytest.raises(ValueError):
        closure(np.array([[0.0, 0.0, 0.0]]))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
