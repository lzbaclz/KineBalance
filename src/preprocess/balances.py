"""Phylogenetic ILR / PhILR balances.

Implements the phylogenetic isometric-log-ratio (PhILR) balance transform:

    B_ib = sqrt( r_b * s_b / (r_b + s_b) )
           * log( g(M_i, R_b) / g(M_i, S_b) )

where, for internal node ``b`` of a rooted bifurcating tree:

* ``R_b`` and ``S_b`` are the leaf sets of the two child subtrees of ``b``,
* ``r_b = |R_b|`` and ``s_b = |S_b|`` are the leaf counts of those subtrees,
* ``M_i`` is the (zero-replaced, closed) composition of sample ``i``,
* ``g(M_i, R_b)`` is the geometric mean of the components of ``M_i`` restricted
  to the leaves in ``R_b``.

The resulting matrix ``B`` (n x (K-1)) is an isometric log-ratio (ILR)
embedding: the ``K-1`` balances form an orthonormal coordinate system on the
simplex, hence they are mutually (near-)orthogonal by construction.

This module depends only on numpy / scipy / networkx and deliberately avoids
scikit-bio. When no tree is supplied a bifurcating tree is constructed from a
Ward hierarchical clustering of the CLR-correlation between taxa, and its merge
order is used directly as the sequential binary partition (SBP).

References
----------
Egozcue et al. (2003), "Isometric logratio transformations for compositional
data analysis"; Silverman et al. (2017), "A phylogenetic transform enhances
analysis of compositional microbiota data" (PhILR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import ward

try:  # networkx is part of the allowed base stack; used for tree traversal.
    import networkx as nx

    _HAVE_NX = True
except Exception:  # pragma: no cover - defensive, nx is expected to be present
    _HAVE_NX = False


__all__ = [
    "closure",
    "multiplicative_replacement",
    "zero_replace",
    "geometric_mean",
    "build_sbp_from_tree",
    "taxonomy_tree",
    "philr_balances",
    "filter_balances",
    "mrf_tree_graph",
    "SBPNode",
]


# ---------------------------------------------------------------------------
# Compositional helpers
# ---------------------------------------------------------------------------
def closure(counts: np.ndarray) -> np.ndarray:
    """Close a count/abundance matrix so each row sums to 1.

    Parameters
    ----------
    counts : numpy.ndarray
        Array of shape ``(n_samples, n_parts)`` (a single sample of shape
        ``(n_parts,)`` is also accepted). Values must be non-negative.

    Returns
    -------
    numpy.ndarray
        Array of the same shape with each row (last axis) summing to 1.

    Raises
    ------
    ValueError
        If any value is negative or any row sums to zero.
    """
    arr = np.asarray(counts, dtype=np.float64)
    reshaped = arr.ndim == 1
    if reshaped:
        arr = arr[np.newaxis, :]
    if np.any(arr < 0):
        raise ValueError("closure: counts must be non-negative.")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("closure: every row must have a positive total.")
    closed = arr / row_sums
    return closed[0] if reshaped else closed


def multiplicative_replacement(
    counts: np.ndarray, pseudo: Optional[float] = None
) -> np.ndarray:
    """Replace zeros with a small pseudo-count and re-close the composition.

    Implements the multiplicative replacement strategy: zero entries are set to
    ``pseudo`` (default ``0.5 / n_parts``), non-zero entries are scaled down to
    preserve the unit sum, and rows are re-closed to sum to 1.

    Parameters
    ----------
    counts : numpy.ndarray
        Array of shape ``(n_samples, n_parts)`` (or ``(n_parts,)``) of
        non-negative counts/abundances.
    pseudo : float, optional
        Pseudo value substituted for zeros. Defaults to ``0.5 / n_parts``.

    Returns
    -------
    numpy.ndarray
        Zero-free, row-closed array of the same shape as ``counts``.
    """
    arr = np.asarray(counts, dtype=np.float64)
    reshaped = arr.ndim == 1
    if reshaped:
        arr = arr[np.newaxis, :]
    if np.any(arr < 0):
        raise ValueError("multiplicative_replacement: counts must be non-negative.")

    n_parts = arr.shape[1]
    if pseudo is None:
        pseudo = 0.5 / n_parts
    if pseudo <= 0:
        raise ValueError("multiplicative_replacement: pseudo must be positive.")

    closed = closure(arr)
    out = np.empty_like(closed)
    for i in range(closed.shape[0]):
        row = closed[i]
        zero_mask = row == 0
        n_zero = int(zero_mask.sum())
        if n_zero == 0:
            out[i] = row
            continue
        # Mass to distribute onto the zeros, removed proportionally from the rest.
        added = n_zero * pseudo
        scaled = row * (1.0 - added)
        scaled[zero_mask] = pseudo
        out[i] = scaled
    out = closure(out)
    return out[0] if reshaped else out


def geometric_mean(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Geometric mean of strictly-positive values.

    Computed in log-space for numerical stability.

    Parameters
    ----------
    x : numpy.ndarray
        Strictly positive array.
    axis : int, optional
        Axis along which to reduce. If ``None`` the geometric mean of the whole
        flattened array is returned.

    Returns
    -------
    numpy.ndarray or float
        Geometric mean(s) along ``axis``.

    Raises
    ------
    ValueError
        If any value is non-positive.
    """
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("geometric_mean: all values must be strictly positive.")
    return np.exp(np.mean(np.log(arr), axis=axis))


# ---------------------------------------------------------------------------
# Sequential binary partition (SBP) from a tree
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SBPNode:
    """One internal node of a sequential binary partition.

    Attributes
    ----------
    label : str
        Human-readable identifier for the internal node ``b``.
    left : tuple of int
        Leaf indices (into the taxa ordering) forming the ``R_b`` subtree
        (numerator group; gets the ``+`` sign).
    right : tuple of int
        Leaf indices forming the ``S_b`` subtree (denominator group; ``-``).
    """

    label: str
    left: Tuple[int, ...]
    right: Tuple[int, ...]


def _leaf_sets_from_networkx(
    tree: "nx.Graph", taxa_ids: Sequence[Hashable]
) -> List[SBPNode]:
    """Build the SBP from a rooted bifurcating ``networkx`` tree.

    The tree is treated as rooted; the root is the unique node with in-degree 0
    for a :class:`networkx.DiGraph`, or inferred for an undirected tree. Leaves
    are matched to ``taxa_ids`` by node identity. Internal nodes are emitted in
    post-order so children precede parents.
    """
    id_to_idx = {tid: i for i, tid in enumerate(taxa_ids)}

    if tree.is_directed():
        digraph = tree
        roots = [n for n in digraph.nodes if digraph.in_degree(n) == 0]
        if len(roots) != 1:
            raise ValueError(
                "build_sbp_from_tree: directed tree must have exactly one root."
            )
        root = roots[0]
    else:
        # Root an undirected tree at an arbitrary internal node (degree > 1),
        # falling back to any node for degenerate inputs.
        internal = [n for n in tree.nodes if tree.degree(n) > 1]
        root = internal[0] if internal else next(iter(tree.nodes))
        digraph = nx.bfs_tree(tree, root)

    # Post-order DFS yielding, for each node, the sorted tuple of leaf indices.
    leaves_of: dict = {}
    order: List[Hashable] = []
    sbp: List[SBPNode] = []

    def _children(node: Hashable) -> List[Hashable]:
        return list(digraph.successors(node))

    # Iterative post-order to avoid recursion limits on large trees.
    stack: List[Tuple[Hashable, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            children = _children(node)
            if not children:
                if node not in id_to_idx:
                    raise ValueError(
                        f"build_sbp_from_tree: leaf {node!r} not present in taxa_ids."
                    )
                leaves_of[node] = (id_to_idx[node],)
            else:
                child_leaf_lists = [leaves_of[c] for c in children]
                if len(children) != 2:
                    # Collapse multifurcations: left = first child, right = rest.
                    left = tuple(sorted(child_leaf_lists[0]))
                    right_idx: List[int] = []
                    for cl in child_leaf_lists[1:]:
                        right_idx.extend(cl)
                    right = tuple(sorted(right_idx))
                else:
                    left = tuple(sorted(child_leaf_lists[0]))
                    right = tuple(sorted(child_leaf_lists[1]))
                leaves_of[node] = tuple(sorted(left + right))
                label = str(node) if not isinstance(node, int) else f"n{node}"
                sbp.append(SBPNode(label=label, left=left, right=right))
            order.append(node)
        else:
            stack.append((node, True))
            for child in _children(node):
                stack.append((child, False))

    return sbp


def _linkage_to_sbp(linkage: np.ndarray, n_leaves: int) -> List[SBPNode]:
    """Convert a scipy linkage matrix into a post-order SBP.

    Each merge in the linkage joins two clusters; their leaf sets become the
    ``R_b`` / ``S_b`` groups of one internal node. Merges are already ordered so
    that descendants precede ancestors, giving a valid post-order SBP.
    """
    cluster_leaves: dict = {i: (i,) for i in range(n_leaves)}
    sbp: List[SBPNode] = []
    for m, row in enumerate(linkage):
        a, b = int(row[0]), int(row[1])
        left = tuple(sorted(cluster_leaves[a]))
        right = tuple(sorted(cluster_leaves[b]))
        new_id = n_leaves + m
        cluster_leaves[new_id] = tuple(sorted(left + right))
        sbp.append(SBPNode(label=f"n{new_id}", left=left, right=right))
    return sbp


def zero_replace(
    counts: np.ndarray,
    pseudo: Optional[float] = None,
    pseudo_counts: Optional[float] = None,
) -> np.ndarray:
    """Zero-free closed composition under a chosen replacement strategy.

    ``pseudo_counts=None`` (default) uses :func:`multiplicative_replacement` with
    ``pseudo`` (default ``0.5 / n_parts``). A numeric ``pseudo_counts`` instead adds
    that constant to every count before closure (the common ``+0.5`` / ``+1``
    pseudo-count rule), used by the zero-replacement sensitivity analysis.
    """
    if pseudo_counts is None:
        return multiplicative_replacement(counts, pseudo=pseudo)
    if pseudo_counts <= 0:
        raise ValueError("zero_replace: pseudo_counts must be positive.")
    return closure(np.asarray(counts, dtype=np.float64) + float(pseudo_counts))


WARD_METRICS = ("corr", "variation")


def _ward_sbp(
    counts: np.ndarray, n_parts: int, seed: int, metric: str = "corr",
    pseudo: Optional[float] = None, pseudo_counts: Optional[float] = None,
) -> List[SBPNode]:
    """Build an SBP via Ward clustering of the taxa.

    ``metric="corr"`` (default) clusters on ``1 - corr`` distances between the
    CLR-transformed taxa profiles. ``metric="variation"`` clusters on the square
    root of the compositional variation matrix ``tau_ij = var(log x_i / x_j)``
    (a proportionality distance). The merge order is used as the SBP. A
    deterministic tiny jitter (seeded) breaks ties so the result is reproducible
    and well-defined even for constant taxa.
    """
    if metric not in WARD_METRICS:
        raise ValueError(f"metric must be one of {WARD_METRICS}; got {metric!r}")
    rng = np.random.default_rng(seed)
    comp = zero_replace(counts, pseudo=pseudo, pseudo_counts=pseudo_counts)
    log_comp = np.log(comp)
    clr = log_comp - log_comp.mean(axis=1, keepdims=True)  # n x K

    # Correlation between taxa (columns). Guard against zero-variance columns.
    taxa = clr.T  # K x n
    taxa = taxa + rng.normal(scale=1e-9, size=taxa.shape)
    if metric == "variation":
        cov = np.cov(taxa)
        var = np.diag(cov)
        tau = var[:, None] + var[None, :] - 2.0 * cov   # var(clr_i - clr_j)
        dist = np.sqrt(np.clip(tau, 0.0, None))
    else:
        corr = np.corrcoef(taxa)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dist = 1.0 - corr
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    iu = np.triu_indices(n_parts, k=1)
    condensed = dist[iu]
    linkage = ward(condensed)
    return _linkage_to_sbp(linkage, n_parts)


TAXONOMY_RESOLVE = ("ladder", "balanced")


def taxonomy_tree(
    taxa_ids: Sequence[str], resolve: str = "ladder", sep: str = ";"
) -> "nx.DiGraph":
    """Rooted binary tree from the taxa's lineage strings: a fixed, data-free basis.

    Each taxon id is a lineage such as ``d__Bacteria;p__Firmicutes_A;...;g__Blautia``
    (the GTDB strings of the curated cohorts). The ranks become nested groups; a group
    with a single child among ``taxa_ids`` is collapsed into that child, so every
    internal node splits at least two taxa. A group with ``c > 2`` children (a
    polytomy) is resolved into ``c - 1`` binary nodes by a fixed rule that uses only
    the child names, never the data:

    * ``"ladder"``: children sorted by name and joined as a comb
      ``(((c1, c2), c3), ...)``, so each node contrasts one child with the children
      sorted before it;
    * ``"balanced"``: children sorted by name and split recursively into halves.

    The tree has ``len(taxa_ids) - 1`` internal nodes, so its balances form a complete
    ILR basis. A node that is a taxon carries its lineage prefix as its id (after
    collapsing, the deepest name of a single-child chain); a node added to resolve a
    polytomy carries that prefix plus ``"|<tag>"``. The first successor of a node is
    the numerator of its balance.
    """
    if not _HAVE_NX:  # pragma: no cover - nx expected in environment
        raise ImportError("networkx is required to build a taxonomy tree.")
    if resolve not in TAXONOMY_RESOLVE:
        raise ValueError(f"resolve must be one of {TAXONOMY_RESOLVE}; got {resolve!r}")
    ids = [str(t) for t in taxa_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("taxonomy_tree: taxa_ids must be unique.")
    if len(ids) < 2:
        raise ValueError("taxonomy_tree: need at least two taxa.")

    # trie of the lineages: prefix -> {child token: child prefix}; leaves are the ids
    children: dict = {"": {}}
    for tid in ids:
        tokens = tid.split(sep)
        prefix = ""
        for tok in tokens[:-1]:
            nxt = tok if prefix == "" else prefix + sep + tok
            children[prefix].setdefault(tok, nxt)
            children.setdefault(nxt, {})
            prefix = nxt
        if tokens[-1] in children[prefix]:
            raise ValueError(f"taxonomy_tree: duplicate lineage {tid!r}")
        children[prefix][tokens[-1]] = ("leaf", tid)

    g = nx.DiGraph()

    def _join(nid: str, left: str, right: str) -> str:
        g.add_edge(nid, left)
        g.add_edge(nid, right)
        return nid

    def _balanced(prefix: str, kids: List[str], tag: str) -> str:
        if len(kids) == 1:
            return kids[0]
        h = (len(kids) + 1) // 2
        left = _balanced(prefix, kids[:h], tag + "a")
        right = _balanced(prefix, kids[h:], tag + "b")
        return _join(prefix if tag == "" else f"{prefix}|{tag}", left, right)

    def _emit(prefix: str) -> str:
        kids = []
        for tok in sorted(children[prefix]):
            ch = children[prefix][tok]
            if isinstance(ch, tuple):          # a leaf taxon
                g.add_node(ch[1])
                kids.append(ch[1])
            else:
                kids.append(_emit(ch))
        if len(kids) == 1:                     # single-child group: collapse
            return kids[0]
        name = prefix if prefix else "(root)"
        if resolve == "balanced":
            return _balanced(name, kids, "")
        cur = kids[0]
        for j, kid in enumerate(kids[1:], start=1):
            cur = _join(name if j == len(kids) - 1 else f"{name}|{j}", cur, kid)
        return cur

    _emit("")
    return g


def build_sbp_from_tree(
    tree: Optional[object],
    taxa_ids: Sequence[Hashable],
    counts: Optional[np.ndarray] = None,
    seed: int = 0,
    ward_metric: str = "corr",
    pseudo: Optional[float] = None,
    pseudo_counts: Optional[float] = None,
) -> List[SBPNode]:
    """Build a sequential binary partition (SBP) for the PhILR balances.

    Parameters
    ----------
    tree : networkx.Graph or networkx.DiGraph or None
        Rooted (or rootable) bifurcating phylogeny whose leaves are the taxa in
        ``taxa_ids``. If ``None`` a tree is induced from a Ward clustering of the
        CLR-correlation between taxa (``counts`` must then be supplied).
    taxa_ids : sequence
        Ordering of the taxa; leaf indices in the returned SBP refer to this
        ordering.
    counts : numpy.ndarray, optional
        Count matrix ``(n_samples, n_parts)``, required only when ``tree`` is
        ``None`` to drive the Ward clustering.
    seed : int, default 0
        Seed for the deterministic tie-breaking jitter used by the Ward
        fallback.

    Returns
    -------
    list of SBPNode
        One :class:`SBPNode` per internal node, in post-order (children before
        parents). Length is ``len(taxa_ids) - 1`` for a bifurcating tree.
    """
    n_parts = len(taxa_ids)
    if n_parts < 2:
        raise ValueError("build_sbp_from_tree: need at least two taxa.")

    if tree is None:
        if counts is None:
            raise ValueError(
                "build_sbp_from_tree: counts required when tree is None."
            )
        return _ward_sbp(np.asarray(counts, dtype=np.float64), n_parts, seed,
                         metric=ward_metric, pseudo=pseudo,
                         pseudo_counts=pseudo_counts)

    if not _HAVE_NX:  # pragma: no cover - nx expected in environment
        raise ImportError("networkx is required to parse an explicit tree.")
    if not isinstance(tree, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            "build_sbp_from_tree: tree must be a networkx Graph/DiGraph or None."
        )
    return _leaf_sets_from_networkx(tree, taxa_ids)


# ---------------------------------------------------------------------------
# PhILR balances
# ---------------------------------------------------------------------------
def philr_balances(
    counts: np.ndarray,
    tree: Optional[object] = None,
    taxa_ids: Optional[Sequence[Hashable]] = None,
    pseudo: Optional[float] = None,
    seed: int = 0,
    ward_metric: str = "corr",
    pseudo_counts: Optional[float] = None,
) -> Tuple[np.ndarray, List[SBPNode], List[str]]:
    """Compute PhILR phylogenetic ILR balances.

    For every internal node ``b`` the balance for sample ``i`` is

    .. math::

        B_{ib} = \\sqrt{\\frac{r_b s_b}{r_b + s_b}}
                 \\, \\log\\frac{g(M_i, R_b)}{g(M_i, S_b)}

    with ``r_b``/``s_b`` the number of leaves in the two child subtrees and
    ``g`` the geometric mean over the indicated leaf group.

    Parameters
    ----------
    counts : numpy.ndarray
        Count/abundance matrix of shape ``(n_samples, n_parts)``.
    tree : networkx.Graph or networkx.DiGraph or None, optional
        Phylogeny defining the SBP. If ``None`` a Ward-clustering tree is built
        from the data (see :func:`build_sbp_from_tree`).
    taxa_ids : sequence, optional
        Taxa identifiers, length ``n_parts``. Defaults to ``range(n_parts)``.
    pseudo : float, optional
        Pseudo-count for zero replacement (default ``0.5 / n_parts``).
    seed : int, default 0
        Seed propagated to the Ward fallback for reproducibility.

    Returns
    -------
    balances : numpy.ndarray
        Array of shape ``(n_samples, K-1)`` of balance coordinates, ordered as
        the returned ``sbp`` list.
    sbp : list of SBPNode
        The sequential binary partition used (post-order).
    node_labels : list of str
        The label of each internal node, aligned with the columns of
        ``balances``.
    """
    arr = np.asarray(counts, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("philr_balances: counts must be 2-D (n_samples, n_parts).")
    n_samples, n_parts = arr.shape
    if n_parts < 2:
        raise ValueError("philr_balances: need at least two taxa.")

    if taxa_ids is None:
        taxa_ids = list(range(n_parts))
    elif len(taxa_ids) != n_parts:
        raise ValueError("philr_balances: len(taxa_ids) must equal n_parts.")

    sbp = build_sbp_from_tree(tree, taxa_ids, counts=arr, seed=seed,
                              ward_metric=ward_metric, pseudo=pseudo,
                              pseudo_counts=pseudo_counts)

    # Zero-free, closed composition; balances are scale-invariant so the
    # closure constant cancels, but replacement keeps every part strictly > 0.
    comp = zero_replace(arr, pseudo=pseudo, pseudo_counts=pseudo_counts)
    log_comp = np.log(comp)  # n x K

    n_nodes = len(sbp)
    balances = np.empty((n_samples, n_nodes), dtype=np.float64)
    node_labels: List[str] = []

    for j, node in enumerate(sbp):
        left = np.fromiter(node.left, dtype=np.intp)
        right = np.fromiter(node.right, dtype=np.intp)
        r_b = left.size
        s_b = right.size
        if r_b == 0 or s_b == 0:
            raise ValueError(
                f"philr_balances: node {node.label} has an empty child subtree."
            )
        # log g(M_i, R_b) = mean over R_b of log M_i  (geometric mean in log-space).
        log_g_left = log_comp[:, left].mean(axis=1)
        log_g_right = log_comp[:, right].mean(axis=1)
        weight = np.sqrt((r_b * s_b) / (r_b + s_b))
        balances[:, j] = weight * (log_g_left - log_g_right)
        node_labels.append(node.label)

    return balances, sbp, node_labels


def filter_balances(
    B: np.ndarray, var_threshold: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop near-constant balance coordinates by variance.

    Because the balances are an isometric log-ratio transform they form a
    mutually near-orthogonal coordinate system, so dropping low-variance
    columns removes uninformative coordinates without inducing collinearity in
    the retained ones.

    Parameters
    ----------
    B : numpy.ndarray
        Balance matrix of shape ``(n_samples, K-1)``.
    var_threshold : float, default 1e-8
        Columns whose sample variance is at or below this threshold are removed.

    Returns
    -------
    B_filtered : numpy.ndarray
        Balance matrix restricted to the retained columns.
    kept_idx : numpy.ndarray
        Integer indices of the retained columns (into the original ``B``).
    """
    arr = np.asarray(B, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("filter_balances: B must be 2-D (n_samples, n_balances).")
    variances = arr.var(axis=0)
    kept_idx = np.nonzero(variances > var_threshold)[0]
    return arr[:, kept_idx], kept_idx


def mrf_tree_graph(
    sbp: Sequence["SBPNode"], kept_idx: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Adjacency of the balance graph from the SBP *tree*, not empirical correlation.

    Each balance is an internal node of the sequential binary partition. Two balances
    are connected iff their nodes are parent--child or siblings in the tree, so a
    signal distributed over a clade (a parent balance and its descendants) can be
    pooled even when the fine PhILR coordinates are not strongly pairwise correlated.

    Parent/child are found by matching leaf sets: node ``p``'s two child subtrees are
    ``p.left`` and ``p.right``; whichever equals another node's full leaf set is that
    node, and the two such children are siblings.

    Parameters
    ----------
    sbp : sequence of SBPNode
        The partition returned by :func:`philr_balances` (post-order, aligned with the
        balance columns *before* variance filtering).
    kept_idx : sequence of int, optional
        Column indices retained by :func:`filter_balances`; if given, the returned
        graph is restricted to those balances (edges to dropped nodes are removed).

    Returns
    -------
    W : numpy.ndarray
        Symmetric ``0/1`` adjacency of shape ``(K, K)`` with zero diagonal, where
        ``K = len(sbp)`` (or ``len(kept_idx)`` when filtering).
    """
    K_full = len(sbp)
    full_leaves = [frozenset(node.left) | frozenset(node.right) for node in sbp]
    index_of: dict = {}
    for i, ls in enumerate(full_leaves):
        index_of.setdefault(ls, i)                 # unique per proper tree
    W = np.zeros((K_full, K_full), dtype=float)
    for p, node in enumerate(sbp):
        ci = index_of.get(frozenset(node.left))    # internal child on the + side
        cj = index_of.get(frozenset(node.right))   # internal child on the - side
        if ci is not None and ci != p:
            W[p, ci] = W[ci, p] = 1.0              # parent--child
        if cj is not None and cj != p:
            W[p, cj] = W[cj, p] = 1.0              # parent--child
        if ci is not None and cj is not None and ci != cj:
            W[ci, cj] = W[cj, ci] = 1.0            # siblings
    if kept_idx is not None:
        idx = np.asarray(list(kept_idx), dtype=int)
        W = W[np.ix_(idx, idx)]
    np.fill_diagonal(W, 0.0)
    return W
