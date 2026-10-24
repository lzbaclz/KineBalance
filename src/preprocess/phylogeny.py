"""Reference phylogeny for log-ratio balances: the GTDB bacterial tree, pruned to genera.

The curated cohorts assign reads to GTDB R07-RS207 genera, so every genus id is a
lineage string ``d__Bacteria;p__...;g__X`` that also labels the GTDB reference tree
(``bac120_r207.tree``, one leaf per species representative; leaf-to-taxonomy map
``bac120_taxonomy_r207.tsv``). :func:`gtdb_genus_tree` places each genus at the most
recent common ancestor (MRCA) of its species representatives and returns the induced
rooted binary tree on the requested genera. Its internal nodes are clades of the
reference phylogeny, so the balances are PhILR balances on a tree that no cohort's
data have touched. ``scripts/get_gtdb.sh`` downloads the two files (SHA-256 pinned).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:  # pragma: no cover - networkx is part of the base stack
    _HAVE_NX = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
GTDB_DIR = os.environ.get("KINEBALANCE_GTDB_DIR", os.path.join(REPO_ROOT, "data", "gtdb"))
GTDB_TREE = os.path.join(GTDB_DIR, "bac120_r207.tree")
GTDB_TAXONOMY = os.path.join(GTDB_DIR, "bac120_taxonomy_r207.tsv")
GENUS_RANKS = 6          # d, p, c, o, f, g

__all__ = ["parse_newick", "gtdb_genus_tree", "to_newick"]


def parse_newick(text: str) -> Tuple[int, List[str], List[int], List[List[int]]]:
    """Parse a Newick string into ``(root, names, parent, children)`` index arrays.

    Branch lengths are skipped. Quoted labels (``'100.0:c__Bacilli'``, as GTDB writes
    support values and taxon names) and unquoted labels are both read verbatim.
    """
    s = text.strip()
    n = len(s)
    names: List[str] = []
    parent: List[int] = []
    children: List[List[int]] = []

    def add(par: int) -> int:
        k = len(names)
        names.append("")
        parent.append(par)
        children.append([])
        if par >= 0:
            children[par].append(k)
        return k

    def label(pos: int) -> Tuple[str, int]:
        if s[pos] == "'":
            j, buf = pos + 1, []
            while True:
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    return "".join(buf), j + 1
                buf.append(s[j])
                j += 1
        j = pos
        while j < n and s[j] not in "(),:;":
            j += 1
        return s[pos:j].strip(), j

    pos, cur, root = 0, -1, -1
    while pos < n:
        c = s[pos]
        if c == "(":
            k = add(cur)
            root = k if root < 0 else root
            cur = k
            pos += 1
        elif c == ")":
            done, cur = cur, parent[cur]
            pos += 1
            if pos < n and s[pos] not in ",);:":
                names[done], pos = label(pos)
        elif c == ":":                       # branch length: skip
            pos += 1
            while pos < n and s[pos] not in ",();":
                pos += 1
        elif c == ";":
            break
        elif c == "," or c.isspace():
            pos += 1
        else:
            k = add(cur)
            names[k], pos = label(pos)
            root = k if root < 0 else root
    if root < 0:
        raise ValueError("parse_newick: empty tree")
    return root, names, parent, children


@lru_cache(maxsize=2)
def _load_gtdb(tree_path: str, taxonomy_path: str):
    for p in (tree_path, taxonomy_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {p!r}; run `bash scripts/get_gtdb.sh` first.")
    with open(tree_path) as fh:
        root, names, parent, children = parse_newick(fh.read())
    leaf_of = {names[k]: k for k in range(len(names)) if not children[k]}
    genus_leaves: Dict[str, List[int]] = {}
    with open(taxonomy_path) as fh:
        for line in fh:
            acc, lineage = line.rstrip("\n").split("\t")
            k = leaf_of.get(acc)
            if k is not None:
                g = ";".join(lineage.split(";")[:GENUS_RANKS])
                genus_leaves.setdefault(g, []).append(k)
    # leaves per node (post-order over an explicit stack)
    n_leaf = np.zeros(len(names), dtype=np.int64)
    order, stack = [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children[v])
    for v in reversed(order):
        n_leaf[v] = 1 if not children[v] else sum(int(n_leaf[c]) for c in children[v])
    return root, names, parent, children, genus_leaves, n_leaf


def _mrca(leaves: Sequence[int], parent: List[int]) -> int:
    paths = []
    for v in leaves:
        p = []
        while v >= 0:
            p.append(v)
            v = parent[v]
        paths.append(p[::-1])                 # root first
    m = paths[0]
    for p in paths[1:]:
        i = 0
        while i < len(m) and i < len(p) and m[i] == p[i]:
            i += 1
        m = m[:i]
    return m[-1]


def _taxon_label(name: str) -> str:
    """'100.0:c__Bacilli; o__X' -> 'c__Bacilli; o__X' ('' for a bare support value)."""
    lab = name.split(":", 1)[1] if ":" in name else name
    return lab.strip() if "__" in lab else ""


def gtdb_genus_tree(taxa_ids: Sequence[str], tree_path: str = GTDB_TREE,
                    taxonomy_path: str = GTDB_TAXONOMY) -> "nx.DiGraph":
    """Induced rooted binary GTDB tree on the given genus lineages.

    Each genus sits at the MRCA of its species representatives in the reference tree;
    the GTDB genera used here must be monophyletic there (checked). Internal nodes
    with a single child that leads to a requested genus are collapsed. Node ids are
    ``"gtdb:<index>"``; a node attribute ``taxon`` keeps the GTDB clade label (empty
    for unnamed clades). Leaves are the ``taxa_ids`` strings.
    """
    if not _HAVE_NX:  # pragma: no cover
        raise ImportError("networkx is required to build the GTDB tree.")
    root, names, parent, children, genus_leaves, n_leaf = _load_gtdb(tree_path, taxonomy_path)
    ids = [str(t) for t in taxa_ids]
    missing = [t for t in ids if t not in genus_leaves]
    if missing:
        raise KeyError(f"gtdb_genus_tree: {len(missing)} genera not in the reference tree, "
                       f"e.g. {missing[:3]}")
    node_of: Dict[int, str] = {}
    for t in ids:
        v = _mrca(genus_leaves[t], parent)
        if int(n_leaf[v]) != len(genus_leaves[t]):
            raise ValueError(f"gtdb_genus_tree: genus {t!r} is not monophyletic in the tree")
        if v in node_of:
            raise ValueError(f"gtdb_genus_tree: {t!r} and {node_of[v]!r} share a node")
        node_of[v] = t
    # mark every ancestor of a requested genus
    has = np.zeros(len(names), dtype=bool)
    for v in node_of:
        u = v
        while u >= 0 and not has[u]:
            has[u] = True
            u = parent[u]
    g = nx.DiGraph()
    out: Dict[int, str] = {}
    order, stack = [], [root]
    while stack:                               # pre-order over marked nodes only
        v = stack.pop()
        order.append(v)
        if v not in node_of:
            stack.extend(c for c in children[v] if has[c])
    for v in reversed(order):                  # children before parents
        if v in node_of:
            out[v] = node_of[v]
            g.add_node(out[v])
            continue
        kids = [out[c] for c in children[v] if has[c]]
        if len(kids) == 1:                     # pass-through: collapse, keeping the
            out[v] = kids[0]                   # clade name on the retained node
            tx = _taxon_label(names[v])
            if tx and kids[0] in g and g.out_degree(kids[0]) > 0:
                old_tx = g.nodes[kids[0]].get("taxon", "")
                g.nodes[kids[0]]["taxon"] = f"{old_tx}; {tx}" if old_tx else tx
            continue
        cur = kids[0]
        for j, kid in enumerate(kids[1:], start=1):   # ladder if a polytomy remains
            nid = f"gtdb:{v}" if j == len(kids) - 1 else f"gtdb:{v}|{j}"
            g.add_edge(nid, cur)
            g.add_edge(nid, kid)
            g.nodes[nid]["taxon"] = _taxon_label(names[v]) if j == len(kids) - 1 else ""
            cur = nid
        out[v] = cur
    return g


def to_newick(tree: "nx.DiGraph", short: bool = True) -> str:
    """Newick string of a rooted DiGraph (leaf names shortened to the genus if ``short``)."""
    root = [v for v in tree.nodes if tree.in_degree(v) == 0][0]

    def lab(v: str) -> str:
        s = v.split(";")[-1] if short else v
        return "'" + s.replace("'", "''") + "'"

    out: Dict[str, str] = {}
    order, stack = [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(tree.successors(v))
    for v in reversed(order):
        kids = list(tree.successors(v))
        if not kids:
            out[v] = lab(v)
        else:
            tx = tree.nodes[v].get("taxon", "")
            out[v] = "(" + ",".join(out[c] for c in kids) + ")" + (
                "'" + tx.replace("'", "''") + "'" if tx else "")
    return out[root] + ";"
