"""Build a NetworkX directed multigraph from transactions.

Nodes are accounts; edges are individual transactions. Parallel edges between
the same pair of accounts are preserved (``MultiDiGraph``) so that aggregate
flow and individual hops can both be reasoned about.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx

from fincheck.data.schema import Transaction


def build_graph(transactions: Iterable[Transaction]) -> nx.MultiDiGraph:
    """Construct a transaction ``MultiDiGraph``.

    Each edge stores ``amount``, ``timestamp``, ``txid`` and the ground-truth
    ``is_laundering`` flag. Node attributes accumulate in/out totals and counts.
    """
    g = nx.MultiDiGraph()
    for tx in transactions:
        if not g.has_node(tx.src):
            g.add_node(tx.src, out_total=0.0, in_total=0.0, out_count=0, in_count=0)
        if not g.has_node(tx.dst):
            g.add_node(tx.dst, out_total=0.0, in_total=0.0, out_count=0, in_count=0)

        g.add_edge(
            tx.src,
            tx.dst,
            key=tx.id,
            txid=tx.id,
            amount=float(tx.amount),
            timestamp=tx.timestamp,
            is_laundering=bool(tx.is_laundering),
        )
        g.nodes[tx.src]["out_total"] += float(tx.amount)
        g.nodes[tx.src]["out_count"] += 1
        g.nodes[tx.dst]["in_total"] += float(tx.amount)
        g.nodes[tx.dst]["in_count"] += 1
    return g


def simple_digraph(g: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse the multigraph into a weighted ``DiGraph``.

    Edge ``weight`` is the summed amount and ``count`` the number of hops.
    Useful for structural algorithms (cycle search, embeddings) that do not
    need individual transactions.
    """
    simple = nx.DiGraph()
    simple.add_nodes_from(g.nodes(data=True))
    agg: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for u, v, data in g.edges(data=True):
        total, count = agg.get((u, v), (0.0, 0))
        agg[(u, v)] = (total + data["amount"], count + 1)
    for (u, v), (total, count) in agg.items():
        simple.add_edge(u, v, weight=total, count=count)
    return simple


def edge_frame(g: nx.MultiDiGraph) -> List[dict]:
    """Return edges as a list of plain dicts (handy for serialization/tests)."""
    rows: List[dict] = []
    for u, v, data in g.edges(data=True):
        rows.append(
            {
                "txid": data["txid"],
                "src": u,
                "dst": v,
                "amount": data["amount"],
                "timestamp": data["timestamp"],
                "is_laundering": data["is_laundering"],
            }
        )
    return rows
