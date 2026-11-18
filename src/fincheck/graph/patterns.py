"""Structural money-laundering pattern detectors (DFS / BFS based).

Three motif families are detected directly on the transaction graph:

* :func:`detect_cycles`        — bounded-depth DFS cycle search.
* :func:`detect_fan_outs`      — BFS one-hop expansion from high out-degree hubs.
* :func:`detect_peeling_chains`— DFS longest decreasing-amount path search.

Each detector returns :class:`PatternHit` objects carrying the transaction ids
it covers and a heuristic confidence in ``[0, 1]``. These confidences are later
combined with the ML anomaly signal by the scoring layer.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from fincheck.graph.builder import simple_digraph


@dataclass
class PatternHit:
    """A single structural detection."""

    pattern: str                      # "cycle" | "fan_out" | "peeling_chain"
    nodes: List[str]                  # accounts involved, in traversal order
    txids: List[str]                  # transactions covered by this hit
    score: float                      # confidence in [0, 1]
    detail: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))


def _rep_edge(g: nx.MultiDiGraph, u: str, v: str):
    """Return (txid, amount, timestamp) of the largest transaction on u -> v."""
    best = None
    best_amt = -1.0
    if not g.has_edge(u, v):
        return None, 0.0, None
    for key, data in g[u][v].items():
        if data["amount"] > best_amt:
            best_amt = data["amount"]
            best = (data["txid"], data["amount"], data["timestamp"])
    return best if best is not None else (None, 0.0, None)


# --------------------------------------------------------------------------- #
# Cycles — bounded DFS                                                          #
# --------------------------------------------------------------------------- #
def detect_cycles(
    g: nx.MultiDiGraph,
    max_len: int = 6,
    min_mean_amount: float = 3_000.0,
    min_balance: float = 0.5,
    max_span_hours: float = 72.0,
) -> List[PatternHit]:
    """Find directed cycles up to ``max_len`` nodes via bounded DFS.

    Because a dense transaction graph contains many *incidental* short cycles,
    a cycle is only reported when it looks deliberate: its hop amounts are large
    and balanced (money returns close to its origin) **and** the hops occur
    within a tight time window (``max_span_hours``). These filters separate
    laundering rings from coincidental loops in background traffic. Cycles are
    de-duplicated by a canonical rotation of their node sequence.
    """
    simple = simple_digraph(g)
    seen: set = set()
    hits: List[PatternHit] = []

    nodes = list(simple.nodes())

    def canonical(cycle: List[str]) -> Tuple[str, ...]:
        # Rotate so the lexicographically smallest node is first.
        i = min(range(len(cycle)), key=lambda k: cycle[k])
        rotated = cycle[i:] + cycle[:i]
        return tuple(rotated)

    def dfs(start: str, current: str, path: List[str]) -> None:
        for nxt in simple.successors(current):
            if nxt == start and len(path) >= 3:
                key = canonical(path)
                if key not in seen:
                    seen.add(key)
                    _record_cycle(
                        g, list(key), hits, min_mean_amount, min_balance, max_span_hours
                    )
            elif nxt not in path and len(path) < max_len and nxt > start:
                # ``nxt > start`` keeps each cycle discovered from its smallest
                # node, pruning rotations early.
                path.append(nxt)
                dfs(start, nxt, path)
                path.pop()

    for node in nodes:
        dfs(node, node, [node])
    return hits


def _record_cycle(
    g: nx.MultiDiGraph,
    cycle: List[str],
    hits: List[PatternHit],
    min_mean_amount: float,
    min_balance: float,
    max_span_hours: float,
) -> None:
    amounts: List[float] = []
    txids: List[str] = []
    times: List = []
    length = len(cycle)
    for i in range(length):
        u = cycle[i]
        v = cycle[(i + 1) % length]
        txid, amt, ts = _rep_edge(g, u, v)
        if txid is None:
            return
        txids.append(txid)
        amounts.append(amt)
        times.append(ts)

    if not amounts:
        return
    mean_amt = statistics.mean(amounts)
    if mean_amt < min_mean_amount:
        return

    # Temporal cohesion: deliberate rings move fast; incidental loops are spread
    # across the whole history.
    span_hours = (max(times) - min(times)).total_seconds() / 3600.0
    if span_hours > max_span_hours:
        return

    total = sum(amounts)
    # Balance: low relative spread around the ring => strong laundering signal.
    spread = (statistics.pstdev(amounts) / mean_amt) if mean_amt else 1.0
    balance = max(0.0, 1.0 - spread)
    if balance < min_balance:
        return

    # Larger rings, tighter timing and larger balanced flows are more suspicious.
    size_factor = min(1.0, length / 6.0)
    value_factor = min(1.0, total / 100_000.0)
    recency_factor = max(0.0, 1.0 - span_hours / max_span_hours)
    score = (
        0.45 * balance
        + 0.20 * size_factor
        + 0.15 * value_factor
        + 0.20 * recency_factor
    )
    hits.append(
        PatternHit(
            pattern="cycle",
            nodes=cycle,
            txids=txids,
            score=score,
            detail={
                "length": float(length),
                "total_amount": round(total, 2),
                "balance": round(balance, 3),
                "span_hours": round(span_hours, 2),
            },
        )
    )


# --------------------------------------------------------------------------- #
# Fan-outs — BFS one-hop expansion                                             #
# --------------------------------------------------------------------------- #
def detect_fan_outs(
    g: nx.MultiDiGraph,
    min_fanout: int = 8,
    window_minutes: float = 60.0,
) -> List[PatternHit]:
    """Detect structuring hubs that disburse to many accounts in a short window.

    For every node a one-hop BFS collects outgoing transactions; if enough land
    on distinct destinations within ``window_minutes`` and their amounts are
    tightly clustered (low coefficient of variation), the burst is flagged.
    """
    hits: List[PatternHit] = []
    for hub in g.nodes():
        out_edges = [
            (data["timestamp"], data["txid"], v, data["amount"])
            for _, v, data in g.out_edges(hub, data=True)
        ]
        if len(out_edges) < min_fanout:
            continue
        out_edges.sort(key=lambda e: e[0])

        # Slide a time window across the sorted bursts.
        n = len(out_edges)
        i = 0
        while i < n:
            j = i
            window_tx: List[Tuple[str, str, float]] = []
            while j < n and _minutes(out_edges[i][0], out_edges[j][0]) <= window_minutes:
                window_tx.append((out_edges[j][1], out_edges[j][2], out_edges[j][3]))
                j += 1
            distinct = {dst for _, dst, _ in window_tx}
            if len(distinct) >= min_fanout:
                amounts = [a for _, _, a in window_tx]
                mean_amt = statistics.mean(amounts)
                cov = (statistics.pstdev(amounts) / mean_amt) if mean_amt else 1.0
                similarity = max(0.0, 1.0 - cov)  # structured => uniform slices
                size_factor = min(1.0, len(distinct) / 20.0)
                score = 0.6 * similarity + 0.4 * size_factor
                hits.append(
                    PatternHit(
                        pattern="fan_out",
                        nodes=[hub] + sorted(distinct),
                        txids=[t for t, _, _ in window_tx],
                        score=score,
                        detail={
                            "fanout": float(len(distinct)),
                            "mean_amount": round(mean_amt, 2),
                            "uniformity": round(similarity, 3),
                        },
                    )
                )
                i = j  # non-overlapping windows
            else:
                i += 1
    return hits


# --------------------------------------------------------------------------- #
# Peeling chains — DFS longest decreasing path                                 #
# --------------------------------------------------------------------------- #
def detect_peeling_chains(
    g: nx.MultiDiGraph,
    min_len: int = 4,
    peel_low: float = 0.80,
    peel_high: float = 0.995,
    min_mean_amount: float = 5_000.0,
    max_span_hours: float = 72.0,
    max_depth: int = 24,
) -> List[PatternHit]:
    """Detect peeling chains via DFS over time-ordered decreasing transfers.

    A peeling chain is a path where each hop forwards most (but not all) of the
    incoming amount — the ratio ``next / prev`` sits in ``[peel_low, peel_high]``
    — and timestamps strictly increase. DFS extends each chain greedily and the
    maximal chains are reported. As with cycles, a chain must carry material
    value (``min_mean_amount``) and stay within a time window
    (``max_span_hours``) to be reported, which suppresses coincidental
    decreasing runs in background traffic.
    """
    hits: List[PatternHit] = []
    covered: set = set()

    # Pre-index out edges sorted by time for quick extension.
    def extend(chain_txids: List[str], chain_nodes: List[str], last_amt: float,
               last_time, node: str, ratios: List[float],
               amounts: List[float], times: List) -> None:
        best_extended = False
        if len(chain_nodes) < max_depth:
            for _, v, data in g.out_edges(node, data=True):
                amt = data["amount"]
                if data["timestamp"] <= last_time:
                    continue
                if v in chain_nodes:
                    continue
                ratio = amt / last_amt if last_amt else 0.0
                if peel_low <= ratio <= peel_high:
                    best_extended = True
                    extend(
                        chain_txids + [data["txid"]],
                        chain_nodes + [v],
                        amt,
                        data["timestamp"],
                        v,
                        ratios + [ratio],
                        amounts + [amt],
                        times + [data["timestamp"]],
                    )
        if not best_extended:
            _emit_chain(
                chain_txids, chain_nodes, ratios, amounts, times, hits, covered,
                min_len, min_mean_amount, max_span_hours,
            )

    for start in g.nodes():
        for _, v, data in g.out_edges(start, data=True):
            extend(
                [data["txid"]],
                [start, v],
                data["amount"],
                data["timestamp"],
                v,
                [],
                [data["amount"]],
                [data["timestamp"]],
            )
    return hits


def _emit_chain(
    txids, nodes, ratios, amounts, times, hits, covered, min_len,
    min_mean_amount, max_span_hours,
) -> None:
    if len(txids) < min_len:
        return
    key = tuple(txids)
    if key in covered:
        return
    # Suppress chains that are a prefix subset already emitted.
    covered.add(key)
    if statistics.mean(amounts) < min_mean_amount:
        return
    span_hours = (max(times) - min(times)).total_seconds() / 3600.0
    if span_hours > max_span_hours:
        return
    consistency = 1.0 - statistics.pstdev(ratios) if len(ratios) > 1 else 0.6
    consistency = max(0.0, min(1.0, consistency))
    length_factor = min(1.0, len(txids) / 8.0)
    score = 0.5 * consistency + 0.5 * length_factor
    hits.append(
        PatternHit(
            pattern="peeling_chain",
            nodes=list(nodes),
            txids=list(txids),
            score=score,
            detail={
                "length": float(len(txids)),
                "consistency": round(consistency, 3),
            },
        )
    )


def detect_all_patterns(g: nx.MultiDiGraph) -> List[PatternHit]:
    """Run every structural detector and return the combined hit list."""
    hits: List[PatternHit] = []
    hits.extend(detect_cycles(g))
    hits.extend(detect_fan_outs(g))
    hits.extend(detect_peeling_chains(g))
    return hits


def _minutes(a, b) -> float:
    return abs((b - a).total_seconds()) / 60.0
