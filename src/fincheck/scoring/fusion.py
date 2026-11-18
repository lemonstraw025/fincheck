"""Combine structural and ML signals into ranked, explainable alerts.

Every flagged transaction accumulates independent evidence:

* the IsolationForest anomaly probability of its endpoints, and
* a confidence from each structural pattern (cycle / fan-out / peeling chain)
  that covers it.

These are fused with a **noisy-OR** — ``P = 1 - prod(1 - p_i)`` — which treats
each signal as an independent chance of true suspicion, so multiple weak cues
combine into a strong one without ever exceeding 1. Alerts at or above the
configured threshold are ranked with a **max-heap** (:mod:`heapq`).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import networkx as nx

from fincheck.config import settings
from fincheck.data.schema import Transaction
from fincheck.graph.builder import build_graph
from fincheck.graph.patterns import PatternHit, detect_all_patterns
from fincheck.ml.anomaly import AnomalyModel


def noisy_or(probs: Iterable[float]) -> float:
    """Fuse independent probabilities via noisy-OR, clamped to ``[0, 1]``."""
    prod = 1.0
    for p in probs:
        p = min(1.0, max(0.0, float(p)))
        prod *= (1.0 - p)
    return 1.0 - prod


@dataclass
class Alert:
    """A ranked, explainable suspicion about one transaction."""

    txid: str
    src: str
    dst: str
    amount: float
    score: float
    signals: Dict[str, float] = field(default_factory=dict)
    patterns: List[dict] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "txid": self.txid,
            "src": self.src,
            "dst": self.dst,
            "amount": self.amount,
            "score": round(self.score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "patterns": self.patterns,
            "explanation": self.explanation,
        }


class AlertEngine:
    """Run the full detection pipeline and emit ranked alerts."""

    def __init__(
        self,
        threshold: float = settings.alert_threshold,
        anomaly_model: Optional[AnomalyModel] = None,
    ) -> None:
        self.threshold = threshold
        self.anomaly_model = anomaly_model or AnomalyModel()
        self.graph: Optional[nx.MultiDiGraph] = None
        self.pattern_hits: List[PatternHit] = []

    def run(self, transactions: List[Transaction]) -> List[Alert]:
        """Analyse transactions and return alerts ranked most-suspicious first."""
        from fincheck.scoring.explain import explain_alert  # avoid cycle

        g = build_graph(transactions)
        self.graph = g
        self.pattern_hits = detect_all_patterns(g)
        anomaly = self.anomaly_model.transaction_scores(g)

        # Index edge metadata for alert construction.
        meta: Dict[str, dict] = {}
        for u, v, data in g.edges(data=True):
            meta[data["txid"]] = {"src": u, "dst": v, "amount": data["amount"]}

        # Collect per-transaction signals.
        signals: Dict[str, Dict[str, float]] = {}
        patterns: Dict[str, List[dict]] = {}

        for txid, score in anomaly.items():
            signals.setdefault(txid, {})["anomaly"] = score

        for hit in self.pattern_hits:
            for txid in hit.txids:
                signals.setdefault(txid, {})[hit.pattern] = max(
                    signals.setdefault(txid, {}).get(hit.pattern, 0.0), hit.score
                )
                patterns.setdefault(txid, []).append(
                    {"pattern": hit.pattern, "score": round(hit.score, 4), **hit.detail}
                )

        alerts: List[Alert] = []
        for txid, sig in signals.items():
            fused = noisy_or(sig.values())
            if fused < self.threshold:
                continue
            m = meta.get(txid, {"src": "?", "dst": "?", "amount": 0.0})
            alert = Alert(
                txid=txid,
                src=m["src"],
                dst=m["dst"],
                amount=m["amount"],
                score=fused,
                signals=sig,
                patterns=patterns.get(txid, []),
            )
            alert.explanation = explain_alert(alert)
            alerts.append(alert)

        return self.rank(alerts)

    @staticmethod
    def rank(alerts: List[Alert], top_k: Optional[int] = None) -> List[Alert]:
        """Rank alerts by fused score using a heap.

        A max-heap is emulated via :func:`heapq.nlargest`, keyed on
        ``(score, amount)`` so ties break toward larger transactions.
        """
        if top_k is None:
            top_k = len(alerts)
        return heapq.nlargest(top_k, alerts, key=lambda a: (a.score, a.amount))
