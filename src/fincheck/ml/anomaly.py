"""Unsupervised anomaly scoring with IsolationForest over node features.

Each account is represented by its node2vec embedding concatenated with
structural features. An :class:`sklearn.ensemble.IsolationForest` assigns an
anomaly score, which is squashed to a probability in ``[0, 1]``. A transaction
inherits the maximum anomaly probability of its two endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fincheck.config import settings
from fincheck.ml.features import Node2Vec, structural_features


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AnomalyModel:
    """node2vec + IsolationForest anomaly detector."""

    def __init__(
        self,
        contamination: float = settings.contamination,
        embedding_dim: int = settings.embedding_dim,
        seed: int = settings.seed,
    ) -> None:
        self.contamination = contamination
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.scaler = StandardScaler()
        self.forest: Optional[IsolationForest] = None
        self.n2v: Optional[Node2Vec] = None
        self.node_scores: Dict[str, float] = {}

    def _node_matrix(self, g: nx.MultiDiGraph):
        self.n2v = Node2Vec(
            dim=self.embedding_dim,
            walk_length=settings.walk_length,
            num_walks=settings.num_walks,
            p=settings.p_return,
            q=settings.q_inout,
            window=settings.window,
            seed=self.seed,
        ).fit(g)
        nodes, struct = structural_features(g)
        emb = np.vstack([self.n2v.vector(n) for n in nodes]) if nodes else np.empty((0, self.embedding_dim))
        X = np.hstack([emb, struct]) if len(nodes) else np.empty((0, self.embedding_dim + 6))
        return nodes, X

    def fit_score(self, g: nx.MultiDiGraph) -> Dict[str, float]:
        """Fit the model on the graph and return per-node anomaly probabilities."""
        nodes, X = self._node_matrix(g)
        if len(nodes) == 0:
            self.node_scores = {}
            return {}

        Xs = self.scaler.fit_transform(X)
        self.forest = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.forest.fit(Xs)
        # score_samples: higher => more normal. Negate so higher => anomalous,
        # then standardise and squash to a probability.
        raw = -self.forest.score_samples(Xs)
        centred = (raw - raw.mean()) / (raw.std() + 1e-9)
        probs = _sigmoid(centred)
        self.node_scores = {n: float(p) for n, p in zip(nodes, probs)}
        return self.node_scores

    def transaction_scores(self, g: nx.MultiDiGraph) -> Dict[str, float]:
        """Map node anomaly probabilities onto transactions (max of endpoints)."""
        if not self.node_scores:
            self.fit_score(g)
        tx_scores: Dict[str, float] = {}
        for u, v, data in g.edges(data=True):
            s = max(self.node_scores.get(u, 0.0), self.node_scores.get(v, 0.0))
            tx_scores[data["txid"]] = s
        return tx_scores
