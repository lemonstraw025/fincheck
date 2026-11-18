"""node2vec embeddings and structural node features.

A compact, dependency-light node2vec implementation: biased second-order random
walks (controlled by return parameter ``p`` and in-out parameter ``q``) feed a
skip-gram model trained with negative sampling in NumPy. Embeddings are
deterministic given a seed.

The goal is not to beat a tuned gensim pipeline but to produce stable,
structure-aware node vectors that feed the downstream anomaly model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from fincheck.graph.builder import simple_digraph


def _undirected_weighted(g: nx.MultiDiGraph) -> nx.Graph:
    simple = simple_digraph(g)
    und = nx.Graph()
    und.add_nodes_from(simple.nodes())
    for u, v, data in simple.edges(data=True):
        w = data.get("weight", 1.0)
        if und.has_edge(u, v):
            und[u][v]["weight"] += w
        else:
            und.add_edge(u, v, weight=w)
    return und


class Node2Vec:
    """Train node2vec embeddings on a transaction graph."""

    def __init__(
        self,
        dim: int = 32,
        walk_length: int = 20,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 0.5,
        window: int = 5,
        epochs: int = 3,
        lr: float = 0.025,
        negatives: int = 5,
        seed: int = 7,
    ) -> None:
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.negatives = negatives
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.nodes: List[str] = []
        self.index: Dict[str, int] = {}
        self.embeddings: np.ndarray = np.empty((0, dim))

    # -- random walks -------------------------------------------------------
    def _walk(self, g: nx.Graph, start: str) -> List[str]:
        walk = [start]
        if g.degree(start) == 0:
            return walk
        # First step: uniform over neighbours (weighted).
        walk.append(self._weighted_choice(g, start))
        while len(walk) < self.walk_length:
            prev, cur = walk[-2], walk[-1]
            nbrs = list(g.neighbors(cur))
            if not nbrs:
                break
            weights = np.empty(len(nbrs))
            prev_nbrs = set(g.neighbors(prev))
            for i, x in enumerate(nbrs):
                ew = g[cur][x].get("weight", 1.0)
                if x == prev:
                    weights[i] = ew / self.p
                elif x in prev_nbrs:
                    weights[i] = ew
                else:
                    weights[i] = ew / self.q
            weights = weights / weights.sum()
            walk.append(nbrs[self._rng.choice(len(nbrs), p=weights)])
        return walk

    def _weighted_choice(self, g: nx.Graph, node: str) -> str:
        nbrs = list(g.neighbors(node))
        weights = np.array([g[node][x].get("weight", 1.0) for x in nbrs], dtype=float)
        weights = weights / weights.sum()
        return nbrs[self._rng.choice(len(nbrs), p=weights)]

    def _generate_walks(self, g: nx.Graph) -> List[List[int]]:
        walks: List[List[int]] = []
        nodes = list(g.nodes())
        for _ in range(self.num_walks):
            self._rng.shuffle(nodes)
            for node in nodes:
                walk = self._walk(g, node)
                walks.append([self.index[n] for n in walk])
        return walks

    # -- skip-gram with negative sampling -----------------------------------
    def fit(self, g: nx.MultiDiGraph) -> "Node2Vec":
        und = _undirected_weighted(g)
        self.nodes = list(und.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        if n == 0:
            self.embeddings = np.empty((0, self.dim))
            return self

        # Xavier-ish init for reproducibility.
        self._emb_in = (self._rng.standard_normal((n, self.dim)) * 0.1)
        self._emb_out = np.zeros((n, self.dim))

        walks = self._generate_walks(und)

        # Unigram (negative sampling) distribution ^0.75.
        freq = np.ones(n)
        for w in walks:
            for idx in w:
                freq[idx] += 1
        noise = freq ** 0.75
        noise = noise / noise.sum()

        for _ in range(self.epochs):
            self._rng.shuffle(walks)
            for walk in walks:
                self._train_walk(walk, noise)

        # L2-normalise so downstream distances are scale-free.
        norms = np.linalg.norm(self._emb_in, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self._emb_in / norms
        return self

    def _train_walk(self, walk: List[int], noise: np.ndarray) -> None:
        L = len(walk)
        for i, center in enumerate(walk):
            lo = max(0, i - self.window)
            hi = min(L, i + self.window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                context = walk[j]
                self._sgd_step(center, context, noise, label=1.0)
                negs = self._rng.choice(len(noise), size=self.negatives, p=noise)
                for neg in negs:
                    if neg == context:
                        continue
                    self._sgd_step(center, int(neg), noise, label=0.0)

    def _sgd_step(self, center: int, context: int, noise: np.ndarray, label: float) -> None:
        v = self._emb_in[center]
        u = self._emb_out[context]
        score = float(np.dot(v, u))
        score = max(-30.0, min(30.0, score))  # guard against exp overflow
        pred = 1.0 / (1.0 + np.exp(-score))
        grad = (pred - label) * self.lr
        self._emb_in[center] = v - grad * u
        self._emb_out[context] = u - grad * v

    def vector(self, node: str) -> np.ndarray:
        if node not in self.index:
            return np.zeros(self.dim)
        return self.embeddings[self.index[node]]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {n: self.embeddings[i] for n, i in self.index.items()}


def structural_features(g: nx.MultiDiGraph) -> Tuple[List[str], np.ndarray]:
    """Return per-node structural features complementing the embeddings.

    Features: in/out degree, in/out amount totals (log1p), amount ratio, and a
    net-flow term. These capture hub/mule behaviour that embeddings alone miss.
    """
    nodes = list(g.nodes())
    feats = np.zeros((len(nodes), 6))
    for i, node in enumerate(nodes):
        attrs = g.nodes[node]
        in_total = attrs.get("in_total", 0.0)
        out_total = attrs.get("out_total", 0.0)
        in_count = attrs.get("in_count", 0)
        out_count = attrs.get("out_count", 0)
        feats[i] = [
            in_count,
            out_count,
            np.log1p(in_total),
            np.log1p(out_total),
            out_total / (in_total + 1.0),
            np.log1p(abs(in_total - out_total)),
        ]
    return nodes, feats
