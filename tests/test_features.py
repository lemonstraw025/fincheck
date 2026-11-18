"""Tests for node2vec embeddings and structural features."""

import numpy as np

from fincheck.graph.builder import build_graph
from fincheck.ml.features import Node2Vec, structural_features


def test_embedding_shape(small_dataset):
    g = build_graph(small_dataset)
    n2v = Node2Vec(dim=16, num_walks=3, walk_length=8, seed=7).fit(g)
    assert n2v.embeddings.shape[1] == 16
    assert n2v.embeddings.shape[0] == g.number_of_nodes()


def test_embedding_determinism(cycle_txns):
    g = build_graph(cycle_txns)
    a = Node2Vec(dim=8, num_walks=3, walk_length=8, seed=3).fit(g)
    b = Node2Vec(dim=8, num_walks=3, walk_length=8, seed=3).fit(g)
    assert np.allclose(a.vector("A"), b.vector("A"))


def test_embedding_normalised(cycle_txns):
    g = build_graph(cycle_txns)
    n2v = Node2Vec(dim=8, num_walks=3, walk_length=8, seed=3).fit(g)
    norms = np.linalg.norm(n2v.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_unknown_node_returns_zero(cycle_txns):
    g = build_graph(cycle_txns)
    n2v = Node2Vec(dim=8, num_walks=2, walk_length=6, seed=3).fit(g)
    assert np.allclose(n2v.vector("does-not-exist"), np.zeros(8))


def test_structural_features_shape(cycle_txns):
    g = build_graph(cycle_txns)
    nodes, feats = structural_features(g)
    assert len(nodes) == 4
    assert feats.shape == (4, 6)


def test_structural_features_nonnegative_counts(fanout_txns):
    g = build_graph(fanout_txns)
    nodes, feats = structural_features(g)
    # columns 0,1 are in/out counts
    assert (feats[:, 0] >= 0).all()
    assert (feats[:, 1] >= 0).all()
