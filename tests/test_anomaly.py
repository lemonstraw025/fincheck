"""Tests for the IsolationForest anomaly model."""

from fincheck.graph.builder import build_graph
from fincheck.ml.anomaly import AnomalyModel


def test_node_scores_in_unit_interval(small_dataset):
    g = build_graph(small_dataset)
    scores = AnomalyModel(seed=7).fit_score(g)
    assert scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_transaction_scores_cover_all_edges(small_dataset):
    g = build_graph(small_dataset)
    model = AnomalyModel(seed=7)
    tx_scores = model.transaction_scores(g)
    txids = {d["txid"] for _, _, d in g.edges(data=True)}
    assert set(tx_scores.keys()) == txids


def test_empty_graph_returns_empty():
    import networkx as nx

    g = nx.MultiDiGraph()
    assert AnomalyModel(seed=7).fit_score(g) == {}


def test_determinism(small_dataset):
    g = build_graph(small_dataset)
    a = AnomalyModel(seed=7).fit_score(g)
    b = AnomalyModel(seed=7).fit_score(g)
    assert a == b
