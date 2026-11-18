"""Tests for graph construction."""

from fincheck.graph.builder import build_graph, edge_frame, simple_digraph


def test_build_nodes_and_edges(cycle_txns):
    g = build_graph(cycle_txns)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 4


def test_node_totals(cycle_txns):
    g = build_graph(cycle_txns)
    # A sends 10_000 (to B) and receives 9_850 (from D).
    assert g.nodes["A"]["out_total"] == 10_000
    assert g.nodes["A"]["in_total"] == 9_850
    assert g.nodes["A"]["out_count"] == 1
    assert g.nodes["A"]["in_count"] == 1


def test_edge_attributes(cycle_txns):
    g = build_graph(cycle_txns)
    data = g["A"]["B"]["c1"]
    assert data["txid"] == "c1"
    assert data["amount"] == 10_000
    assert data["is_laundering"] is True


def test_simple_digraph_aggregates(fanout_txns):
    g = build_graph(fanout_txns)
    simple = simple_digraph(g)
    # Hub fans out to 10 distinct accounts => 10 simple edges.
    assert simple.out_degree("HUB") == 10
    for _, _, d in simple.edges(data=True):
        assert d["count"] == 1
        assert d["weight"] > 0


def test_edge_frame_shape(cycle_txns):
    g = build_graph(cycle_txns)
    rows = edge_frame(g)
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"txid", "src", "dst", "amount", "timestamp", "is_laundering"}


def test_parallel_edges_preserved(make_tx):
    g = build_graph([make_tx("a", "X", "Y", 1, 1), make_tx("b", "X", "Y", 2, 2)])
    assert g.number_of_edges() == 2
    assert simple_digraph(g)["X"]["Y"]["count"] == 2
