"""Tests for the DFS/BFS structural pattern detectors."""

from fincheck.graph.builder import build_graph
from fincheck.graph.patterns import (
    PatternHit,
    detect_all_patterns,
    detect_cycles,
    detect_fan_outs,
    detect_peeling_chains,
)


def test_detect_cycle(cycle_txns):
    g = build_graph(cycle_txns)
    hits = detect_cycles(g)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.pattern == "cycle"
    assert set(hit.nodes) == {"A", "B", "C", "D"}
    assert len(hit.txids) == 4
    assert 0.0 <= hit.score <= 1.0


def test_cycle_score_bounds(make_tx):
    g = build_graph(
        [
            make_tx("1", "A", "B", 5000, 1),
            make_tx("2", "B", "C", 5000, 2),
            make_tx("3", "C", "A", 5000, 3),
        ]
    )
    hits = detect_cycles(g)
    assert hits and all(0.0 <= h.score <= 1.0 for h in hits)


def test_no_cycle_on_linear_chain(make_tx):
    g = build_graph(
        [
            make_tx("1", "A", "B", 5000, 1),
            make_tx("2", "B", "C", 5000, 2),
            make_tx("3", "C", "D", 5000, 3),
        ]
    )
    assert detect_cycles(g) == []


def test_detect_fan_out(fanout_txns):
    g = build_graph(fanout_txns)
    hits = detect_fan_outs(g, min_fanout=8)
    assert len(hits) == 1
    assert hits[0].pattern == "fan_out"
    assert hits[0].nodes[0] == "HUB"
    assert hits[0].detail["fanout"] >= 8


def test_fan_out_requires_minimum(make_tx):
    g = build_graph([make_tx(f"f{i}", "HUB", f"M{i}", 100, i) for i in range(3)])
    assert detect_fan_outs(g, min_fanout=8) == []


def test_detect_peeling_chain(peeling_txns):
    g = build_graph(peeling_txns)
    hits = detect_peeling_chains(g, min_len=4)
    assert hits
    best = max(hits, key=lambda h: len(h.txids))
    assert best.pattern == "peeling_chain"
    assert len(best.txids) >= 4


def test_peeling_chain_rejects_short(make_tx):
    g = build_graph([make_tx("1", "A", "B", 1000, 1), make_tx("2", "B", "C", 900, 2)])
    assert detect_peeling_chains(g, min_len=4) == []


def test_detect_all_combines(cycle_txns, fanout_txns):
    g = build_graph(cycle_txns + fanout_txns)
    hits = detect_all_patterns(g)
    patterns = {h.pattern for h in hits}
    assert "cycle" in patterns
    assert "fan_out" in patterns


def test_pattern_hit_clamps_score():
    hit = PatternHit(pattern="cycle", nodes=["A"], txids=["x"], score=5.0)
    assert hit.score == 1.0
    hit2 = PatternHit(pattern="cycle", nodes=["A"], txids=["x"], score=-2.0)
    assert hit2.score == 0.0
