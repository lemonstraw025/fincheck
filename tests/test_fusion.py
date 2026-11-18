"""Tests for noisy-OR fusion, ranking and the alert engine."""

import pytest

from fincheck.scoring.fusion import Alert, AlertEngine, noisy_or


def test_noisy_or_single():
    assert noisy_or([0.5]) == pytest.approx(0.5)


def test_noisy_or_combines():
    # 1 - (1-0.5)(1-0.5) = 0.75
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)


def test_noisy_or_clamps():
    assert noisy_or([1.5, -0.2]) == pytest.approx(1.0)


def test_noisy_or_empty():
    assert noisy_or([]) == 0.0


def test_alert_to_dict():
    a = Alert(txid="t", src="A", dst="B", amount=100.0, score=0.9, signals={"anomaly": 0.9})
    d = a.to_dict()
    assert d["txid"] == "t"
    assert d["score"] == 0.9
    assert "anomaly" in d["signals"]


def test_rank_orders_descending():
    alerts = [
        Alert(txid="a", src="A", dst="B", amount=1, score=0.3),
        Alert(txid="b", src="A", dst="B", amount=1, score=0.9),
        Alert(txid="c", src="A", dst="B", amount=1, score=0.6),
    ]
    ranked = AlertEngine.rank(alerts)
    assert [a.txid for a in ranked] == ["b", "c", "a"]


def test_rank_top_k():
    alerts = [Alert(txid=str(i), src="A", dst="B", amount=1, score=i / 10) for i in range(10)]
    ranked = AlertEngine.rank(alerts, top_k=3)
    assert len(ranked) == 3
    assert ranked[0].score == pytest.approx(0.9)


def test_engine_flags_cycle(cycle_txns):
    engine = AlertEngine(threshold=0.3)
    alerts = engine.run(cycle_txns)
    assert alerts
    assert all(0.0 <= a.score <= 1.0 for a in alerts)
    assert any(p["pattern"] == "cycle" for a in alerts for p in a.patterns)


def test_engine_threshold_filters(cycle_txns):
    low = AlertEngine(threshold=0.0).run(cycle_txns)
    high = AlertEngine(threshold=0.99).run(cycle_txns)
    assert len(high) <= len(low)


def test_engine_explanations_present(fanout_txns):
    alerts = AlertEngine(threshold=0.3).run(fanout_txns)
    assert alerts
    assert all(a.explanation for a in alerts)
