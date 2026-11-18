"""Tests for plain-English explanations."""

from fincheck.scoring.explain import explain_alert
from fincheck.scoring.fusion import Alert


def _alert(**kw):
    base = dict(txid="t1", src="A", dst="B", amount=12345.0, score=0.9)
    base.update(kw)
    return Alert(**base)


def test_explanation_mentions_txid_and_amount():
    text = explain_alert(_alert())
    assert "t1" in text
    assert "12,345" in text


def test_explanation_includes_cycle_reason():
    a = _alert(patterns=[{"pattern": "cycle", "length": 4.0, "total_amount": 40000.0}])
    text = explain_alert(a)
    assert "loop" in text
    assert "4 hops" in text


def test_explanation_includes_fanout_reason():
    a = _alert(patterns=[{"pattern": "fan_out", "fanout": 12.0}])
    assert "12 recipients" in explain_alert(a)


def test_explanation_includes_anomaly_signal():
    a = _alert(patterns=[], signals={"anomaly": 0.8})
    assert "anomaly" in explain_alert(a)


def test_explanation_confidence_words():
    assert "very high" in explain_alert(_alert(score=0.95))
    assert "moderate" in explain_alert(_alert(score=0.6))


def test_explanation_anomaly_only_fallback():
    a = _alert(patterns=[], signals={"anomaly": 0.3}, score=0.3)
    assert "anomaly alone" in explain_alert(a)
