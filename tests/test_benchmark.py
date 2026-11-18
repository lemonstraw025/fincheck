"""Tests for the benchmark harness."""

import numpy as np

from scripts.benchmark import _precision_recall, _recall_at_precision, run


def test_precision_recall_perfect():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    p, r = _precision_recall(labels, scores, thr=0.5)
    assert p == 1.0
    assert r == 1.0


def test_recall_at_precision_returns_triplet():
    labels = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.3, 0.7])
    thr, p, r = _recall_at_precision(labels, scores, target=0.8)
    assert 0.0 <= r <= 1.0
    assert 0.0 <= p <= 1.0


def test_benchmark_run_structure():
    res = run(background=300, seed=7, target_precision=0.7)
    assert set(res.keys()) == {"anomaly_only", "fused", "_meta"}
    for name in ("anomaly_only", "fused"):
        assert {"threshold", "precision", "recall"} <= set(res[name].keys())


def test_fused_recall_not_worse_than_anomaly():
    res = run(background=600, seed=7, target_precision=0.7)
    assert res["fused"]["recall"] >= res["anomaly_only"]["recall"] - 1e-6
