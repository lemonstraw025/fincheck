"""Benchmark the detector against synthetic ground truth.

Compares two operating regimes on the same labelled dataset:

* **anomaly-only** — IsolationForest over node2vec + structural features.
* **fused** — anomaly combined with the structural pattern detectors via
  noisy-OR (the full fincheck pipeline).

For each regime we sweep the decision threshold, find the operating point whose
precision is closest to a target (default 0.83), and report the recall there.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --target-precision 0.83 --background 4300
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from fincheck.data.synthetic import SyntheticConfig, generate_transactions
from fincheck.graph.builder import build_graph
from fincheck.graph.patterns import detect_all_patterns
from fincheck.ml.anomaly import AnomalyModel
from fincheck.scoring.fusion import noisy_or


def _precision_recall(labels: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[float, float]:
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _recall_at_precision(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> Tuple[float, float, float]:
    """Return (threshold, precision, recall) closest to target precision."""
    best = (0.5, 0.0, 0.0)
    best_gap = 1e9
    for thr in np.linspace(scores.min(), scores.max(), 200):
        p, r = _precision_recall(labels, scores, thr)
        if p + 1e-9 >= target:
            # Prefer the highest recall among points that meet precision.
            if r > best[2] or (best_gap > 0):
                best = (float(thr), p, r)
                best_gap = 0
        elif best_gap > 0:
            gap = abs(p - target)
            if gap < best_gap:
                best_gap = gap
                best = (float(thr), p, r)
    return best


def run(background: int, seed: int, target_precision: float) -> Dict[str, dict]:
    cfg = SyntheticConfig(n_background=background, seed=seed)
    txns = generate_transactions(cfg)
    g = build_graph(txns)

    labels_map = {t.id: int(t.is_laundering) for t in txns}
    txids = [t.id for t in txns]
    labels = np.array([labels_map[t] for t in txids])

    # Anomaly-only scores.
    model = AnomalyModel(seed=seed)
    anomaly = model.transaction_scores(g)
    anomaly_scores = np.array([anomaly.get(t, 0.0) for t in txids])

    # Fused scores.
    hits = detect_all_patterns(g)
    pattern_scores: Dict[str, List[float]] = {}
    for hit in hits:
        for txid in hit.txids:
            pattern_scores.setdefault(txid, []).append(hit.score)
    fused_scores = np.array(
        [noisy_or([anomaly.get(t, 0.0)] + pattern_scores.get(t, [])) for t in txids]
    )

    out = {}
    for name, scores in (("anomaly_only", anomaly_scores), ("fused", fused_scores)):
        thr, p, r = _recall_at_precision(labels, scores, target_precision)
        out[name] = {
            "threshold": round(thr, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "met_target": bool(p + 1e-9 >= target_precision),
        }
    out["_meta"] = {
        "transactions": len(txids),
        "laundering": int(labels.sum()),
        "target_precision": target_precision,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background", type=int, default=4300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-precision", type=float, default=0.83)
    args = ap.parse_args()

    res = run(args.background, args.seed, args.target_precision)
    meta = res["_meta"]
    print(
        f"Dataset: {meta['transactions']:,} transactions, "
        f"{meta['laundering']} laundering  "
        f"(target precision {meta['target_precision']:.0%})\n"
    )
    for name in ("anomaly_only", "fused"):
        m = res[name]
        note = "" if m["met_target"] else "  [target precision unreachable; nearest point]"
        print(
            f"  {name:<13}  precision={m['precision']:.2%}  "
            f"recall={m['recall']:.2%}  (thr={m['threshold']}){note}"
        )
    delta = res["fused"]["recall"] - res["anomaly_only"]["recall"]
    print(f"\n  recall lift (fused - anomaly): {delta:+.2%}")


if __name__ == "__main__":
    main()
