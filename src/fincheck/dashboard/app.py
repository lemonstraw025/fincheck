"""Streamlit dashboard for the fincheck fraud-detection service.

Run with::

    streamlit run src/fincheck/dashboard/app.py

The dashboard talks to the FastAPI backend over HTTP (configure the base URL in
the sidebar). It lets an analyst generate synthetic data, launch analysis, and
review ranked alerts with their plain-English explanations and the subgraph of
accounts involved.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_API = os.getenv("FINCHECK_API_URL", "http://localhost:8000")


def _get(base: str, path: str, **params) -> Dict[str, Any]:
    r = requests.get(f"{base}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _post(base: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{base}{path}", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def _score_color(score: float) -> str:
    if score >= 0.85:
        return "🔴"
    if score >= 0.7:
        return "🟠"
    return "🟡"


def main() -> None:
    st.set_page_config(page_title="fincheck", page_icon="🕸️", layout="wide")
    st.title("🕸️ fincheck — money-laundering detection")
    st.caption(
        "Graph pattern detection + ML anomaly, fused into ranked, explainable alerts."
    )

    with st.sidebar:
        st.header("Connection")
        base = st.text_input("API base URL", DEFAULT_API)
        try:
            health = _get(base, "/health")
            st.success(f"Connected · {health.get('service')}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"API unreachable: {exc}")
            st.stop()

        st.header("Synthetic data")
        n_bg = st.number_input("Background transactions", 500, 50_000, 4300, step=100)
        seed = st.number_input("Seed", 0, 9999, 7)
        if st.button("Generate dataset", use_container_width=True):
            with st.spinner("Generating…"):
                res = _post(
                    base,
                    "/generate",
                    {"n_background": int(n_bg), "seed": int(seed), "replace": True},
                )
            st.success(
                f"Generated {res['generated']} tx "
                f"({res['labelled_laundering']} laundering)"
            )

    stats = _get(base, "/stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{stats['transactions']:,}")
    c2.metric("Labelled laundering", f"{stats['labelled_laundering']:,}")
    c3.metric("Jobs run", stats["jobs"])
    c4.metric("Stored alerts", stats["alerts"])

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Run analysis")
        threshold = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)
        top_k = st.slider("Top alerts to show", 5, 100, 25, 5)
        run = st.button("Analyze now", type="primary", use_container_width=True)

    if run:
        with st.spinner("Scoring transactions…"):
            result = _post(
                base, "/analyze", {"threshold": threshold, "top_k": int(top_k)}
            )
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    with right:
        st.subheader("Summary")
        if result:
            a, b, c = st.columns(3)
            a.metric("Analyzed", f"{result['analyzed']:,}")
            b.metric("Alerts", result["alerts"])
            c.metric("Pattern hits", result["pattern_hits"])
        else:
            st.info("Generate data (sidebar) then click **Analyze now**.")

    if not result:
        return

    st.divider()
    st.subheader("🚨 Ranked alerts")
    alerts: List[Dict[str, Any]] = result["top"]

    pattern_filter = st.multiselect(
        "Filter by pattern",
        ["cycle", "fan_out", "peeling_chain"],
        default=[],
    )

    for alert in alerts:
        pats = {p["pattern"] for p in alert.get("patterns", [])}
        if pattern_filter and not pats.intersection(pattern_filter):
            continue
        badge = _score_color(alert["score"])
        header = (
            f"{badge} {alert['score']:.0%} · {alert['txid']} · "
            f"${alert['amount']:,.0f} · {alert['src']} → {alert['dst']}"
        )
        with st.expander(header):
            st.write(alert["explanation"])
            cols = st.columns(2)
            with cols[0]:
                st.caption("Signals")
                st.json(alert.get("signals", {}))
            with cols[1]:
                st.caption("Patterns")
                st.json(alert.get("patterns", []))


if __name__ == "__main__":
    main()
