"""Turn an :class:`~fincheck.scoring.fusion.Alert` into a plain-English note.

Explanations are deterministic templates driven by the alert's signals and
pattern details, so an analyst can read why a transaction was flagged without
knowing the model internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fincheck.scoring.fusion import Alert


_PATTERN_PHRASES = {
    "cycle": "sits on a closed loop where funds return to their origin",
    "fan_out": "is part of a burst that splits one balance across many accounts",
    "peeling_chain": "forwards most of an incoming balance down a long chain",
}


def _confidence_word(score: float) -> str:
    if score >= 0.85:
        return "very high"
    if score >= 0.7:
        return "high"
    if score >= 0.55:
        return "moderate"
    return "low"


def explain_alert(alert: "Alert") -> str:
    """Compose a human-readable explanation for an alert."""
    parts: list = []
    parts.append(
        f"Transaction {alert.txid} (${alert.amount:,.2f} from {alert.src} to "
        f"{alert.dst}) was flagged with {_confidence_word(alert.score)} confidence "
        f"({alert.score:.0%})."
    )

    reasons: list = []
    for pat in alert.patterns:
        phrase = _PATTERN_PHRASES.get(pat["pattern"])
        if not phrase:
            continue
        detail_bits = []
        if "length" in pat:
            detail_bits.append(f"{int(pat['length'])} hops")
        if "fanout" in pat:
            detail_bits.append(f"{int(pat['fanout'])} recipients")
        if "total_amount" in pat:
            detail_bits.append(f"${pat['total_amount']:,.0f} total")
        suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        reasons.append(f"it {phrase}{suffix}")

    anomaly = alert.signals.get("anomaly")
    if anomaly is not None and anomaly >= 0.6:
        reasons.append(
            f"its accounts behave unusually versus peers (anomaly {anomaly:.0%})"
        )

    if reasons:
        if len(reasons) == 1:
            parts.append(f"The main driver: {reasons[0]}.")
        else:
            joined = "; ".join(reasons[:-1]) + f"; and {reasons[-1]}"
            parts.append(f"Contributing signals: {joined}.")
    else:
        parts.append("The signal is driven by statistical anomaly alone.")

    return " ".join(parts)
