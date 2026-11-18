"""Synthetic transaction generator.

Produces a realistic-looking background of legitimate transfers plus a set of
injected money-laundering motifs with ground-truth labels:

* **cycles** — money flows A -> B -> C -> A, returning to its origin.
* **fan-outs** — a single account rapidly disburses to many fresh accounts
  (smurfing / structuring).
* **peeling chains** — a long chain A -> B -> C -> ... where a little is
  "peeled off" at each hop and the bulk keeps moving.

The generator is deterministic given a seed so benchmarks are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

from fincheck.data.schema import Transaction


@dataclass
class SyntheticConfig:
    """Knobs controlling the size and composition of the synthetic dataset."""

    n_accounts: int = 600
    n_background: int = 4300
    n_cycles: int = 18
    n_fan_outs: int = 14
    n_peeling_chains: int = 12
    start: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    seed: int = 7


class _IdFactory:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._n = 0

    def next(self) -> str:
        self._n += 1
        return f"{self._prefix}{self._n:06d}"


def generate_transactions(config: SyntheticConfig | None = None) -> List[Transaction]:
    """Generate a labelled synthetic transaction list.

    The returned transactions are shuffled so that laundering edges are not
    trivially contiguous. Every transaction carries ``is_laundering`` ground
    truth for evaluation.
    """
    cfg = config or SyntheticConfig()
    rng = random.Random(cfg.seed)
    tx_ids = _IdFactory("T")
    acct_ids = _IdFactory("A")

    accounts = [acct_ids.next() for _ in range(cfg.n_accounts)]
    txns: List[Transaction] = []

    def ts(offset_minutes: float) -> datetime:
        return cfg.start + timedelta(minutes=offset_minutes)

    clock = 0.0

    # --- Background legitimate traffic -----------------------------------
    for _ in range(cfg.n_background):
        src, dst = rng.sample(accounts, 2)
        amount = round(rng.lognormvariate(4.0, 1.1), 2)  # ~ tens..thousands
        clock += rng.expovariate(1 / 7.0)
        txns.append(
            Transaction(
                id=tx_ids.next(),
                src=src,
                dst=dst,
                amount=amount,
                timestamp=ts(clock),
                is_laundering=False,
            )
        )

    # --- Injected cycles --------------------------------------------------
    for _ in range(cfg.n_cycles):
        length = rng.randint(3, 6)
        ring = rng.sample(accounts, length)
        base = round(rng.uniform(8_000, 40_000), 2)
        clock += rng.uniform(30, 120)
        t = clock
        for i in range(length):
            src = ring[i]
            dst = ring[(i + 1) % length]
            # Amount decays slightly around the ring but returns near origin.
            amount = round(base * rng.uniform(0.9, 1.0), 2)
            t += rng.uniform(1, 25)
            txns.append(
                Transaction(
                    id=tx_ids.next(),
                    src=src,
                    dst=dst,
                    amount=amount,
                    timestamp=ts(t),
                    is_laundering=True,
                )
            )

    # --- Injected fan-outs (structuring) ---------------------------------
    for _ in range(cfg.n_fan_outs):
        hub = rng.choice(accounts)
        n_mules = rng.randint(8, 20)
        mules = rng.sample([a for a in accounts if a != hub], n_mules)
        total = round(rng.uniform(50_000, 150_000), 2)
        clock += rng.uniform(30, 120)
        t = clock
        for mule in mules:
            # Structured into near-equal sub-threshold slices.
            amount = round(total / n_mules * rng.uniform(0.85, 1.15), 2)
            t += rng.uniform(0.2, 3.0)
            txns.append(
                Transaction(
                    id=tx_ids.next(),
                    src=hub,
                    dst=mule,
                    amount=amount,
                    timestamp=ts(t),
                    is_laundering=True,
                )
            )

    # --- Injected peeling chains -----------------------------------------
    for _ in range(cfg.n_peeling_chains):
        length = rng.randint(5, 10)
        chain = rng.sample(accounts, length)
        remaining = round(rng.uniform(60_000, 200_000), 2)
        clock += rng.uniform(30, 120)
        t = clock
        for i in range(length - 1):
            peel = remaining * rng.uniform(0.05, 0.15)
            moved = round(remaining - peel, 2)
            t += rng.uniform(2, 40)
            txns.append(
                Transaction(
                    id=tx_ids.next(),
                    src=chain[i],
                    dst=chain[i + 1],
                    amount=moved,
                    timestamp=ts(t),
                    is_laundering=True,
                )
            )
            remaining = moved

    rng.shuffle(txns)
    return txns
