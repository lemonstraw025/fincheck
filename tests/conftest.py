"""Shared pytest fixtures.

An isolated SQLite database is configured *before* any fincheck module is
imported, so the module-level engine binds to a throwaway file.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest

# Point the app at a temp sqlite DB before fincheck.db is imported anywhere.
_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".sqlite3", prefix="fincheck_test_")
os.close(_DB_FD)
os.environ["FINCHECK_DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
# Keep node2vec cheap in tests.
os.environ.setdefault("FINCHECK_EMBEDDING_DIM", "16")
os.environ.setdefault("FINCHECK_NUM_WALKS", "4")
os.environ.setdefault("FINCHECK_WALK_LENGTH", "10")

from fincheck.data.schema import Transaction  # noqa: E402
from fincheck.data.synthetic import SyntheticConfig, generate_transactions  # noqa: E402


def tx(txid, src, dst, amount, minute, laundering=False):
    return Transaction(
        id=txid,
        src=src,
        dst=dst,
        amount=amount,
        timestamp=datetime(2024, 1, 1) + timedelta(minutes=minute),
        is_laundering=laundering,
    )


@pytest.fixture
def make_tx():
    """Return the transaction factory for building ad-hoc graphs in tests."""
    return tx


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    try:
        os.remove(_DB_PATH)
    except OSError:
        pass


@pytest.fixture
def fresh_db():
    """Recreate all tables for a clean per-test database."""
    from fincheck.db.models import Base
    from fincheck.db.session import engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def cycle_txns():
    """A clean 4-node cycle A->B->C->D->A with balanced amounts."""
    return [
        tx("c1", "A", "B", 10_000, 1, True),
        tx("c2", "B", "C", 9_800, 2, True),
        tx("c3", "C", "D", 9_900, 3, True),
        tx("c4", "D", "A", 9_850, 4, True),
    ]


@pytest.fixture
def fanout_txns():
    """One hub paying 10 distinct accounts near-uniformly in a short window."""
    out = []
    for i in range(10):
        out.append(tx(f"f{i}", "HUB", f"M{i}", 4_500 + (i % 3) * 50, i * 0.5, True))
    return out


@pytest.fixture
def peeling_txns():
    """A 6-hop chain peeling ~10% at each step."""
    out = []
    amt = 100_000.0
    prev = "P0"
    for i in range(1, 7):
        amt = round(amt * 0.9, 2)
        out.append(tx(f"p{i}", prev, f"P{i}", amt, i, True))
        prev = f"P{i}"
    return out


@pytest.fixture
def small_dataset():
    return generate_transactions(SyntheticConfig(n_background=400, seed=7))
