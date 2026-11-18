"""Runtime configuration, read from environment with sensible defaults.

The application defaults to a local SQLite database so it runs out of the box,
but any SQLAlchemy URL works. Point ``FINCHECK_DATABASE_URL`` at PostgreSQL in
production, e.g. ``postgresql+psycopg2://user:pass@localhost:5432/fincheck``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot."""

    database_url: str = os.getenv(
        "FINCHECK_DATABASE_URL", "sqlite:///fincheck.sqlite3"
    )
    # Deterministic seed shared across synthetic data + ML components.
    seed: int = _getenv_int("FINCHECK_SEED", 7)

    # node2vec hyper-parameters.
    embedding_dim: int = _getenv_int("FINCHECK_EMBEDDING_DIM", 32)
    walk_length: int = _getenv_int("FINCHECK_WALK_LENGTH", 20)
    num_walks: int = _getenv_int("FINCHECK_NUM_WALKS", 10)
    p_return: float = _getenv_float("FINCHECK_P_RETURN", 1.0)
    q_inout: float = _getenv_float("FINCHECK_Q_INOUT", 0.5)
    window: int = _getenv_int("FINCHECK_WINDOW", 5)

    # IsolationForest.
    contamination: float = _getenv_float("FINCHECK_CONTAMINATION", 0.08)

    # Alert fusion threshold (a fused probability at/above this raises an alert).
    alert_threshold: float = _getenv_float("FINCHECK_ALERT_THRESHOLD", 0.5)


def get_settings() -> Settings:
    """Return a fresh settings snapshot (re-reads the environment)."""
    return Settings()


settings = get_settings()
