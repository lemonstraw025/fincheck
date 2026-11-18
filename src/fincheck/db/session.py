"""Engine and session helpers.

Uses a single module-level engine configured from :data:`settings.database_url`.
SQLite gets ``check_same_thread=False`` so FastAPI background tasks can share
the connection pool.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from fincheck.config import settings
from fincheck.db.models import Base

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    future=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Return a new session (caller is responsible for closing)."""
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
