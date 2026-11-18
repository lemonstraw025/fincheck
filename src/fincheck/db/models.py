"""SQLAlchemy ORM models.

The schema is intentionally simple and portable across SQLite (local/dev) and
PostgreSQL (production). JSON columns use SQLAlchemy's generic ``JSON`` type,
which maps to ``JSONB``-compatible storage on PostgreSQL and TEXT on SQLite.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AccountRow(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="normal")


class TransactionRow(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    src: Mapped[str] = mapped_column(String(64), index=True)
    dst: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    is_laundering: Mapped[bool] = mapped_column(Boolean, default=False)


class AlertRow(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.id"), index=True, nullable=True
    )
    txid: Mapped[str] = mapped_column(String(64), index=True)
    src: Mapped[str] = mapped_column(String(64))
    dst: Mapped[str] = mapped_column(String(64))
    amount: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float, index=True)
    signals: Mapped[dict] = mapped_column(JSON, default=dict)
    patterns: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped["JobRow"] = relationship(back_populates="alerts")


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    n_transactions: Mapped[int] = mapped_column(Integer, default=0)
    n_alerts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    alerts: Mapped[List["AlertRow"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
