"""Core domain schema shared by the graph, ML, scoring and API layers."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Account(BaseModel):
    """A party in the transaction network (a graph node)."""

    id: str
    name: Optional[str] = None
    # Free-form label used only by the synthetic generator / evaluation.
    kind: str = "normal"


class Transaction(BaseModel):
    """A single directed money movement (a graph edge)."""

    id: str
    src: str = Field(..., description="Source account id (payer)")
    dst: str = Field(..., description="Destination account id (payee)")
    amount: float = Field(..., gt=0)
    timestamp: datetime
    currency: str = "USD"
    # Ground-truth flag, present only in synthetic / benchmark data.
    is_laundering: bool = False

    @field_validator("src", "dst")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("account id must be non-empty")
        return v

    @field_validator("dst")
    @classmethod
    def _no_self_loop(cls, v: str, info) -> str:
        src = info.data.get("src")
        if src is not None and v == src:
            raise ValueError("self-transfers (src == dst) are not allowed")
        return v


class TransactionBatch(BaseModel):
    """A batch of transactions submitted together for analysis."""

    transactions: List[Transaction]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.transactions)
