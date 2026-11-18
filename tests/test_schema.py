"""Tests for the domain schema and validation."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from fincheck.data.schema import Account, Transaction, TransactionBatch


def test_transaction_valid():
    t = Transaction(id="t1", src="A", dst="B", amount=10.0, timestamp=datetime(2024, 1, 1))
    assert t.currency == "USD"
    assert t.is_laundering is False


def test_transaction_rejects_self_loop():
    with pytest.raises(ValidationError):
        Transaction(id="t1", src="A", dst="A", amount=10.0, timestamp=datetime(2024, 1, 1))


def test_transaction_rejects_non_positive_amount():
    with pytest.raises(ValidationError):
        Transaction(id="t1", src="A", dst="B", amount=0.0, timestamp=datetime(2024, 1, 1))


def test_transaction_rejects_empty_account():
    with pytest.raises(ValidationError):
        Transaction(id="t1", src="", dst="B", amount=5.0, timestamp=datetime(2024, 1, 1))


def test_account_defaults():
    a = Account(id="A")
    assert a.kind == "normal"
    assert a.name is None


def test_batch_len():
    batch = TransactionBatch(
        transactions=[
            Transaction(id="t1", src="A", dst="B", amount=1.0, timestamp=datetime(2024, 1, 1))
        ]
    )
    assert len(batch) == 1
