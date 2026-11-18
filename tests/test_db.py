"""Tests for the database models and session helpers."""

from datetime import datetime

from fincheck.db.models import AccountRow, AlertRow, JobRow, TransactionRow
from fincheck.db.session import session_scope


def test_init_creates_tables(fresh_db):
    from sqlalchemy import inspect

    tables = set(inspect(fresh_db).get_table_names())
    assert {"accounts", "transactions", "alerts", "jobs"} <= tables


def test_insert_and_query_transaction(fresh_db):
    with session_scope() as s:
        s.add(TransactionRow(id="t1", src="A", dst="B", amount=10.0, timestamp=datetime(2024, 1, 1)))
    with session_scope() as s:
        row = s.get(TransactionRow, "t1")
        assert row.src == "A"
        assert row.amount == 10.0


def test_job_alert_relationship(fresh_db):
    with session_scope() as s:
        s.add(JobRow(id="J1", status="completed", n_transactions=5))
        s.add(
            AlertRow(
                job_id="J1",
                txid="t1",
                src="A",
                dst="B",
                amount=100.0,
                score=0.9,
                signals={"anomaly": 0.9},
                patterns=[{"pattern": "cycle"}],
                explanation="because",
            )
        )
    with session_scope() as s:
        job = s.get(JobRow, "J1")
        assert len(job.alerts) == 1
        assert job.alerts[0].signals["anomaly"] == 0.9


def test_cascade_delete(fresh_db):
    with session_scope() as s:
        s.add(JobRow(id="J2", status="completed"))
        s.add(AlertRow(job_id="J2", txid="t", src="A", dst="B", amount=1.0, score=0.5))
    with session_scope() as s:
        s.delete(s.get(JobRow, "J2"))
    with session_scope() as s:
        assert s.query(AlertRow).filter(AlertRow.job_id == "J2").count() == 0


def test_account_row(fresh_db):
    with session_scope() as s:
        s.add(AccountRow(id="A", name="Alice"))
    with session_scope() as s:
        assert s.get(AccountRow, "A").name == "Alice"
