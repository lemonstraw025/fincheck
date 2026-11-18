"""Background analysis jobs.

An analysis job loads transactions, runs the :class:`AlertEngine`, and persists
the resulting alerts. Jobs are tracked in the ``jobs`` table so clients can poll
status. Execution is triggered via FastAPI ``BackgroundTasks`` but the function
is a plain callable and is unit-tested directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fincheck.data.schema import Transaction
from fincheck.db.models import AlertRow, JobRow, TransactionRow
from fincheck.db.session import session_scope
from fincheck.scoring.fusion import AlertEngine


def new_job_id() -> str:
    return "J" + uuid.uuid4().hex[:12]


def create_job(n_transactions: int) -> str:
    """Insert a pending job row and return its id."""
    job_id = new_job_id()
    with session_scope() as s:
        s.add(JobRow(id=job_id, status="pending", n_transactions=n_transactions))
    return job_id


def _load_transactions(job_filter: Optional[List[str]] = None) -> List[Transaction]:
    with session_scope() as s:
        rows = s.query(TransactionRow).all()
        return [
            Transaction(
                id=r.id,
                src=r.src,
                dst=r.dst,
                amount=r.amount,
                timestamp=r.timestamp,
                currency=r.currency,
                is_laundering=r.is_laundering,
            )
            for r in rows
        ]


def run_analysis_job(
    job_id: str,
    transactions: Optional[List[Transaction]] = None,
    threshold: Optional[float] = None,
) -> None:
    """Execute an analysis job and persist its alerts.

    If ``transactions`` is ``None`` the job analyses all stored transactions.
    Any exception is recorded on the job row rather than raised, so a failed
    background task leaves a diagnosable record.
    """
    with session_scope() as s:
        job = s.get(JobRow, job_id)
        if job is None:
            return
        job.status = "running"

    try:
        txns = transactions if transactions is not None else _load_transactions()
        engine = AlertEngine(threshold=threshold) if threshold is not None else AlertEngine()
        alerts = engine.run(txns)

        with session_scope() as s:
            # Clear prior alerts for this job (idempotent re-runs).
            s.query(AlertRow).filter(AlertRow.job_id == job_id).delete()
            for a in alerts:
                s.add(
                    AlertRow(
                        job_id=job_id,
                        txid=a.txid,
                        src=a.src,
                        dst=a.dst,
                        amount=a.amount,
                        score=a.score,
                        signals=a.signals,
                        patterns=a.patterns,
                        explanation=a.explanation,
                    )
                )
            job = s.get(JobRow, job_id)
            job.status = "completed"
            job.n_transactions = len(txns)
            job.n_alerts = len(alerts)
            job.finished_at = datetime.utcnow()
    except Exception as exc:  # noqa: BLE001 - record and surface via API
        with session_scope() as s:
            job = s.get(JobRow, job_id)
            if job is not None:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = datetime.utcnow()
