"""Tests for background analysis jobs."""

from fincheck.api import jobs as jobs_mod
from fincheck.db.models import AlertRow, JobRow
from fincheck.db.session import session_scope


def test_create_job(fresh_db):
    job_id = jobs_mod.create_job(10)
    with session_scope() as s:
        job = s.get(JobRow, job_id)
        assert job.status == "pending"
        assert job.n_transactions == 10


def test_run_analysis_job_persists_alerts(fresh_db, cycle_txns):
    job_id = jobs_mod.create_job(len(cycle_txns))
    jobs_mod.run_analysis_job(job_id, transactions=cycle_txns, threshold=0.3)
    with session_scope() as s:
        job = s.get(JobRow, job_id)
        assert job.status == "completed"
        assert job.n_alerts >= 1
        assert s.query(AlertRow).filter(AlertRow.job_id == job_id).count() == job.n_alerts


def test_run_analysis_job_idempotent(fresh_db, cycle_txns):
    job_id = jobs_mod.create_job(len(cycle_txns))
    jobs_mod.run_analysis_job(job_id, transactions=cycle_txns, threshold=0.3)
    first = _count_alerts(job_id)
    jobs_mod.run_analysis_job(job_id, transactions=cycle_txns, threshold=0.3)
    assert _count_alerts(job_id) == first


def test_run_analysis_job_missing_job_is_noop(fresh_db, cycle_txns):
    # Should not raise even though the job row does not exist.
    jobs_mod.run_analysis_job("nonexistent", transactions=cycle_txns)


def _count_alerts(job_id):
    with session_scope() as s:
        return s.query(AlertRow).filter(AlertRow.job_id == job_id).count()
