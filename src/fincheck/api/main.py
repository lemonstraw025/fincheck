"""FastAPI application exposing the fraud-detection pipeline.

Endpoints (10):

* ``GET  /health``                     — liveness probe.
* ``GET  /stats``                      — dataset + alert summary counts.
* ``POST /transactions``               — ingest a batch of transactions.
* ``GET  /transactions``               — list stored transactions (paginated).
* ``POST /generate``                   — generate + store synthetic data.
* ``POST /analyze``                    — synchronous analysis, returns top alerts.
* ``POST /jobs``                       — start a background analysis job.
* ``GET  /jobs``                       — list jobs.
* ``GET  /jobs/{job_id}``              — poll one job's status.
* ``GET  /alerts``                     — list ranked alerts (filterable).
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from fincheck.data.schema import Transaction
from fincheck.data.synthetic import SyntheticConfig, generate_transactions
from fincheck.db.models import AlertRow, JobRow, TransactionRow
from fincheck.db.session import init_db, session_scope
from fincheck.api import jobs as jobs_mod
from fincheck.scoring.fusion import AlertEngine

app = FastAPI(
    title="fincheck",
    version="0.1.0",
    description="Graph-based money-laundering detection with explainable alerts.",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


# --------------------------------------------------------------------------- #
# Request / response models                                                    #
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    transactions: List[Transaction]


class GenerateRequest(BaseModel):
    n_accounts: int = 600
    n_background: int = 4300
    n_cycles: int = 18
    n_fan_outs: int = 14
    n_peeling_chains: int = 12
    seed: int = 7
    replace: bool = Field(True, description="Clear existing transactions first.")


class AnalyzeRequest(BaseModel):
    threshold: Optional[float] = None
    top_k: int = 50
    transactions: Optional[List[Transaction]] = None


class JobRequest(BaseModel):
    threshold: Optional[float] = None


def _persist_transactions(txns: List[Transaction], replace: bool) -> int:
    with session_scope() as s:
        if replace:
            s.query(TransactionRow).delete()
        for t in txns:
            s.merge(
                TransactionRow(
                    id=t.id,
                    src=t.src,
                    dst=t.dst,
                    amount=t.amount,
                    currency=t.currency,
                    timestamp=t.timestamp,
                    is_laundering=t.is_laundering,
                )
            )
    return len(txns)


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "fincheck"}


@app.get("/stats")
def stats() -> dict:
    with session_scope() as s:
        n_tx = s.query(TransactionRow).count()
        n_laundering = (
            s.query(TransactionRow).filter(TransactionRow.is_laundering).count()
        )
        n_jobs = s.query(JobRow).count()
        n_alerts = s.query(AlertRow).count()
    return {
        "transactions": n_tx,
        "labelled_laundering": n_laundering,
        "jobs": n_jobs,
        "alerts": n_alerts,
    }


@app.post("/transactions")
def ingest(req: IngestRequest) -> dict:
    n = _persist_transactions(req.transactions, replace=False)
    return {"ingested": n}


@app.get("/transactions")
def list_transactions(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    with session_scope() as s:
        q = s.query(TransactionRow).order_by(TransactionRow.timestamp)
        total = q.count()
        rows = q.offset(offset).limit(limit).all()
        items = [
            {
                "id": r.id,
                "src": r.src,
                "dst": r.dst,
                "amount": r.amount,
                "timestamp": r.timestamp.isoformat(),
                "is_laundering": r.is_laundering,
            }
            for r in rows
        ]
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    cfg = SyntheticConfig(
        n_accounts=req.n_accounts,
        n_background=req.n_background,
        n_cycles=req.n_cycles,
        n_fan_outs=req.n_fan_outs,
        n_peeling_chains=req.n_peeling_chains,
        seed=req.seed,
    )
    txns = generate_transactions(cfg)
    n = _persist_transactions(txns, replace=req.replace)
    laundering = sum(1 for t in txns if t.is_laundering)
    return {"generated": n, "labelled_laundering": laundering}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if req.transactions is not None:
        txns = req.transactions
    else:
        txns = jobs_mod._load_transactions()
    if not txns:
        raise HTTPException(status_code=400, detail="no transactions to analyze")
    engine = AlertEngine(threshold=req.threshold) if req.threshold is not None else AlertEngine()
    alerts = engine.run(txns)
    top = AlertEngine.rank(alerts, top_k=req.top_k)
    return {
        "analyzed": len(txns),
        "alerts": len(alerts),
        "pattern_hits": len(engine.pattern_hits),
        "top": [a.to_dict() for a in top],
    }


@app.post("/jobs")
def start_job(req: JobRequest, background: BackgroundTasks) -> dict:
    with session_scope() as s:
        n_tx = s.query(TransactionRow).count()
    if n_tx == 0:
        raise HTTPException(status_code=400, detail="no stored transactions")
    job_id = jobs_mod.create_job(n_tx)
    background.add_task(jobs_mod.run_analysis_job, job_id, None, req.threshold)
    return {"job_id": job_id, "status": "pending", "n_transactions": n_tx}


@app.get("/jobs")
def list_jobs(limit: int = Query(50, ge=1, le=500)) -> dict:
    with session_scope() as s:
        rows = (
            s.query(JobRow).order_by(JobRow.created_at.desc()).limit(limit).all()
        )
        items = [
            {
                "id": j.id,
                "status": j.status,
                "n_transactions": j.n_transactions,
                "n_alerts": j.n_alerts,
                "error": j.error,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            }
            for j in rows
        ]
    return {"items": items}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with session_scope() as s:
        j = s.get(JobRow, job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": j.id,
            "status": j.status,
            "n_transactions": j.n_transactions,
            "n_alerts": j.n_alerts,
            "error": j.error,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        }


@app.get("/alerts")
def list_alerts(
    job_id: Optional[str] = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=1000),
) -> dict:
    with session_scope() as s:
        q = s.query(AlertRow).filter(AlertRow.score >= min_score)
        if job_id:
            q = q.filter(AlertRow.job_id == job_id)
        rows = q.order_by(AlertRow.score.desc()).limit(limit).all()
        items = [
            {
                "txid": r.txid,
                "src": r.src,
                "dst": r.dst,
                "amount": r.amount,
                "score": round(r.score, 4),
                "signals": r.signals,
                "patterns": r.patterns,
                "explanation": r.explanation,
            }
            for r in rows
        ]
    return {"items": items}
