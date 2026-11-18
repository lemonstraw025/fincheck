"""Endpoint tests using FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from fincheck.api.main import app


@pytest.fixture
def client(fresh_db):
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_and_stats(client):
    r = client.post("/generate", json={"n_background": 300, "seed": 7, "replace": True})
    assert r.status_code == 200
    assert r.json()["generated"] > 0

    stats = client.get("/stats").json()
    assert stats["transactions"] > 0
    assert stats["labelled_laundering"] > 0


def test_ingest_and_list(client):
    payload = {
        "transactions": [
            {"id": "t1", "src": "A", "dst": "B", "amount": 10.0, "timestamp": "2024-01-01T00:00:00"},
            {"id": "t2", "src": "B", "dst": "C", "amount": 20.0, "timestamp": "2024-01-01T00:01:00"},
        ]
    }
    assert client.post("/transactions", json=payload).json()["ingested"] == 2
    listed = client.get("/transactions", params={"limit": 10}).json()
    assert listed["total"] == 2
    assert len(listed["items"]) == 2


def test_analyze_endpoint(client):
    client.post("/generate", json={"n_background": 300, "seed": 7, "replace": True})
    r = client.post("/analyze", json={"threshold": 0.4, "top_k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["analyzed"] > 0
    assert "top" in body
    assert len(body["top"]) <= 10


def test_analyze_empty_is_400(client):
    r = client.post("/analyze", json={"top_k": 5})
    assert r.status_code == 400


def test_job_lifecycle(client):
    client.post("/generate", json={"n_background": 300, "seed": 7, "replace": True})
    start = client.post("/jobs", json={"threshold": 0.4}).json()
    job_id = start["job_id"]
    # Background task runs synchronously within TestClient request handling.
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] in {"completed", "running", "pending"}

    jobs = client.get("/jobs").json()
    assert any(j["id"] == job_id for j in jobs["items"])


def test_jobs_requires_transactions(client):
    assert client.post("/jobs", json={}).status_code == 400


def test_alerts_listing(client):
    client.post("/generate", json={"n_background": 300, "seed": 7, "replace": True})
    job_id = client.post("/jobs", json={"threshold": 0.4}).json()["job_id"]
    alerts = client.get("/alerts", params={"job_id": job_id, "limit": 5}).json()
    assert "items" in alerts
    assert len(alerts["items"]) <= 5


def test_get_missing_job_404(client):
    assert client.get("/jobs/nope").status_code == 404
