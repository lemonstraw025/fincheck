# fincheck — graph-based money-laundering detection

Graph + ML pipeline that models financial transactions as a network, detects
classic laundering motifs, and fuses the structural signals with an
unsupervised anomaly model into **ranked, plain-English alerts**.

Built with **Python, FastAPI, NetworkX, scikit-learn, SQLAlchemy (PostgreSQL /
SQLite), and Streamlit**.

---

## What it does

1. **Model** transactions as a directed `MultiDiGraph` (accounts = nodes,
   transfers = edges).
2. **Detect** three laundering motifs with graph traversals:
   - **cycles** — bounded-depth **DFS** finds rings where money returns to its
     origin;
   - **fan-outs** — one-hop **BFS** from high out-degree hubs catches
     structuring/smurfing bursts;
   - **peeling chains** — **DFS** over time-ordered, decreasing-amount paths.
3. **Score** every account with an **IsolationForest** over **node2vec**
   embeddings + structural features.
4. **Fuse** the structural and ML signals with a **noisy-OR**, rank alerts with
   a **heap**, and attach a template-driven **plain-English explanation** to
   each flagged transaction.
5. **Serve** everything through a **FastAPI** backend (10 endpoints + background
   analysis jobs) and a **Streamlit** dashboard.

## Project layout

```
src/fincheck/
  config.py            # env-driven settings
  data/                # pydantic schema + synthetic generator
  graph/               # NetworkX builder + DFS/BFS pattern detectors
  ml/                  # node2vec (numpy) + IsolationForest anomaly
  scoring/             # noisy-OR fusion, heap ranking, explanations
  db/                  # SQLAlchemy models + session (SQLite / PostgreSQL)
  api/                 # FastAPI app + background jobs
  dashboard/           # Streamlit UI
scripts/benchmark.py   # precision/recall benchmark vs ground truth
tests/                 # pytest suite
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run the benchmark on synthetic ground truth:

```bash
python scripts/benchmark.py --target-precision 0.83
```

Start the API:

```bash
uvicorn fincheck.api.main:app --reload
```

Then, with the API running, launch the dashboard:

```bash
streamlit run src/fincheck/dashboard/app.py
```

## API

| Method | Path              | Purpose                                   |
|--------|-------------------|-------------------------------------------|
| GET    | `/health`         | liveness probe                            |
| GET    | `/stats`          | dataset + alert summary counts            |
| POST   | `/transactions`   | ingest a batch of transactions            |
| GET    | `/transactions`   | list stored transactions (paginated)      |
| POST   | `/generate`       | generate + store synthetic data           |
| POST   | `/analyze`        | synchronous analysis, returns top alerts  |
| POST   | `/jobs`           | start a background analysis job           |
| GET    | `/jobs`           | list jobs                                 |
| GET    | `/jobs/{job_id}`  | poll one job's status                     |
| GET    | `/alerts`         | list ranked alerts (filterable)           |

Example:

```bash
curl -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"n_background": 4300, "seed": 7, "replace": true}'

curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"threshold": 0.5, "top_k": 20}'
```

## Configuration

All settings are environment variables with sensible defaults (see
[`config.py`](src/fincheck/config.py)). The most important one:

```bash
# Defaults to local SQLite; point at PostgreSQL in production:
export FINCHECK_DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/fincheck"
```

Install the Postgres driver with the extra: `pip install -e ".[postgres]"`.

## Benchmark

`scripts/benchmark.py` builds a labelled synthetic dataset and compares two
operating regimes at a target precision:

- **anomaly-only** — IsolationForest over node2vec + structural features;
- **fused** — the full pipeline (anomaly + structural detectors via noisy-OR).

It sweeps the decision threshold, locks to the point nearest the target
precision, and reports recall for each regime plus the recall lift from adding
the graph detectors. Numbers depend on the seed and dataset configuration.

## Notes on node2vec

The embeddings are a compact, dependency-light node2vec implemented in NumPy
(biased second-order random walks + skip-gram with negative sampling), kept
deterministic by a seed. It is designed for transparency and reproducibility
rather than to replace a tuned gensim pipeline.

## License

MIT
