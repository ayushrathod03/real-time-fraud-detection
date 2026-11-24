# Real-Time Fraud Detection Service (FastAPI + SQLite + XGBoost)

A low-latency fraud detection pipeline built with **FastAPI**, **SQLite**, and **XGBoost**. The active pipeline scores concurrent transaction streams via a multi-worker FastAPI REST service and persists predictions and latency telemetry to SQLite.

> **Note on Scope & Data**
> - **Synthetic Dataset:** Models are trained and evaluated on a synthetic imbalanced dataset generated with `sklearn.datasets.make_classification` (not real-world financial records).
> - **Active Stack vs. Planned Upgrade:** The implemented and benchmarked pipeline runs on **FastAPI + SQLite + XGBoost** (`scripts/local_benchmark.py`). Scaffolding for **Apache Kafka**, **PostgreSQL**, and **Docker Compose** (`streaming/`, `sql/`, `docker-compose.yml`) is included in the repository as a drafted future upgrade path, not as part of the benchmarked runtime.

## System Architecture

```text
[Concurrent CSV Stream (ThreadPoolExecutor)] ──► HTTP POST ──► [FastAPI /predict (4 Uvicorn Workers, XGBoost)]
                                                       │                        │
                                                       │◄────── JSON Score ─────┘
                                                       ▼
                                          [SQLite (fraud_predictions.db)]
```

## Dataset

- **Source:** Synthetic imbalanced dataset generated via `sklearn.datasets.make_classification` matching the Kaggle Credit Card Fraud schema (`Time`, `V1`–`V28`, `Amount`, `Class`), with **284,807 total rows** and **0.172% positive class prevalence**.
- **Split:** Stratified 80/20 train-test split (**227,845** train / **56,962** test rows), with `StandardScaler` fit strictly on the training set.

## Model Metrics (Holdout Test Set, Threshold = 0.50)

| Model | Precision | Recall | F1-Score | PR-AUC | 1-Row Inference Latency |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression (Baseline, `class_weight="balanced"`) | 0.0059 | 0.7719 | 0.0117 | 0.0154 | 0.133 ms |
| XGBoost (Final, `scale_pos_weight` tuned) | 0.3780 | 0.4211 | 0.3983 | 0.3673 | 0.982 ms |

With `class_weight="balanced"` (~580x minority penalty), the linear baseline over-predicted fraud (77.2% recall at 0.59% precision). XGBoost cut false positives significantly, improving PR-AUC from `0.0154` to `0.3673` and F1 from `0.0117` to `0.3983`.

## System Throughput & Latency

Measured with a **4-worker Uvicorn FastAPI + Local SQLite Stream** (`scripts/local_benchmark.py`) across **2,000 concurrent test transactions** (8 worker threads, 4 flagged fraud):

| Metric | Value |
| --- | --- |
| Sustained Throughput | 325.3 transactions/sec (TPS) |
| Standalone API p50 Latency | 4.90 ms |
| End-to-End Pipeline Latency (p50 / p95) | 19.22 ms / 36.89 ms |

## Hardware & Environment

- **CPU:** Intel Core i5-10300H @ 2.50GHz (4 physical cores / 8 logical processors)
- **RAM:** 8 GB (7.8 GB usable)
- **OS & Runtime:** Windows (x64), Python 3.13, `xgboost 2.1.1`, `scikit-learn 1.5.2`, `fastapi 0.115.0`

## Project Structure

```text
.
├── app/
│   ├── main.py                # FastAPI service (/health, /predict) loading XGBoost artifacts
│   └── schemas.py             # Pydantic request/response schemas
├── data/
│   └── test_transactions.csv  # Held-out 20% test split for replay & benchmarking
├── models/
│   ├── xgb_fraud.joblib       # Trained XGBoost classifier
│   ├── scaler.joblib          # Fitted StandardScaler
│   └── features.joblib        # Ordered feature list
├── scripts/
│   ├── train.py               # Trains Logistic Regression baseline vs. XGBoost and exports artifacts
│   ├── local_benchmark.py     # Concurrent FastAPI + SQLite benchmark
│   └── benchmark.py           # Drafted standalone API & PostgreSQL metrics script
├── sql/
│   └── init.sql               # Drafted PostgreSQL schema for future upgrade
├── streaming/
│   ├── producer.py            # Drafted Kafka producer for future upgrade
│   └── consumer.py            # Drafted Kafka consumer for future upgrade
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting Started (FastAPI + SQLite + XGBoost)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

2. *(Optional)* Retrain the models and regenerate `data/creditcard.csv` + `data/test_transactions.csv` (`data/creditcard.csv` is excluded from Git due to GitHub's 100 MB file limit):

```bash
python scripts/train.py
```

3. Start the 4-worker FastAPI server and run the local SQLite stream benchmark:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
python scripts/local_benchmark.py
```

## Future Upgrade: Distributed Streaming (Kafka + PostgreSQL)

The repository includes drafted scaffolding (`docker-compose.yml`, `streaming/producer.py`, `streaming/consumer.py`, `sql/init.sql`, and `scripts/benchmark.py`) to swap the local concurrent stream + SQLite sink for Apache Kafka and PostgreSQL in a containerized setup.
