# Real-Time Fraud Detection Service (FastAPI + SQLite + XGBoost)

A low-latency fraud detection pipeline built with **FastAPI**, **SQLite**, and **XGBoost**. The active pipeline scores concurrent transaction streams via a multi-worker FastAPI REST service and persists predictions and latency telemetry to SQLite.

> **Note on Stack**
> The implemented and benchmarked pipeline runs on **FastAPI + SQLite + XGBoost** (`scripts/local_benchmark.py`). Scaffolding for **Apache Kafka**, **PostgreSQL**, and **Docker Compose** (`streaming/`, `sql/`, `docker-compose.yml`) is included in the repository as a drafted future upgrade path, not as part of the benchmarked runtime.

## System Architecture

```text
[Concurrent CSV Stream (ThreadPoolExecutor)] ──► HTTP POST ──► [FastAPI /predict (4 Uvicorn Workers, XGBoost)]
                                                       │                        │
                                                       │◄────── JSON Score ─────┘
                                                       ▼
                                          [SQLite (fraud_predictions.db)]
```

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection dataset (`mlg-ulb/creditcardfraud`)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing **284,807** real European cardholder transactions across **30 numerical features** (`Time`, `V1`–`V28` PCA components, `Amount`) with **492 fraud cases** (**0.172%** positive class prevalence).
- **Split:** Stratified 80/20 train-test split (**227,845** train / **56,962** test rows; **98** fraud cases in test), with `StandardScaler` fit strictly on the training set to prevent data leakage.

## Model Metrics (Holdout Test Set, Threshold = 0.50)

| Model | Precision | Recall | F1-Score | PR-AUC | 1-Row Inference Latency |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression (Baseline, `class_weight="balanced"`) | 0.0610 | 0.9184 | 0.1144 | 0.7190 | 0.154 ms |
| XGBoost (Final, `scale_pos_weight` tuned) | 0.8632 | 0.8367 | 0.8497 | 0.8809 | 0.753 ms |

While the cost-weighted linear baseline achieved **91.84%** recall, it suffered from a **93.9%** false-positive rate (`0.0610` precision). Tuned XGBoost increased precision by **14.1x** (`0.0610` → `0.8632`), lifted F1-score from `0.1144` to `0.8497`, and improved PR-AUC by **+22.5%** (`0.7190` → `0.8809`) at `0.753 ms` single-row inference latency.

## System Throughput & Latency

Measured with a **4-worker Uvicorn FastAPI + Local SQLite Stream** (`scripts/local_benchmark.py`) across **2,000 concurrent holdout test transactions** (8 worker threads):

| Metric | Value |
| --- | --- |
| Sustained Throughput | 348.8 transactions/sec (TPS) |
| Standalone API p50 Latency | 5.04 ms |
| End-to-End Pipeline Latency (p50 / p95) | 18.69 ms / 37.66 ms |

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

2. *(Optional)* To retrain the models on the real dataset, download `creditcard.csv` from [Kaggle (`mlg-ulb/creditcardfraud`)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/creditcard.csv` (`data/creditcard.csv` is excluded from Git due to GitHub's 100 MB file limit; if missing, `train.py` falls back to generating a synthetic dataset):

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
