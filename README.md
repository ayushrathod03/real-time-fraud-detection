# Real-Time Fraud Detection Service

An end-to-end, low-latency fraud detection pipeline built with **XGBoost**, **FastAPI**, **Apache Kafka**, **PostgreSQL**, and **Docker**. The system ingests streaming transaction events, scores them asynchronously via a REST inference service, and persists predictions with latency telemetry.

## System Architecture

```text
[CSV Replay Producer] ──► [Apache Kafka (Topic: transactions)]
                                      │
                                      ▼
                            [Python Consumer] ──► HTTP POST ──► [FastAPI /predict (XGBoost)]
                                      │                                  │
                                      │◄──────── JSON Score ─────────────┘
                                      ▼
                         [PostgreSQL (fraud_predictions)]
```

## Benchmark Results

Measured across 2,000 streamed transactions using `scripts/local_benchmark.py`:

| Metric | Value |
| --- | --- |
| Total Processed | 2,000 transactions (4 flagged fraud) |
| Sustained Throughput | 325.3 transactions/sec (TPS) |
| End-to-End p50 Latency | 19.22 ms |
| End-to-End p95 Latency | 36.89 ms |
| Standalone API p50 Latency | 4.90 ms |

## Project Structure

```text
.
├── app/
│   ├── main.py              # FastAPI service (/health, /predict) loading XGBoost artifacts
│   └── schemas.py           # Pydantic request/response schemas
├── data/
│   └── test_transactions.csv # Held-out 20% test split for replay & benchmarking
├── models/
│   ├── xgb_fraud.joblib     # Trained XGBoost classifier
│   ├── scaler.joblib        # Fitted StandardScaler
│   └── features.joblib      # Ordered feature list
├── scripts/
│   ├── train.py             # Trains Logistic Regression baseline vs. XGBoost and exports artifacts
│   ├── benchmark.py         # Standalone API & PostgreSQL streaming metrics
│   └── local_benchmark.py   # Local concurrent FastAPI + SQLite benchmark
├── sql/
│   └── init.sql             # PostgreSQL schema for fraud_predictions
├── streaming/
│   ├── producer.py          # Kafka transaction replay producer
│   └── consumer.py          # Kafka consumer calling FastAPI and batch-inserting to PostgreSQL
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting Started

### 1. Run Full Pipeline with Docker Compose

Start Zookeeper, Kafka, PostgreSQL, the FastAPI inference service (`4` Uvicorn workers), and the stream consumer:

```bash
docker compose up --build -d
```

Replay transactions into Kafka (`transactions` topic):

```bash
python streaming/producer.py
```

Collect standalone API and end-to-end PostgreSQL streaming metrics:

```bash
python scripts/benchmark.py
```

### 2. Run Locally (Without Kafka/Docker)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

*(Optional)* Retrain the model and regenerate `data/creditcard.csv` + `data/test_transactions.csv` (the full `data/creditcard.csv` dataset is excluded from Git due to GitHub's 100 MB file limit):

```bash
python scripts/train.py
```

Start the FastAPI server and run the local benchmark:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
python scripts/local_benchmark.py
```
