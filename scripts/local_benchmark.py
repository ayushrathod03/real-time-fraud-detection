import time
import sqlite3
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://127.0.0.1:8000/predict"


def send_tx(args):
    idx, row = args
    row["transaction_id"] = f"tx_{idx}"
    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=row, timeout=5)
    resp.raise_for_status()
    e2e_ms = (time.perf_counter() - t0) * 1000
    data = resp.json()
    return (
        data["transaction_id"],
        row["Amount"],
        data["fraud_probability"],
        int(data["is_fraud"]),
        round(e2e_ms, 3),
        data["inference_latency_ms"],
    )


def main():
    df = pd.read_csv("data/test_transactions.csv").drop(columns=["Class"]).head(2000)
    records = list(enumerate(df.to_dict(orient="records")))

    conn = sqlite3.connect("fraud_predictions.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_predictions (
            transaction_id TEXT PRIMARY KEY,
            amount REAL,
            fraud_probability REAL,
            is_fraud INTEGER,
            e2e_latency_ms REAL,
            api_latency_ms REAL
        )
        """
    )

    print(f"[INFO] Streaming {len(records)} transactions through FastAPI...")
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send_tx, records))

    cur.executemany(
        "INSERT OR REPLACE INTO fraud_predictions VALUES (?, ?, ?, ?, ?, ?)",
        results,
    )
    conn.commit()
    total_sec = time.perf_counter() - t_start

    e2e_latencies = [r[4] for r in results]
    api_latencies = [r[5] for r in results]
    frauds = sum(r[3] for r in results)

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Total Processed:      {len(results)} transactions ({frauds} flagged fraud)")
    print(f"Sustained Throughput: {len(results) / total_sec:.1f} transactions/sec (TPS)")
    print(f"End-to-End p50:       {np.percentile(e2e_latencies, 50):.2f} ms")
    print(f"End-to-End p95:       {np.percentile(e2e_latencies, 95):.2f} ms")
    print(f"Standalone API p50:   {np.percentile(api_latencies, 50):.2f} ms")
    print("===================================================\n")
    conn.close()


if __name__ == "__main__":
    main()