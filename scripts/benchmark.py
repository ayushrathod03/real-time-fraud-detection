import time
import requests
import psycopg2
import numpy as np
import pandas as pd

API_URL = "http://localhost:8000/predict"
DB_DSN = "dbname=fraud_db user=postgres password=postgres host=localhost port=5432"


def benchmark_standalone_api(n=500):
    df = pd.read_csv("data/test_transactions.csv").drop(columns=["Class"]).head(n)
    latencies = []
    session = requests.Session()

    for i, row in enumerate(df.to_dict(orient="records")):
        row["transaction_id"] = f"bench_{i}"
        t0 = time.perf_counter()
        r = session.post(API_URL, json=row)
        r.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000)

    print("\n================ STANDALONE FASTAPI LATENCY ================")
    print(f"Requests Sent: {n}")
    print(f"p50 Latency:   {np.percentile(latencies, 50):.2f} ms")
    print(f"p95 Latency:   {np.percentile(latencies, 95):.2f} ms")
    print(f"Mean Latency:  {np.mean(latencies):.2f} ms")


def benchmark_postgres_stream():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            COUNT(*) AS total_processed,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS flagged_frauds,
            ROUND(
                COUNT(*) / NULLIF(EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))), 0)::numeric, 
                1
            ) AS throughput_tps,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY e2e_latency_ms)::numeric, 2) AS p50_e2e_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e2e_latency_ms)::numeric, 2) AS p95_e2e_ms,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY api_latency_ms)::numeric, 2) AS p50_api_ms
        FROM fraud_predictions;
        """
    )
    row = cur.fetchone()
    conn.close()

    print("\n================ END-TO-END STREAMING METRICS ================")
    print(f"Total Scored & Stored: {row[0]} transactions ({row[1]} flagged fraud)")
    print(f"Sustained Throughput:  {row[2]} transactions/sec (TPS)")
    print(f"End-to-End p50:        {row[3]} ms")
    print(f"End-to-End p95:        {row[4]} ms")
    print(f"In-Pipeline API p50:   {row[5]} ms")
    print("==============================================================\n")


if __name__ == "__main__":
    benchmark_standalone_api(500)
    benchmark_postgres_stream()