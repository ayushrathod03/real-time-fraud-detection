import os
import json
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
API_URL = os.getenv("API_URL", "http://fraud-api:8000/predict")
DB_DSN = os.getenv("DB_DSN", "dbname=fraud_db user=postgres password=postgres host=postgres port=5432")


def wait_for_dependencies():
    while True:
        try:
            r = requests.get(API_URL.replace("/predict", "/health"), timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print("[CONSUMER] Waiting for FastAPI...")
        time.sleep(2)

    while True:
        try:
            conn = psycopg2.connect(DB_DSN)
            conn.close()
            break
        except Exception:
            pass
        print("[CONSUMER] Waiting for PostgreSQL...")
        time.sleep(2)


def main():
    wait_for_dependencies()
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    session = requests.Session()
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="fraud-scoring-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print("[CONSUMER] Listening for transactions...")
    batch = []

    for msg in consumer:
        event = msg.value
        produced_at = event.pop("produced_at", time.time())

        resp = session.post(API_URL, json=event, timeout=5)
        result = resp.json()
        e2e_ms = (time.time() - produced_at) * 1000

        batch.append((
            result["transaction_id"],
            event["Amount"],
            result["fraud_probability"],
            result["is_fraud"],
            round(e2e_ms, 3),
            result["inference_latency_ms"],
        ))

        if len(batch) >= 50:
            execute_values(
                cur,
                """
                INSERT INTO fraud_predictions
                (transaction_id, amount, fraud_probability, is_fraud, e2e_latency_ms, api_latency_ms)
                VALUES %s ON CONFLICT (transaction_id) DO NOTHING
                """,
                batch,
            )
            batch.clear()


if __name__ == "__main__":
    main()