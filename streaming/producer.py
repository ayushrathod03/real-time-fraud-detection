import os
import json
import time
import uuid
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
NUM_EVENTS = int(os.getenv("NUM_EVENTS", "5000"))


def main():
    df = pd.read_csv("data/test_transactions.csv").drop(columns=["Class"])
    sample_df = df.sample(n=min(NUM_EVENTS, len(df)), random_state=42)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
        batch_size=32768,
    )

    print(f"[PRODUCER] Sending {len(sample_df)} transactions to '{TOPIC}'...")
    t0 = time.perf_counter()

    for idx, row in enumerate(sample_df.to_dict(orient="records")):
        row["transaction_id"] = f"tx_{uuid.uuid4().hex[:12]}"
        row["produced_at"] = time.time()
        producer.send(TOPIC, value=row)

    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"[PRODUCER] Published {len(sample_df)} events in {elapsed:.2f}s ({len(sample_df)/elapsed:.1f} msg/s)")


if __name__ == "__main__":
    main()