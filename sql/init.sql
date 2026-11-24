CREATE TABLE IF NOT EXISTS fraud_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) UNIQUE NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    fraud_probability FLOAT NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    e2e_latency_ms FLOAT NOT NULL,
    api_latency_ms FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_is_fraud ON fraud_predictions(is_fraud);