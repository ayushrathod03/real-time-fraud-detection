import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

DATA_PATH = "data/creditcard.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


def load_or_create_data():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}...")
        return pd.read_csv(DATA_PATH)

    print("[INFO] data/creditcard.csv not found. Generating realistic 284,807-row dataset (0.172% fraud)...")
    X, y = make_classification(
        n_samples=284807,
        n_features=29,
        n_informative=18,
        n_redundant=6,
        n_clusters_per_class=2,
        weights=[0.99828, 0.00172],
        flip_y=0.0005,
        random_state=42,
    )
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = np.abs(df["Amount"] * 50 + 88).round(2)
    df["Time"] = np.sort(np.random.uniform(0, 172792, size=len(df))).round(1)
    df["Class"] = y
    df = df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]]
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Saved dataset to {DATA_PATH}")
    return df


def evaluate_model(name, model, X_test, y_test):
    t0 = time.perf_counter()
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    total_ms = (time.perf_counter() - t0) * 1000

    # Measure single-row latency over 500 samples
    single_times = []
    for i in range(500):
        row = X_test[i : i + 1]
        ts = time.perf_counter()
        _ = model.predict_proba(row)
        single_times.append((time.perf_counter() - ts) * 1000)

    pr_auc = average_precision_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    return {
        "Model": name,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "PR-AUC": round(pr_auc, 4),
        "1-Row Latency (ms)": round(float(np.mean(single_times)), 3),
    }


def main():
    df = load_or_create_data()
    fraud_rate = df["Class"].mean() * 100
    print(f"\nDataset Shape: {df.shape} | Fraud Rate: {fraud_rate:.3f}% ({df['Class'].sum()} frauds)")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save test set for Kafka replay
    test_df = X_test.copy()
    test_df["Class"] = y_test
    test_df.to_csv("data/test_transactions.csv", index=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline: Logistic Regression
    print("[INFO] Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate_model("Logistic Regression", lr, X_test_scaled, y_test)

    # 2. XGBoost
    print("[INFO] Training XGBoost classifier...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos * 0.25,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_metrics = evaluate_model("XGBoost", xgb, X_test_scaled, y_test)

    results_df = pd.DataFrame([lr_metrics, xgb_metrics])
    print("\n================ MODEL BENCHMARK RESULTS ================")
    print(results_df.to_string(index=False))

    lift = ((xgb_metrics["PR-AUC"] - lr_metrics["PR-AUC"]) / lr_metrics["PR-AUC"]) * 100
    print(f"\nPR-AUC Relative Lift: +{lift:.1f}% ({lr_metrics['PR-AUC']} -> {xgb_metrics['PR-AUC']})")
    print(f"Recall Improvement:   {lr_metrics['Recall']*100:.1f}% -> {xgb_metrics['Recall']*100:.1f}%")
    print("=========================================================\n")

    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_fraud.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "features.joblib"))
    print("[INFO] Saved model artifacts to models/")


if __name__ == "__main__":
    main()