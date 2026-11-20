
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Loading validation dataset...")

df = pd.read_csv("data/processed/validation_features.csv")

X_val = df.drop("class", axis=1)
y_val = df["class"]

print("Loading retrained Version 3 model...")

model = joblib.load("models/dns_detector_v3.pkl")

print("Generating predictions...")

probabilities = model.predict_proba(X_val)

# Probability of DNS tunneling (class 2)
tunneling_probabilities = probabilities[:, 2]

actual = (y_val == 2).astype(int)

thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

print("\n--- VALIDATION THRESHOLD RESULTS ---")

for threshold in thresholds:

    predictions = (
        tunneling_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        actual, predictions, zero_division=0
    )

    recall = recall_score(
        actual, predictions, zero_division=0
    )

    f1 = f1_score(
        actual, predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )
