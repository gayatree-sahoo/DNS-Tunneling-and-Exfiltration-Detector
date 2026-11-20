
import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

print("Loading test dataset...")

df = pd.read_csv("data/processed/test_features.csv")

X_test = df.drop("class", axis=1)
y_test = df["class"]

print("Loading Version 3 model...")

model = joblib.load("models/dns_detector_v3.pkl")

print("Calculating prediction probabilities...")

probabilities = model.predict_proba(X_test)

# Class 2 represents DNS tunneling
tunneling_probabilities = probabilities[:, 2]

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

print("\n--- DNS TUNNELING THRESHOLD EXPERIMENT ---")

for threshold in thresholds:

    predictions = (
        tunneling_probabilities >= threshold
    ).astype(int)

    actual = (y_test == 2).astype(int)

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
