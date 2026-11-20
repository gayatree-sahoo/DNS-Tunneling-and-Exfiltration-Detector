
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

print("Loading validation dataset...")

data = pd.read_csv("data/processed/validation_features.csv")

X = data.drop(columns=["class"])
y = data["class"]

y_true = (y == 2).astype(int)

models = {
    "Version 3": "models/dns_detector_v3.pkl",
    "Version 4": "models/dns_detector_v4.pkl"
}

THRESHOLD = 0.80

print("\n===== VERSION 3 vs VERSION 4 =====")

for name, path in models.items():

    model = joblib.load(path)

    probabilities = model.predict_proba(X)[:, 2]

    predictions = (probabilities >= THRESHOLD).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    print(f"\n{name}")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
