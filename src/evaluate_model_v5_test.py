
import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("Loading V5 model and test data...")

model = joblib.load("models/dns_detector_v5.pkl")
df = pd.read_csv("data/processed/test_features.csv")

X = df.drop(columns=["class"])
y = df["class"]

print("\nTEST DATASET EVALUATION")
print("=" * 50)

predictions = model.predict(X)

print(classification_report(y, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y, predictions))

probabilities = model.predict_proba(X)[:, 2]
threshold = 0.80

binary_predictions = (probabilities >= threshold).astype(int)
binary_actual = (y == 2).astype(int)

print("\nDNS TUNNELING DETECTION")
print("=" * 50)
print("Threshold:", threshold)

print("Precision:", round(
    precision_score(binary_actual, binary_predictions, zero_division=0), 4
))

print("Recall:", round(
    recall_score(binary_actual, binary_predictions, zero_division=0), 4
))

print("F1 Score:", round(
    f1_score(binary_actual, binary_predictions, zero_division=0), 4
))

print("\nTest evaluation completed!")
