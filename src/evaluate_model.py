
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading test dataset...")
test_df = pd.read_csv("data/processed/test_features.csv")

X_test = test_df.drop("class", axis=1)
y_test = test_df["class"]

print("Loading Version 3 model...")
model = joblib.load("models/dns_detector_v3.pkl")

print("Making predictions...")
y_pred = model.predict(X_test)

print("\n--- VERSION 3 ACCURACY ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Benign", "DGA", "DNS Tunneling"],
    zero_division=0
))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
