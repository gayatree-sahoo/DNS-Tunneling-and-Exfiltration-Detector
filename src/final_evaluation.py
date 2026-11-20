
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load test dataset
test_data = pd.read_csv("data/processed/test_features.csv")

X_test = test_data.drop(columns=["class"])
y_test = test_data["class"]

# Load trained model
model = joblib.load("models/dns_detector_v3.pkl")

# Get prediction probabilities
probabilities = model.predict_proba(X_test)

# Selected threshold from validation
THRESHOLD = 0.80

# Binary detection: tunneling (2) vs non-tunneling
y_true_binary = (y_test == 2).astype(int)
y_pred_binary = (probabilities[:, 2] >= THRESHOLD).astype(int)

print("\n===== BINARY TUNNELING DETECTION =====")
print("Threshold:", THRESHOLD)
print("Precision:", round(precision_score(y_true_binary, y_pred_binary), 4))
print("Recall:", round(recall_score(y_true_binary, y_pred_binary), 4))
print("F1-score:", round(f1_score(y_true_binary, y_pred_binary), 4))

print("\nBinary Confusion Matrix:")
print(confusion_matrix(y_true_binary, y_pred_binary))

# Three-class predictions
y_pred = probabilities[:, :2].argmax(axis=1)
y_pred[probabilities[:, 2] >= THRESHOLD] = 2

print("\n===== THREE-CLASS CLASSIFICATION =====")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Benign", "DGA", "DNS Tunneling"],
    zero_division=0
))

print("Three-Class Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
