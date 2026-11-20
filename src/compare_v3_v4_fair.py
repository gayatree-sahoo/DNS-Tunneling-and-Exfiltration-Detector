
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

print("Loading validation datasets...")

old_df = pd.read_csv(
    "data/processed/validation_features_v3_old.csv"
)

new_df = pd.read_csv(
    "data/processed/validation_features.csv"
)

print("Loading models...")

model_v3 = joblib.load("models/dns_detector_v3.pkl")
model_v4 = joblib.load("models/dns_detector_v4.pkl")

# Check that both datasets contain the same records and labels
if len(old_df) != len(new_df):
    raise ValueError("The validation datasets have different row counts.")

if not old_df["class"].equals(new_df["class"]):
    raise ValueError("The validation labels or row order do not match.")

y_true = old_df["class"].values
y_binary = (y_true == 2).astype(int)

# Use each model's own feature representation
features_v3 = list(model_v3.feature_names_in_)
features_v4 = list(model_v4.feature_names_in_)

X_v3 = old_df[features_v3]
X_v4 = new_df[features_v4]

print("\nValidation records:", len(y_true))
print("V3 feature count:", len(features_v3))
print("V4 feature count:", len(features_v4))

# Generate tunneling probabilities
prob_v3 = model_v3.predict_proba(X_v3)[:, 2]
prob_v4 = model_v4.predict_proba(X_v4)[:, 2]

# Use the same threshold for both models
threshold = 0.80

pred_v3 = (prob_v3 >= threshold).astype(int)
pred_v4 = (prob_v4 >= threshold).astype(int)

def evaluate_model(name, y_true, predictions):
    print(f"\n{name}")
    print("-" * 40)

    print(
        "Precision:",
        round(precision_score(y_true, predictions, zero_division=0), 4)
    )
    print(
        "Recall:",
        round(recall_score(y_true, predictions, zero_division=0), 4)
    )
    print(
        "F1 Score:",
        round(f1_score(y_true, predictions, zero_division=0), 4)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

evaluate_model("VERSION 3", y_binary, pred_v3)
evaluate_model("VERSION 4", y_binary, pred_v4)

print("\nComparison completed.")
