
import joblib
import pandas as pd

# Load test data and model
data = pd.read_csv("data/processed/test_features.csv")
model = joblib.load("models/dns_detector_v3.pkl")

X = data.drop(columns=["class"])
y = data["class"]

# Predict probabilities
probabilities = model.predict_proba(X)

THRESHOLD = 0.80

# Find DGA records incorrectly flagged as tunneling
false_positives = data[
    (y == 1) & (probabilities[:, 2] >= THRESHOLD)
].copy()

false_positives["tunneling_probability"] = probabilities[
    (y == 1) & (probabilities[:, 2] >= THRESHOLD), 2
]

# Find DGA records correctly not flagged as tunneling
correct_dga = data[
    (y == 1) & (probabilities[:, 2] < THRESHOLD)
].copy()

correct_dga["tunneling_probability"] = probabilities[
    (y == 1) & (probabilities[:, 2] < THRESHOLD), 2
]

features = [
    "domain_length",
    "num_digits",
    "num_letters",
    "digit_ratio",
    "entropy",
    "subdomain_count",
    "unique_char_ratio",
    "max_digit_sequence",
    "unique_bigram_count",
    "max_char_frequency"
]

print("False positives:", len(false_positives))
print("Correctly classified DGA:", len(correct_dga))

print("\n===== FALSE POSITIVE FEATURE AVERAGES =====")
print(false_positives[features].mean().round(4))

print("\n===== CORRECT DGA FEATURE AVERAGES =====")
print(correct_dga[features].mean().round(4))

print("\n===== FALSE POSITIVE PROBABILITY =====")
print(false_positives["tunneling_probability"].describe())
