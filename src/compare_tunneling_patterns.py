
import joblib
import pandas as pd

data = pd.read_csv("data/processed/test_features.csv")
model = joblib.load("models/dns_detector_v3.pkl")

X = data.drop(columns=["class"])
y = data["class"]

probabilities = model.predict_proba(X)[:, 2]

THRESHOLD = 0.80

# DGA domains incorrectly flagged as tunneling
false_positives = data[
    (y == 1) & (probabilities >= THRESHOLD)
]

# Actual tunneling domains correctly detected
true_positives = data[
    (y == 2) & (probabilities >= THRESHOLD)
]

# Actual tunneling domains missed by the model
false_negatives = data[
    (y == 2) & (probabilities < THRESHOLD)
]


features = [
    "domain_length",
    "num_digits",
    "num_letters",
    "num_hyphens",
    "num_dots",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "entropy",
    "subdomain_count",
    "longest_label_length",
    "unique_char_ratio",
    "vowel_count",
    "consonant_count",
    "vowel_ratio",
    "consonant_ratio",
    "max_digit_sequence",
    "unique_bigram_count",
    "max_char_frequency"
]


groups = {
    "DGA False Positives": false_positives,
    "Detected Tunneling": true_positives,
    "Missed Tunneling": false_negatives
}

for name, group in groups.items():
    print(f"\n===== {name} =====")
    print("Number of records:", len(group))
    print("\nFeature averages:")
    print(group[features].mean().round(4))

    print("\nFeature medians:")
    print(group[features].median().round(4))
