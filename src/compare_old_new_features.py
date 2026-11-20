
import pandas as pd
from collections import Counter

from feature_extraction import extract_features

print("Loading original test domains...")

raw = pd.read_csv(
    "data/raw/test_combined_multiclass.csv.gz"
)

new_features = pd.read_csv(
    "data/processed/test_features.csv"
)

old_bigram = []
old_char_frequency = []

print("Reconstructing old feature logic...")

for domain in raw["domain"]:

    domain = str(domain).lower().strip().strip(".")
    labels = [x for x in domain.split(".") if x]

    old_main_label = (
        labels[-2] if len(labels) >= 2
        else labels[0] if labels
        else ""
    )

    bigrams = [
        old_main_label[i:i + 2]
        for i in range(len(old_main_label) - 1)
    ]

    old_bigram.append(len(set(bigrams)))

    counts = Counter(old_main_label)

    old_char_frequency.append(
        max(counts.values(), default=0)
    )

print("\n===== OLD VS NEW FEATURE COMPARISON =====")

for name, old_values in [
    ("unique_bigram_count", old_bigram),
    ("max_char_frequency", old_char_frequency)
]:

    new_values = new_features[name].to_numpy()

    differences = sum(
        old != new
        for old, new in zip(old_values, new_values)
    )

    print(f"\nFeature: {name}")
    print("Different records:", differences)
    print("Total records:", len(old_values))
    print("Old average:", round(sum(old_values) / len(old_values), 4))
    print("New average:", round(new_values.mean(), 4))

print("\nComparison completed!")
