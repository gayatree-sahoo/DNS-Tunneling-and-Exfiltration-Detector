
import pandas as pd
import numpy as np

# Load the original and regenerated test features
old = pd.read_csv("reports/version2/test_features.csv")
new = pd.read_csv("data/processed/test_features.csv")

print("===== FEATURE DATA COMPARISON =====")

print("Old dataset shape:", old.shape)
print("New dataset shape:", new.shape)

features = [
    "consonant_count",
    "vowel_count",
    "unique_bigram_count",
    "max_char_frequency"
]

for feature in features:
    if feature not in old.columns or feature not in new.columns:
        print(f"\n{feature}: Missing from one dataset")
        continue

    differences = np.sum(
        old[feature].to_numpy() != new[feature].to_numpy()
    )

    print(f"\nFeature: {feature}")
    print("Different records:", differences)

    print("Old mean:", round(old[feature].mean(), 4))
    print("New mean:", round(new[feature].mean(), 4))
