
import pandas as pd

old_df = pd.read_csv(
    "data/processed/validation_features_v3_old.csv"
)

new_df = pd.read_csv(
    "data/processed/validation_features.csv"
)

raw_df = pd.read_csv("data/raw/validation.csv")

print("VALIDATION FEATURE INVESTIGATION")
print("--------------------------------")

# Count domains containing different numbers of labels
label_counts = raw_df["domain"].astype(str).apply(
    lambda domain: len(domain.strip(".").split("."))
)

print("\nDomain label distribution:")
print(label_counts.value_counts().sort_index())

# Check feature differences
for feature in ["unique_bigram_count", "max_char_frequency"]:
    differences = old_df[feature] != new_df[feature]

    print(f"\n{feature}")
    print("Changed records:", differences.sum())

    if differences.any():
        print("Old values:")
        print(old_df.loc[differences, feature].head(5).tolist())

        print("New values:")
        print(new_df.loc[differences, feature].head(5).tolist())

print("\nInvestigation completed.")
