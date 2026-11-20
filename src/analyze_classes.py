
import pandas as pd

# Load original training dataset
df = pd.read_csv(
    "data/raw/train_combined_multiclass.csv.gz"
)

print("\n--- CLASS COUNTS ---")
print(df["class"].value_counts())

print("\n--- BENIGN DOMAIN SAMPLES ---")
print(df[df["class"] == 0]["domain"].head(10).to_string(index=False))

print("\n--- DGA DOMAIN SAMPLES ---")
print(df[df["class"] == 1]["domain"].head(10).to_string(index=False))

print("\n--- DNS TUNNELING DOMAIN SAMPLES ---")
print(df[df["class"] == 2]["domain"].head(20).to_string(index=False))
