
import pandas as pd

df = pd.read_csv("data/processed/train_features.csv")

feature_columns = [
    "domain_length",
    "num_digits",
    "num_letters",
    "entropy",
    "digit_ratio",
    "vowel_ratio",
    "consonant_ratio",
    "max_digit_sequence",
    "unique_bigram_count",
    "max_char_frequency"
]

print("\n--- FEATURE STATISTICS BY CLASS ---")

stats = df.groupby("class")[feature_columns].mean().round(3)

stats.index = ["Benign", "DGA", "DNS Tunneling"]

print(stats.to_string())

print("\n--- MEDIAN DOMAIN LENGTH BY CLASS ---")

print(
    df.groupby("class")["domain_length"]
    .median()
)
