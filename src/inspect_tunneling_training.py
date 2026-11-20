
import pandas as pd

df = pd.read_csv("data/raw/train_split.csv")

tunnel = df[df["class"] == 2]

print("DNS TUNNELING TRAINING SAMPLES")
print("=" * 50)

print("Total tunneling records:", len(tunnel))

print("\nFirst 30 tunneling domains:")
for domain in tunnel["domain"].head(30):
    print(domain)

print("\nDomain length statistics:")
print(tunnel["domain"].astype(str).str.len().describe())

print("\nUnique domain count:")
print(tunnel["domain"].nunique())

print("\nClass 2 sample label distribution:")
print(
    tunnel["domain"]
    .astype(str)
    .str.split(".")
    .str.len()
    .value_counts()
    .sort_index()
)
