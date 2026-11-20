
import pandas as pd

df = pd.read_csv("data/raw/train_split.csv")

tunnel = df[df["class"] == 2].copy()

tunnel["domain_length"] = tunnel["domain"].astype(str).str.len()

print("DNS TUNNELING DOMAIN LENGTH ANALYSIS")
print("=" * 45)

bins = [0, 10, 20, 30, 40, 50, 60, 100]

tunnel["length_group"] = pd.cut(
    tunnel["domain_length"],
    bins=bins
)

print("\nRecords by domain length:")
print(tunnel["length_group"].value_counts().sort_index())

print("\nClass 2 domain length statistics:")
print(tunnel["domain_length"].describe())

print("\nLongest 15 tunneling domains:")
print(
    tunnel.nlargest(15, "domain_length")["domain"].to_string(index=False)
)
