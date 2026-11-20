
import pandas as pd

df = pd.read_csv("data/raw/test_combined_multiclass.csv.gz")

print("SAMPLE TEST DOMAINS")
print("-------------------")

for label in [0, 1, 2]:
    print(f"\nCLASS {label}")

    samples = df[df["class"] == label]["domain"].head(10)

    for domain in samples:
        print(domain)

print("\nDOMAIN LENGTH BY CLASS")
print("----------------------")

df["domain_length"] = df["domain"].astype(str).str.len()

print(df.groupby("class")["domain_length"].describe())
