
import pandas as pd

files = {
    "Original Training": "data/raw/train_combined_multiclass.csv.gz",
    "Test": "data/raw/test_combined_multiclass.csv.gz"
}

for name, path in files.items():
    print(f"\n{name.upper()}")
    print("=" * 45)

    df = pd.read_csv(path)

    labels = (
        df["domain"]
        .astype(str)
        .str.strip(".")
        .str.split(".")
        .str.len()
    )

    print("Total records:", len(df))

    print("\nDomain label distribution:")
    print(labels.value_counts().sort_index())

    print("\nClass distribution:")
    print(df["class"].value_counts().sort_index())

    print("\nClass-wise domain label distribution:")
    print(pd.crosstab(df["class"], labels))
