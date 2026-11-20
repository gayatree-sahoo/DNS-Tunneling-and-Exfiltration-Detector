
import pandas as pd

files = {
    "Training": "data/raw/train_split.csv",
    "Validation": "data/raw/validation.csv",
    "Test": "data/raw/test_combined_multiclass.csv.gz"
}

for name, path in files.items():
    print(f"\n{name.upper()} DATASET")
    print("-" * 35)

    df = pd.read_csv(path)

    labels = (
        df["domain"]
        .astype(str)
        .str.strip(".")
        .str.split(".")
        .str.len()
    )

    print("Total records:", len(df))
    print("One label:", (labels == 1).sum())
    print("Two labels:", (labels == 2).sum())
    print("More than two labels:", (labels > 2).sum())

    print("\nLabel distribution:")
    print(labels.value_counts().sort_index().to_string())
