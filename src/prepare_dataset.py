
import os
import pandas as pd
from feature_extraction import extract_features


def process_dataset(input_path, output_path):
    print(f"\nProcessing: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    print("Total records:", len(df))
    print("Extracting 24 features...")

    features = pd.DataFrame(
        df["domain"].apply(extract_features).tolist()
    )

    # Add target class
    features["class"] = df["class"].values

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    features.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Processed shape:", features.shape)
    print("Number of features:", len(features.columns) - 1)


if __name__ == "__main__":

    process_dataset(
        "data/raw/train_split.csv",
        "data/processed/train_features.csv"
    )

    process_dataset(
        "data/raw/validation.csv",
        "data/processed/validation_features.csv"
    )

    process_dataset(
        "data/raw/test_combined_multiclass.csv.gz",
        "data/processed/test_features.csv"
    )

    print("\nAll datasets processed successfully!")
