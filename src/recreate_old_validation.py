
import pandas as pd
from old_feature_extraction import extract_features

print("Loading original validation data...")

df = pd.read_csv("data/raw/validation.csv")

print("Validation records:", len(df))
print("Reconstructing original features...")

features = df["domain"].apply(extract_features)

features_df = pd.DataFrame(features.tolist())

result = pd.concat(
    [features_df, df["class"].reset_index(drop=True)],
    axis=1
)

output_path = "data/processed/validation_features_v3_old.csv"

result.to_csv(output_path, index=False)

print("\nOld validation dataset reconstructed!")
print("Saved:", output_path)
print("Dataset shape:", result.shape)
print("Number of features:", features_df.shape[1])
