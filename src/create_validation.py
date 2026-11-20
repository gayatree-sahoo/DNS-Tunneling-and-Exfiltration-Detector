
import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading original training dataset...")

df = pd.read_csv("data/raw/train_combined_multiclass.csv.gz")

print("Original dataset shape:", df.shape)

# Split while preserving class proportions
train_df, validation_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)

# Save the new datasets
train_df.to_csv(
    "data/raw/train_split.csv",
    index=False
)

validation_df.to_csv(
    "data/raw/validation.csv",
    index=False
)

print("\nTraining dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)

print("\nValidation class distribution:")
print(validation_df["class"].value_counts())

print("\nValidation dataset created successfully!")
