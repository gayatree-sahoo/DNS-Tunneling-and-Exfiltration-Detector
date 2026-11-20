import pandas as pd

# Load the training dataset
file_path = "data/raw/train_combined_multiclass.csv.gz"
df = pd.read_csv(file_path, compression="gzip")

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 RECORDS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- CLASS DISTRIBUTION ---")
print(df.iloc[:, -1].value_counts())