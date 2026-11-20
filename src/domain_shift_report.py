
import pandas as pd

train = pd.read_csv("data/processed/train_features.csv")
test = pd.read_csv("data/processed/test_features.csv")

features = [
    "domain_length",
    "num_digits",
    "num_letters",
    "num_dots",
    "entropy",
    "subdomain_count",
    "longest_label_length",
    "digit_ratio",
    "unique_char_ratio"
]

print("DOMAIN SHIFT REPORT")
print("=" * 65)

for class_id, class_name in {
    0: "Benign",
    1: "DGA",
    2: "DNS Tunneling"
}.items():

    train_class = train[train["class"] == class_id]
    test_class = test[test["class"] == class_id]

    print(f"\nCLASS {class_id}: {class_name}")
    print("-" * 65)

    print("Training records:", len(train_class))
    print("Test records:", len(test_class))

    comparison = pd.DataFrame({
        "Train Median": train_class[features].median(),
        "Test Median": test_class[features].median(),
        "Train Mean": train_class[features].mean(),
        "Test Mean": test_class[features].mean()
    })

    print(comparison.round(3).to_string())

comparison.to_csv(
    "reports/domain_shift_comparison.csv"
)

print("\nReport saved to reports/domain_shift_comparison.csv")
