
import pandas as pd

validation = pd.read_csv("data/processed/validation_features.csv")
test = pd.read_csv("data/processed/test_features.csv")

# Keep only DGA records (class 1)
validation_dga = validation[validation["class"] == 1]
test_dga = test[test["class"] == 1]

features = [
    "domain_length",
    "num_digits",
    "num_letters",
    "num_hyphens",
    "digit_ratio",
    "entropy",
    "subdomain_count",
    "longest_label_length",
    "unique_char_ratio",
    "max_digit_sequence",
    "unique_bigram_count",
    "max_char_frequency"
]

print("===== DGA FEATURE COMPARISON =====")

for feature in features:
    print(f"\nFeature: {feature}")

    print(
        "Validation median:",
        round(validation_dga[feature].median(), 4)
    )

    print(
        "Test median:",
        round(test_dga[feature].median(), 4)
    )

    print(
        "Validation mean:",
        round(validation_dga[feature].mean(), 4)
    )

    print(
        "Test mean:",
        round(test_dga[feature].mean(), 4)
    )
