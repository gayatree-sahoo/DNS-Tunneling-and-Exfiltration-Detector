
from feature_extraction import extract_features

# Sample domain names
test_domains = [
    "google.com",
    "amazon.in",
    "xj92kdl82jd93.com",
    "a8f7d9c2b1.example.com"
]

for domain in test_domains:
    print("\nDomain:", domain)

    features = extract_features(domain)

    for feature, value in features.items():
        print(f"{feature}: {value}")
