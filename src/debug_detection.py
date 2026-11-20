
from detection_engine import detect_domain
from feature_extraction import extract_features

domains = [
    "google.com",
    "skqgowoggookqigy.org",
    "pafiifcy.securitytesting.online",
    "3eabapdnggydixon2aaaedrlp55zvgcamdmoehp2zp313hh4j5cibdeylkmqih.yddcaeir2gls.securitytesting.online"
]

for domain in domains:
    print("\n" + "=" * 70)
    print("DOMAIN:", domain)

    features = extract_features(domain)

    print("\nImportant features:")
    for key in [
        "domain_length",
        "entropy",
        "num_digits",
        "subdomain_count",
        "max_subdomain_length",
        "subdomain_entropy",
        "longest_label_ratio",
        "payload_to_domain_ratio"
    ]:
        print(f"{key}: {features[key]}")

    print("\nMODEL RESULT:")
    print(detect_domain(domain))
