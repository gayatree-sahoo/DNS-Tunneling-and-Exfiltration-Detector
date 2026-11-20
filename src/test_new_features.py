
from feature_extraction import extract_features

samples = [
    "google.com",
    "askforspecial.com",
    "skqgowoggookqigy.org",
    "pafiifcy.securitytesting.online",
    "3eabapdnggydixon2aaaedrlp55zvgcamdmoehp2zp313hh4j5cibdeylkmqih.yddcaeir2gls.securitytesting.online"
]

for domain in samples:
    features = extract_features(domain)

    print("\nDomain:", domain)
    print("Total features:", len(features))
    print("Domain length:", features["domain_length"])
    print("Entropy:", round(features["entropy"], 3))
    print("Subdomain count:", features["subdomain_count"])
    print("Maximum subdomain length:", features["max_subdomain_length"])
    print("Subdomain entropy:", round(features["subdomain_entropy"], 3))
    print("Payload ratio:", round(features["payload_to_domain_ratio"], 3))

print("\nFeature extraction test completed!")
