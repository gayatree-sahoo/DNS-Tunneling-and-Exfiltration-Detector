
import math
import re
from collections import Counter


def calculate_entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0

    counts = Counter(text)
    probabilities = [count / len(text) for count in counts.values()]

    return -sum(p * math.log2(p) for p in probabilities)


def extract_features(domain):
    """Extract lexical and structural features from a DNS domain."""

    domain = str(domain).lower().strip().strip(".")
    labels = [label for label in domain.split(".") if label]

    letters = [c for c in domain if c.isalpha()]
    digits = [c for c in domain if c.isdigit()]

    vowels = [c for c in letters if c in "aeiou"]
    consonants = [c for c in letters if c not in "aeiou"]

    main_label = labels[0] if labels else ""

    # Treat the final two labels as the registered domain and suffix.
    subdomains = labels[:-2] if len(labels) > 2 else []

    payload = "".join(subdomains) if subdomains else main_label

    digit_sequences = re.findall(r"\d+", domain)
    bigrams = [
        main_label[i:i + 2]
        for i in range(len(main_label) - 1)
    ]

    char_counts = Counter(payload)

    features = {
        "domain_length": len(domain),
        "num_digits": len(digits),
        "num_letters": len(letters),
        "num_hyphens": domain.count("-"),
        "num_dots": domain.count("."),
        "digit_ratio": len(digits) / len(domain) if domain else 0,
        "letter_ratio": len(letters) / len(domain) if domain else 0,
        "special_char_ratio": (
            sum(not c.isalnum() for c in domain) / len(domain)
            if domain else 0
        ),
        "entropy": calculate_entropy(domain),
        "subdomain_count": len(subdomains),
        "longest_label_length": max(
            (len(label) for label in labels), default=0
        ),
        "unique_char_ratio": (
            len(set(domain)) / len(domain) if domain else 0
        ),
        "vowel_count": len(vowels),
        "consonant_count": len(consonants),
        "vowel_ratio": len(vowels) / len(letters) if letters else 0,
        "consonant_ratio": (
            len(consonants) / len(letters) if letters else 0
        ),
        "max_digit_sequence": max(
            (len(sequence) for sequence in digit_sequences), default=0
        ),
        "unique_bigram_count": len(set(bigrams)),
        "max_char_frequency": max(char_counts.values(), default=0),

        # Five additional structural features
        "max_subdomain_length": max(
            (len(label) for label in subdomains), default=0
        ),
        "average_subdomain_length": (
            sum(map(len, subdomains)) / len(subdomains)
            if subdomains else 0
        ),
        "subdomain_entropy": calculate_entropy("".join(subdomains)),
        "longest_label_ratio": (
            max((len(label) for label in labels), default=0) / len(domain)
            if domain else 0
        ),
        "payload_to_domain_ratio": (
            len(payload) / len(domain) if domain else 0
        )
    }

    return features
