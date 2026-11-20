
import math
import re
from collections import Counter


def calculate_entropy(domain):
    if not domain:
        return 0

    probabilities = [
        count / len(domain)
        for count in Counter(domain).values()
    ]

    return -sum(p * math.log2(p) for p in probabilities)


def extract_features(domain):
    domain = str(domain).lower().strip()

    features = {}

    letters = [c for c in domain if c.isalpha()]
    digits = [c for c in domain if c.isdigit()]

    vowels = [c for c in letters if c in "aeiou"]
    consonants = [c for c in letters if c not in "aeiou"]

    labels = domain.split(".")
    main_label = labels[-2] if len(labels) >= 2 else labels[0]

    features["domain_length"] = len(domain)
    features["num_digits"] = len(digits)
    features["num_letters"] = len(letters)
    features["num_hyphens"] = domain.count("-")
    features["num_dots"] = domain.count(".")

    features["digit_ratio"] = len(digits) / len(domain) if domain else 0
    features["letter_ratio"] = len(letters) / len(domain) if domain else 0

    features["special_char_ratio"] = (
        sum(not c.isalnum() for c in domain) / len(domain)
        if domain else 0
    )

    features["entropy"] = calculate_entropy(domain)
    features["subdomain_count"] = max(len(labels) - 2, 0)

    features["longest_label_length"] = max(
        (len(part) for part in labels), default=0
    )

    features["unique_char_ratio"] = (
        len(set(domain)) / len(domain) if domain else 0
    )

    features["vowel_count"] = len(vowels)
    features["consonant_count"] = len(consonants)

    features["vowel_ratio"] = (
        len(vowels) / len(letters) if letters else 0
    )

    features["consonant_ratio"] = (
        len(consonants) / len(letters) if letters else 0
    )

    digit_sequences = re.findall(r"\d+", domain)

    features["max_digit_sequence"] = max(
        (len(sequence) for sequence in digit_sequences),
        default=0
    )

    bigrams = [
        main_label[i:i + 2]
        for i in range(len(main_label) - 1)
    ]

    features["unique_bigram_count"] = len(set(bigrams))

    char_counts = Counter(main_label)

    features["max_char_frequency"] = max(
        char_counts.values(), default=0
    )

    return features
