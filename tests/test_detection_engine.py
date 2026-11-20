
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_extraction import extract_features
from detection_engine import detect_domain


def test_feature_extraction():
    features = extract_features("google.com")

    assert isinstance(features, dict)
    assert len(features) == 24
    assert features["domain_length"] == 10


def test_empty_domain_features():
    features = extract_features("")

    assert isinstance(features, dict)
    assert len(features) == 24


def test_model_prediction_structure():
    result = detect_domain("google.com")

    assert result["prediction"] in [
        "Benign",
        "DGA",
        "DNS Tunneling"
    ]

    assert 0 <= result["tunneling_score"] <= 1


def test_prediction_probabilities():
    result = detect_domain("google.com")

    probabilities = result["probabilities"]

    assert set(probabilities.keys()) == {
        "Benign",
        "DGA",
        "DNS Tunneling"
    }

    assert abs(sum(probabilities.values()) - 1) < 0.001


def test_empty_domain_rejected():
    with pytest.raises(ValueError):
        detect_domain("")


def test_long_tunneling_domain():
    domain = (
        "3eabapdnggydixon2aaaedrlp55zvgcamdmoehp2zp313hh4j5cibdeylkmqih."
        "yddcaeir2gls.securitytesting.online"
    )

    result = detect_domain(domain)

    assert result["prediction"] == "DNS Tunneling"
