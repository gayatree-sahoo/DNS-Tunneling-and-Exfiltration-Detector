
import joblib
import pandas as pd
from feature_extraction import extract_features

MODEL_PATH = "models/dns_detector_v5.pkl"
TUNNEL_THRESHOLD = 0.80

model = joblib.load(MODEL_PATH)


def detect_domain(domain):
    """Analyze a domain and return its predicted category."""

    domain = str(domain).strip()

    if not domain:
        raise ValueError("Please enter a valid domain name.")

    features = extract_features(domain)

    input_data = pd.DataFrame([features])
    input_data = input_data[list(model.feature_names_in_)]

    probabilities = model.predict_proba(input_data)[0]

    tunnel_score = float(probabilities[2])

    if tunnel_score >= TUNNEL_THRESHOLD:
        prediction = 2
    else:
        prediction = int(probabilities[:2].argmax())

    labels = {
        0: "Benign",
        1: "DGA",
        2: "DNS Tunneling"
    }

    return {
        "domain": domain,
        "prediction": labels[prediction],
        "class_id": prediction,
        "tunneling_score": round(tunnel_score, 4),
        "threshold": TUNNEL_THRESHOLD,
        "probabilities": {
            labels[i]: round(float(probabilities[i]), 4)
            for i in range(3)
        }
    }


if __name__ == "__main__":

    samples = [
        "google.com",
        "skqgowoggookqigy.org",
        "pafiifcy.securitytesting.online"
    ]

    for domain in samples:
        result = detect_domain(domain)

        print("\nDomain:", result["domain"])
        print("Prediction:", result["prediction"])
        print("Tunneling score:", result["tunneling_score"])
        print("Class scores:", result["probabilities"])
