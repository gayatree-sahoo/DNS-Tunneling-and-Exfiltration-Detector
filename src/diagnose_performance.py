
import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/dns_detector_v3.pkl")

datasets = {
    "Validation": "data/processed/validation_features.csv",
    "Test": "data/processed/test_features.csv"
}

for name, path in datasets.items():
    print(f"\n===== {name.upper()} DATASET =====")

    data = pd.read_csv(path)
    X = data.drop(columns=["class"])
    y = data["class"]

    probabilities = model.predict_proba(X)[:, 2]
    predictions = probabilities >= 0.80

    print("Total records:", len(data))
    print("Actual tunneling records:", (y == 2).sum())
    print("Predicted tunneling records:", predictions.sum())

    print("\nClass-wise probability statistics:")

    for class_id, class_name in [
        (0, "Benign"),
        (1, "DGA"),
        (2, "DNS Tunneling")
    ]:
        scores = probabilities[y == class_id]

        print(f"\n{class_name}:")
        print("Median probability:", round(np.median(scores), 4))
        
        print("90th percentile:", round(np.percentile(scores, 90), 4))
        print("95th percentile:", round(np.percentile(scores, 95), 4))
        print("99th percentile:", round(np.percentile(scores, 99), 4))
