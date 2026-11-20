
import joblib
import pandas as pd

# Load Version 3 model
model = joblib.load("models/dns_detector_v3.pkl")

# Extract feature importance
importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- VERSION 3 FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))

print("\nTotal features:", len(importance))
