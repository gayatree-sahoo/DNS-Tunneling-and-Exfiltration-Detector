
import joblib
import pandas as pd
import numpy as np

data = pd.read_csv("data/processed/validation_features.csv")

X = data.drop(columns=["class"])

model_v3 = joblib.load("models/dns_detector_v3.pkl")
model_v4 = joblib.load("models/dns_detector_v4.pkl")

prob_v3 = model_v3.predict_proba(X)[:, 2]
prob_v4 = model_v4.predict_proba(X)[:, 2]

threshold = 0.80

pred_v3 = prob_v3 >= threshold
pred_v4 = prob_v4 >= threshold

print("===== MODEL COMPARISON =====")

print("\nVersion 3 mean probability:",
      round(float(np.mean(prob_v3)), 8))

print("Version 4 mean probability:",
      round(float(np.mean(prob_v4)), 8))

print("\nAverage absolute probability difference:",
      round(float(np.mean(np.abs(prob_v3 - prob_v4))), 8))

print("\nMaximum probability difference:",
      round(float(np.max(np.abs(prob_v3 - prob_v4))), 8))

print("\nDifferent probability predictions:",
      int(np.sum(np.abs(prob_v3 - prob_v4) > 1e-8)))

print("\nDifferent binary classifications:",
      int(np.sum(pred_v3 != pred_v4)))

print("\nVersion 3 detected:",
      int(pred_v3.sum()))

print("Version 4 detected:",
      int(pred_v4.sum()))
