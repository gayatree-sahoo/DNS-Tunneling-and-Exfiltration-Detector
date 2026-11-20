
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

print("Loading training dataset...")
train_df = pd.read_csv("data/processed/train_features.csv")

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

print("Training records:", len(X_train))
print("Number of features:", X_train.shape[1])

print("Calculating class weights...")
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Initializing XGBoost model...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

print("Training Version 2 model...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    verbose=True
)

joblib.dump(model, "models/dns_detector_v2.pkl")

print("\nVersion 2 training completed!")
print("Model saved: models/dns_detector_v2.pkl")
