
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

print("Loading Version 4 training dataset...")

df = pd.read_csv("data/processed/train_features.csv")

X_train = df.drop("class", axis=1)
y_train = df["class"]

print("Training records:", len(X_train))
print("Number of features:", X_train.shape[1])

class_weights = {
    0: 1,
    1: 1,
    2: 10
}

print("\nCalculating class weights...")

sample_weights = compute_sample_weight(
    class_weight=class_weights,
    y=y_train
)

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

print("\nTraining Version 4...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    verbose=True
)

joblib.dump(
    model,
    "models/dns_detector_v4.pkl"
)

print("\nVersion 4 training completed!")
print("Model saved: models/dns_detector_v4.pkl")
print("Features used:", len(model.feature_names_in_))
