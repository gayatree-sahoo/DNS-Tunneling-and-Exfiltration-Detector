
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

print("Loading training data...")

train = pd.read_csv("data/processed/train_features.csv")
X_train = train.drop(columns=["class"])
y_train = train["class"]

print("Training records:", len(train))
print("Features:", len(X_train.columns))
print("\nClass distribution:")
print(y_train.value_counts().sort_index())

print("\nCalculating sample weights...")

weights = compute_sample_weight(
    class_weight={0: 1, 1: 1, 2: 10},
    y=y_train
)

print("\nTraining XGBoost V5...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=weights)

joblib.dump(model, "models/dns_detector_v5.pkl")

print("\nModel V5 trained successfully!")
print("Saved: models/dns_detector_v5.pkl")
