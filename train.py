import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "clean_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# 2) Define features and target
X = df.drop(columns=["pm_risk"])
y = df["pm_risk"]

print("\n=== FEATURES ===")
print(X.columns.tolist())

print("\n=== TARGET DISTRIBUTION ===")
print(y.value_counts())

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Scale features for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Train Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_scaled, y_train)

# 6) Train Random Forest
# Class weights give the smaller Clean class a fairer say during training.
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42
)

rf.fit(X_train, y_train)

# 7) Save models and preprocessing objects
joblib.dump(
    lr,
    f"{MODEL_DIR}/logistic_regression.pkl"
)

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    f"{MODEL_DIR}/feature_columns.pkl"
)

print("\nTraining complete.")
print("Saved Logistic Regression, Random Forest, scaler, and feature columns.")
