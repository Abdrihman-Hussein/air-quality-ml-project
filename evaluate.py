import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "clean_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CLASS_NAMES = ["Clean", "Moderate", "High"]

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# 2) Define features and target
X = df.drop(columns=["pm_risk"])
y = df["pm_risk"]

# 3) Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Load models and scaler
lr = joblib.load(
    f"{MODEL_DIR}/logistic_regression.pkl"
)

rf = joblib.load(
    f"{MODEL_DIR}/random_forest.pkl"
)

scaler = joblib.load(
    f"{MODEL_DIR}/scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

# 5) Predictions
p_lr = lr.predict(X_test_scaled)
p_rf = rf.predict(X_test)

prob_lr = lr.predict_proba(X_test_scaled)
prob_rf = rf.predict_proba(X_test)

# 6) Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro)   : {recall:.4f}")
    print(f"F1 Score (macro) : {f1:.4f}")
    print(f"ROC-AUC (macro)  : {roc_auc:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, digits=3))

evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    p_lr,
    prob_lr
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    p_rf,
    prob_rf
)

# 7) Single-row sanity check
# Pick one unseen row from X_test and predict with both models.
# Use iloc[[i]] to keep it as a DataFrame with column names.

i = 2

x_one = X_test.iloc[[i]]
y_true_one = y_test.iloc[i]

x_one_scaled = scaler.transform(x_one)

p_lr_one = lr.predict(x_one_scaled)[0]
p_rf_one = rf.predict(x_one)[0]

prob_lr_one = lr.predict_proba(x_one_scaled)[0].max()
prob_rf_one = rf.predict_proba(x_one)[0].max()

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"Actual Class : {y_true_one}")
print(f"LR Prediction: {p_lr_one} ({prob_lr_one:.2%})")
print(f"RF Prediction: {p_rf_one} ({prob_rf_one:.2%})")

# 8) Confusion Matrix - Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_lr,
    labels=CLASS_NAMES
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/confusion_matrix_lr.png"
)

plt.close()

# 9) Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_rf,
    labels=CLASS_NAMES
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/confusion_matrix_rf.png"
)

plt.close()

print(f"\nSaved confusion matrix plots to: {OUT_DIR}")
