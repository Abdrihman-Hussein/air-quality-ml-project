import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 1) Load saved objects
lr = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

rf = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# 2) New sensor reading
# One realistic snapshot from a moderately polluted street.
new_data = pd.DataFrame({
    "PM10": [112.0],
    "NOx": [105.0],
    "NO2": [58.0],
    "SO2": [14.5],
    "VOCs": [148.0],
    "CO": [2.6],
    "CO2": [486.0],
    "CH4": [2.3],
    "Temperature": [31.5]
})

# 3) Align columns with the training data
new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4) Prepare Logistic Regression input
new_data_scaled = scaler.transform(new_data)

# 5) Predict
lr_prediction = lr.predict(new_data_scaled)[0]
rf_prediction = rf.predict(new_data)[0]

lr_probability = float(
    lr.predict_proba(new_data_scaled).max()
)

rf_probability = float(
    rf.predict_proba(new_data).max()
)

print("\n=== AIR QUALITY PM2.5 RISK PREDICTION ===")
print(
    f"Logistic Regression: {lr_prediction} "
    f"({lr_probability:.2%})"
)
print(
    f"Random Forest      : {rf_prediction} "
    f"({rf_probability:.2%})"
)
