import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "raw_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "dataset", "clean_dataset.csv")

# 1) Load dataset
df = pd.read_csv(CSV_PATH)

# 2) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

# 3) Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"\nDuplicates removed: {before - after}")

# 4) Clip negative sensor readings to zero
# A few gas sensors report small negative values that are not physically possible.
sensor_cols = [
    
    "PM2.5",
    "PM10",
    "NOx",
    "NO2",
    "SO2",
    "VOCs",
    "CO",
    "CO2",
    "CH4",
    "Temperature"
]

negative_before = int((df[sensor_cols] < 0).sum().sum())

df[sensor_cols] = df[sensor_cols].clip(lower=0)

print(f"\nNegative sensor readings clipped: {negative_before}")

# 5) Build the pollution class from PM2.5
# Bands: Clean up to 35, Moderate up to 60, High above 60.
df["pm_risk"] = pd.cut(
    df["PM2.5"],
    bins=[0, 35, 60, 1000],
    labels=["Clean", "Moderate", "High"]
)

print("\n=== TARGET DISTRIBUTION ===")
print(df["pm_risk"].value_counts())

# 6) Drop PM2.5 so the model cannot copy the answer
# The class comes from PM2.5, so it must not stay in the feature set.
df = df.drop(columns=["PM2.5"])

# 7) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== FINAL TARGET DISTRIBUTION ===")
print(df["pm_risk"].value_counts())

# 8) Save clean dataset
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
