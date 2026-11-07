# 🌫️ Air Quality PM2.5 Risk Classification

A Machine Learning classification project that predicts the PM2.5 pollution class (`pm_risk`: Clean, Moderate or High) from nine low-cost gas and weather sensor readings: PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4 and Temperature.

---

## 📌 Problem Statement & Overview

### The Problem
Fine particulate matter (PM2.5) is the pollutant with the strongest link to respiratory and cardiovascular disease, but reference-grade PM2.5 monitors are expensive and sparse. Cheap gas sensors are everywhere, yet they do not measure PM2.5 directly. If the other readings carry enough signal, a model can flag when particulate levels are likely to be unhealthy and trigger a closer measurement or a public warning.

### The Machine Learning Solution
This project turns the measured PM2.5 value into three pollution classes, removes PM2.5 from the inputs so the model cannot copy the answer, and learns to recover the class from the remaining sensors. It compares **Logistic Regression** and **Random Forest** classifiers on a stratified train/test split, with macro-averaged precision, recall, F1 and ROC-AUC to keep every class visible.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────────┐
                 │ dataset/raw_dataset.csv      │
                 │ 500 rows, 10 sensor columns  │
                 └─────────────┬────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │       preprocessing.py       │
                 │ • Drop Duplicate Records     │
                 │ • Clip Negative Sensor Reads │
                 │ • Build pm_risk from PM2.5   │
                 │ • Drop PM2.5 (no leakage)    │
                 └─────────────┬────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │     clean_dataset.csv        │
                 │ 9 features + pm_risk         │
                 └─────────────┬────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │          train.py            │
                 │ • Stratified Train/Test      │
                 │ • StandardScaler (9 features)│
                 │ • Train LR & RF Models       │
                 └─────────────┬────────────────┘
                               │
      ┌────────────────────────┴────────────────────────┐
      ▼                                                 ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│         evaluate.py           │             │          predict.py           │
│ • Accuracy, Macro P/R/F1      │             │ • One Sensor Snapshot         │
│ • Macro ROC-AUC               │             │ • Class + Probability         │
│ • Confusion Matrix Plots      │             │                               │
└───────────────────────────────┘             └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset holds 500 sensor snapshots with no missing values and no duplicate rows.

### Input Features

| Feature | Description | Range / Units |
| :--- | :--- | :--- |
| `PM10` | Coarse particulate matter | 12.9 - 171.6 µg/m³ |
| `NOx` | Nitrogen oxides | 0 - 225.9 ppb |
| `NO2` | Nitrogen dioxide | 0 - 115.9 ppb |
| `SO2` | Sulphur dioxide | 0 - 31.9 ppb |
| `VOCs` | Volatile organic compounds | 27.6 - 231.9 ppb |
| `CO` | Carbon monoxide | 0.37 - 4.80 ppm |
| `CO2` | Carbon dioxide | 366 - 559 ppm |
| `CH4` | Methane | 1.39 - 3.35 ppm |
| `Temperature` | Ambient temperature | 19.6 - 39.8 °C |

A handful of readings (one NOx, one NO2, two SO2) came in slightly negative, which a real sensor cannot report. `preprocessing.py` clips them to zero.

### Target Variable
- **`pm_risk`**: pollution class derived from the measured PM2.5 value.

| Class | PM2.5 range | Rows |
| :--- | :--- | :--- |
| `Clean` | ≤ 35 µg/m³ | 100 |
| `Moderate` | 35 - 60 µg/m³ | 230 |
| `High` | > 60 µg/m³ | 170 |

The 35 µg/m³ cut-off is the EPA 24-hour PM2.5 standard. The other two bands split the rest of the data close to even thirds, so all three classes have enough rows to train on. The strict EPA four-band scale is unusable here because only one row would fall in "Good".

PM2.5 is **dropped from the feature set** after the class is built, since keeping it would leak the answer.

---

## 🤖 Models Trained

1. **Logistic Regression**
   - All nine sensor features standardized with `StandardScaler`.
   - Interpretable linear baseline that shows how far simple additive effects go.

2. **Random Forest Classifier**
   - 300 trees with `balanced_subsample` class weights, which give the smaller `Clean` class a fairer say.
   - Captures non-linear sensor interactions and threshold effects.

Both models use `random_state=42` and share the same stratified 80/20 split.

---

## 📈 Evaluation & Diagnostics

Metrics are macro-averaged so the smaller `Clean` class counts as much as the larger ones:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC** (one-vs-rest, macro)
- **Confusion matrix plots** saved in `outputs/`:
  - `confusion_matrix_lr.png`
  - `confusion_matrix_rf.png`

Results on the 100-row test split:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7200 | 0.6356 | 0.6250 | 0.5965 | 0.8774 |
| Random Forest | 0.7500 | 0.7754 | 0.6630 | 0.6554 | 0.8985 |

Random Forest comes out ahead on accuracy, macro F1 and ROC-AUC, so it is the model to use. `evaluate.py` also prints a per-class report, which shows where the remaining mistakes sit.

### Why some rows are still misclassified
The class comes from PM2.5, but PM2.5 is not one of the inputs. Rebuilding it from the nine sensors with a Random Forest regressor reaches R² ≈ 0.66 with an error of about 14 µg/m³, and 115 of the 500 rows sit between 30 and 40 µg/m³. Around the 35 µg/m³ cut-off the sensor noise is therefore wider than the band itself, so `Clean` and `Moderate` overlap and no classifier can separate every row. On the test split the Random Forest gets the extremes right (94% of `High` and 85% of `Moderate` recalled) and mostly struggles with `Clean`, where 20 test rows are not enough to learn a stable low-pollution signature. This is a limit of the data, not of the code.

---

## 💻 How to Run

```bash
# 1. Clean data, build pm_risk, drop PM2.5
python preprocessing.py

# 2. Train classifiers & export model binaries
python train.py

# 3. Evaluate performance metrics & generate confusion matrix plots
python evaluate.py

# 4. Run a sample sensor-snapshot prediction
python predict.py
```

The scripts resolve `dataset/`, `models/` and `outputs/` from their own location, so the same commands also work when you are one folder above the project.

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

On Windows, `pandas` is pinned below 3.x because newer builds are blocked by Application Control on this machine.

---

## 📁 Directory Structure

```
air-quality-pm25/
├── dataset/
│   ├── raw_dataset.csv          # Raw sensor readings
│   └── clean_dataset.csv        # Clipped features + pm_risk target
├── models/                      # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Visual diagnostic plots
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_rf.png
├── preprocessing.py             # Cleaning, clipping & target creation
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Sensor-snapshot prediction script
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
