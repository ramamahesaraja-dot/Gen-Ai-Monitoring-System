import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score
)

from preprocessing import load_data, create_target, clean_data, FEATURES

# =====================
# CONFIG
# =====================
DATA_PATH = "data/chemical_process_timeseries.csv"
MODEL_PATH = "outputs/models/random_forest_fault_model.pkl"
THRESHOLD_PATH = "outputs/models/best_threshold.txt"

# bikin lebih ringan
SAMPLE_SIZE = 120000
N_ESTIMATORS = 80
RANDOM_STATE = 42

# =====================
# LOAD & PREPROCESS
# =====================
df = load_data(DATA_PATH)
print("Data loaded:", df.shape)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = create_target(df)

print("\nTarget distribution (full data):")
print(df["fault_soon"].value_counts())

df = df.sort_values("timestamp").reset_index(drop=True)

# clean numeric data
df_model = df[["timestamp"] + FEATURES + ["fault_soon"]].copy()
df_model[FEATURES] = df_model[FEATURES].ffill().bfill()
df_model = df_model.dropna(subset=FEATURES + ["fault_soon"])

print("\nAfter cleaning:", df_model.shape)

# =====================
# SAMPLING BIAR RINGAN
# =====================
# ambil sample acak tapi tetap reproducible
if len(df_model) > SAMPLE_SIZE:
    df_model = df_model.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

# urutkan lagi by waktu setelah sampling supaya split tetap time-based
df_model = df_model.sort_values("timestamp").reset_index(drop=True)

print("After sampling:", df_model.shape)

# =====================
# TIME-BASED SPLIT
# =====================
split_idx = int(len(df_model) * 0.8)

train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

X_train = train_df[FEATURES]
y_train = train_df["fault_soon"]

X_test = test_df[FEATURES]
y_test = test_df["fault_soon"]

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("\nTarget distribution (train):")
print(y_train.value_counts())

print("\nTarget distribution (test):")
print(y_test.value_counts())

# =====================
# MODEL
# =====================
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=10,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 4},
    random_state=RANDOM_STATE,
    n_jobs=1  # lebih stabil di laptop; ganti ke -1 kalau sudah kuat
)

model.fit(X_train, y_train)

# =====================
# THRESHOLD TUNING
# =====================
y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

best_threshold = 0.5
best_f1 = -1

for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    if (p + r) == 0:
        continue
    f1 = 2 * p * r / (p + r)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_prob >= best_threshold).astype(int)

# =====================
# EVALUATION
# =====================
cm = confusion_matrix(y_test, y_pred)

print(f"\nBest threshold (F1): {best_threshold:.4f}")
print(f"Best F1 score      : {best_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {roc_auc:.4f}")

pr_auc = auc(recall, precision)
print(f"PR AUC : {pr_auc:.4f}")

feature_importance = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
)

print("\nTop 10 Feature Importance:")
print(feature_importance.head(10))


# =====================
# FEATURE IMPORTANCE
# =====================
importance = pd.Series(model.feature_importances_, index=FEATURES)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot
plt.figure(figsize=(8,5))
importance.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =====================
# SAVE MODEL & THRESHOLD
# =====================
output_dir = Path("outputs/models")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
    f.write(str(best_threshold))

print(f"\n✅ Model saved to: {MODEL_PATH}")
print(f"✅ Threshold saved to: {THRESHOLD_PATH}")