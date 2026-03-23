import pandas as pd
import joblib
from pathlib import Path
from hazop_engine import hazop_analysis

# =====================
# LOAD MODEL & THRESHOLD
# =====================
model_path = Path("outputs/models/random_forest_fault_model.pkl")
threshold_path = Path("outputs/models/best_threshold.txt")

model = joblib.load(model_path)

with open(threshold_path, "r", encoding="utf-8") as f:
    threshold = float(f.read().strip())

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/chemical_process_timeseries.csv")

FEATURES = [
    "reactor_temp",
    "reactor_pressure",
    "feed_flow_rate",
    "coolant_flow_rate",
    "agitator_speed_rpm",
    "reaction_rate",
    "conversion_rate",
    "selectivity",
    "yield_pct",
    "vibration_rms",
    "motor_current",
    "power_consumption_kw",
    "temp_setpoint",
    "pressure_setpoint",
    "efficiency_loss_pct"
]

sample = df[FEATURES + ["timestamp"]].copy()
sample[FEATURES] = sample[FEATURES].ffill().bfill()
sample = sample.iloc[-1].copy()

# =====================
# SIMULASI ABNORMAL
# =====================
sample["vibration_rms"] = 5.0
sample["motor_current"] = 80
sample["reactor_pressure"] = 20

X = pd.DataFrame([sample[FEATURES]], columns=FEATURES)

# =====================
# PREDICT
# =====================
y_prob = model.predict_proba(X)[0][1]
prediction = 1 if y_prob >= threshold else 0


# =====================
# EXPLAINABILITY (Feature Contribution)
# =====================
importance = pd.Series(model.feature_importances_, index=FEATURES)

# kontribusi sample (approx)
contribution = X.iloc[0] * importance

top_features = contribution.sort_values(ascending=False).head(3)

print("\nFaktor dominan:")
for f in top_features.index:
    print(f"- {f}")
    
hazop_result = hazop_analysis(sample.to_dict())
# =====================
# OUTPUT
# =====================
print("\n===== HASIL ANALISIS PREDICTIVE MAINTENANCE =====")
print(f"Timestamp: {sample['timestamp']}")
print(f"Threshold: {threshold:.4f}")
print(f"Probabilitas fault soon: {y_prob:.4f}")

if prediction == 1:
    print("Status: POTENSI FAILURE TERDETEKSI")
else:
    print("Status: KONDISI RELATIF NORMAL")

print("\nDeviasi HAZOP:")
for i, item in enumerate(hazop_result["deviation"], 1):
    print(f"{i}. {item}")

print("\nKemungkinan Penyebab:")
for i, item in enumerate(hazop_result["causes"], 1):
    print(f"{i}. {item}")

print("\nKonsekuensi:")
for i, item in enumerate(hazop_result["consequences"], 1):
    print(f"{i}. {item}")

print("\nRekomendasi:")
for i, item in enumerate(hazop_result["recommendations"], 1):
    print(f"{i}. {item}")