import pandas as pd
import joblib
from pathlib import Path
from hazop_engine import hazop_analysis

# Load model
model_path = Path("outputs") / "models" / "random_forest_fault_model.pkl"
model = joblib.load(model_path)

# Load data
df = pd.read_csv("data/chemical_process_timeseries.csv")

features = [
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

sample = df[features + ["timestamp"]].copy().ffill().bfill().iloc[-1]

# SIMULASI KONDISI ABNORMAL
sample["vibration_rms"] = 5.0
sample["motor_current"] = 80
sample["reactor_pressure"] = 20

X = pd.DataFrame([sample[features]], columns=features)

prediction = model.predict(X)[0]
prediction_proba = model.predict_proba(X)[0][1]

hazop_result = hazop_analysis(sample.to_dict())

status = "POTENSI FAILURE TERDETEKSI" if prediction == 1 else "KONDISI RELATIF NORMAL"

if prediction_proba > 0.7:
    risk_level = "TINGGI"
elif prediction_proba > 0.3:
    risk_level = "SEDANG"
else:
    risk_level = "RENDAH"

report = f"""
LAPORAN OTOMATIS PREDICTIVE MAINTENANCE DAN INTERPRETASI HAZOP
================================================================

Timestamp pengamatan : {sample['timestamp']}
Probabilitas fault   : {prediction_proba:.4f}
Level risiko         : {risk_level}
Status sistem        : {status}

Ringkasan Kondisi
-----------------
Sistem predictive maintenance mendeteksi kondisi operasi dengan tingkat risiko {risk_level.lower()}.
Analisis dilakukan terhadap parameter kritis seperti temperatur, tekanan, aliran, getaran,
arus motor, dan efisiensi sistem.

Interpretasi HAZOP
------------------
Deviasi:
- """ + "\n- ".join(hazop_result["deviation"]) + """

Kemungkinan penyebab:
- """ + "\n- ".join(hazop_result["causes"]) + """

Konsekuensi potensial:
- """ + "\n- ".join(hazop_result["consequences"]) + """

Rekomendasi tindakan:
- """ + "\n- ".join(hazop_result["recommendations"]) + """

Kesimpulan
----------
Sistem ini mampu memberikan peringatan dini dan rekomendasi teknis berbasis data
untuk mendukung pengambilan keputusan engineer pada predictive maintenance boiler/LNG.
"""

print(report)

report_dir = Path("outputs") / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report_path = report_dir / "hazop_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nLaporan berhasil disimpan di: {report_path}")