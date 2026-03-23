from src.hazop_engine import hazop_analysis
from src.analytics_engine import (
    get_model_metrics,
    get_last_hours_trend,
    get_top_feature_importance,
)
from src.maintenance_engine import recommend_maintenance


TAG_ALIASES = {
    "flow rate": "feed_flow_rate",
    "feed flow": "feed_flow_rate",
    "umpan": "feed_flow_rate",
    "coolant flow": "coolant_flow_rate",
    "pressure": "reactor_pressure",
    "temperatur": "reactor_temp",
    "temperature": "reactor_temp",
    "vibration": "vibration_rms",
    "getaran": "vibration_rms",
    "motor current": "motor_current",
    "arus motor": "motor_current",
    "efficiency": "efficiency_loss_pct",
    "efisiensi": "efficiency_loss_pct",
}

def resolve_tag(prompt: str):
    p = prompt.lower()
    for k, v in TAG_ALIASES.items():
        if k in p:
            return v
    return None

def format_deviation_detail(dev):
    return f"""
Parameter: {dev['parameter']}
Nilai aktual: {dev['value']} {dev['unit']}
Batas warning: {dev['warning_limit']} {dev['unit']}
Batas critical: {dev['critical_limit']} {dev['unit']}
Severity: {dev['severity']}
Gap ke warning: {dev['gap_warning']} {dev['unit']}
Gap ke critical: {dev['gap_critical']} {dev['unit']}
Deviasi: {dev['deviation']}
Penyebab teknis: {dev['cause']}
Konsekuensi operasional: {dev['consequence']}
Rekomendasi engineer: {dev['recommendation']}
""".strip()

def generate_engineer_response(sensor_data, fault_probability, threshold, user_prompt):
    prompt = user_prompt.lower()

    if "akurasi" in prompt or "accuracy" in prompt or "precision" in prompt or "recall" in prompt or "f1" in prompt:
        metrics = get_model_metrics()
        return f"""
Metrik performa model:
- Accuracy: {metrics['accuracy']:.4f}
- Precision fault: {metrics['precision_fault']:.4f}
- Recall fault: {metrics['recall_fault']:.4f}
- F1-score fault: {metrics['f1_fault']:.4f}
- Threshold: {metrics['threshold']:.4f}

Interpretasi engineer:
Accuracy tinggi pada dataset imbalance tidak cukup sebagai indikator utama.
Untuk predictive maintenance, recall fault lebih penting karena terkait kemampuan mendeteksi potensi kegagalan lebih dini.
""".strip()

    if "fitur" in prompt or "feature importance" in prompt or "parameter dominan" in prompt or "paling berpengaruh" in prompt:
        top = get_top_feature_importance()
        text = "Parameter paling dominan dalam model:\n"
        for k, v in top.items():
            text += f"- {k}: {v:.4f}\n"
        text += "\nInterpretasi engineer:\nParameter dengan importance tertinggi adalah kandidat utama untuk monitoring ketat dan prioritas analisis tren."
        return text

    if "12 jam" in prompt or "24 jam" in prompt or "trend" in prompt or "historis" in prompt:
        hours = 12 if "12 jam" in prompt else 24 if "24 jam" in prompt else 12
        tag = resolve_tag(prompt)

        if tag is None:
            return "Saya bisa cek trend historis, tapi sebutkan parameternya. Contoh: flow rate 12 jam terakhir, vibration 24 jam terakhir."

        trend = get_last_hours_trend(tag, hours=hours)
        if trend is None:
            return f"Tag {tag} tidak ditemukan di dataset."

        return f"""
Analisis historis {tag} selama {trend['hours']} jam terakhir:
- Nilai awal: {trend['first']:.3f}
- Nilai akhir: {trend['last']:.3f}
- Minimum: {trend['min']:.3f}
- Maksimum: {trend['max']:.3f}
- Rata-rata: {trend['mean']:.3f}
- Arah trend: {trend['trend']}

Interpretasi engineer:
Perubahan parameter historis perlu dikaitkan dengan deviasi proses, kestabilan kontrol, dan kemungkinan degradasi equipment.
""".strip()

    if "maintenance" in prompt or "diganti" in prompt or "sering dicek" in prompt or "alat apa" in prompt or "equipment" in prompt:
        actions = recommend_maintenance(sensor_data)
        text = "Prioritas maintenance berdasarkan kondisi saat ini:\n"
        for a in actions:
            text += f"""
- Equipment: {a['equipment']}
  Priority: {a['priority']}
  Reason: {a['reason']}
  Action: {a['action']}
"""
        return text

    hazop_result = hazop_analysis(sensor_data)
    prediction = "POTENSI FAILURE TERDETEKSI" if fault_probability >= threshold else "KONDISI RELATIF NORMAL"

    response = f"""
Analisis kondisi plant:
- Probabilitas fault: {fault_probability:.3f}
- Threshold keputusan: {threshold:.3f}
- Status prediksi: {prediction}
- Risk level: {hazop_result['risk_level']}
- Risk score: {hazop_result['risk_score']}

Ringkasan:
{hazop_result['summary']}
"""

    if hazop_result["deviations"]:
        response += "\nDetail deviasi:\n"
        for i, dev in enumerate(hazop_result["deviations"], 1):
            response += f"\n[{i}]\n{format_deviation_detail(dev)}\n"
    else:
        response += "\nTidak ada deviasi signifikan. Operasi masih dalam batas aman."

    return response.strip()