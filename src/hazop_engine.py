# =====================
# LABEL PARAMETER (WAJIB ADA)
# =====================
PARAMETER_LABELS = {
    "reactor_temp": "Temperatur Reaktor",
    "reactor_pressure": "Tekanan Reaktor",
    "feed_flow_rate": "Laju Alir Umpan",
    "coolant_flow_rate": "Laju Alir Pendingin",
    "agitator_speed_rpm": "Kecepatan Agitator",
    "reaction_rate": "Laju Reaksi",
    "conversion_rate": "Konversi",
    "selectivity": "Selektivitas",
    "yield_pct": "Yield Produk",
    "vibration_rms": "Getaran Peralatan",
    "motor_current": "Arus Motor",
    "power_consumption_kw": "Konsumsi Daya",
    "temp_setpoint": "Setpoint Temperatur",
    "pressure_setpoint": "Setpoint Tekanan",
    "efficiency_loss_pct": "Penurunan Efisiensi",
}

def hazop_analysis(data):
    deviations = []
    risk_score = 0

    def add_deviation(
        parameter,
        value,
        unit,
        normal_max,
        warning_limit,
        critical_limit,
        deviation,
        cause,
        consequence,
        recommendation,
    ):
        severity = None
        score = 0

        if value >= critical_limit:
            severity = "HIGH"
            score = 3
        elif value >= warning_limit:
            severity = "MEDIUM"
            score = 1
        else:
            return None

        return {
            "parameter": parameter,
            "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
            "value": float(value),
            "unit": unit,
            "normal_max": normal_max,
            "warning_limit": warning_limit,
            "critical_limit": critical_limit,
            "gap_warning": float(value - warning_limit),
            "gap_critical": float(value - critical_limit),
            "severity": severity,
            "deviation": deviation,
            "cause": cause,
            "consequence": consequence,
            "recommendation": recommendation,
            "score": score,
        }

    checks = [
        add_deviation(
            "vibration_rms",
            data.get("vibration_rms", 0),
            "mm/s",
            2.5, 2.5, 4.5,
            "Getaran tinggi",
            "Terjadi degradasi mekanis seperti bearing wear atau misalignment.",
            "Meningkatkan keausan komponen dan risiko kerusakan rotating equipment.",
            "Lakukan inspeksi bearing, alignment, balancing, dan monitoring getaran."
        ),
        add_deviation(
            "motor_current",
            data.get("motor_current", 0),
            "A",
            70, 70, 85,
            "Arus motor tinggi",
            "Motor bekerja di atas beban normal atau terjadi peningkatan tahanan mekanis.",
            "Risiko overheating, penurunan efisiensi, dan potensi trip.",
            "Periksa beban motor, sistem penggerak, dan kondisi mekanis."
        ),
        add_deviation(
            "reactor_pressure",
            data.get("reactor_pressure", 0),
            "bar",
            18, 18, 22,
            "Tekanan tinggi",
            "Gangguan pada sistem kontrol tekanan atau flow restriction.",
            "Risiko overpressure dan gangguan keselamatan proses.",
            "Periksa control valve, pressure loop, dan relief system."
        ),
        add_deviation(
            "reactor_temp",
            data.get("reactor_temp", 0),
            "°C",
            340, 340, 380,
            "Temperatur tinggi",
            "Pendinginan tidak optimal atau peningkatan heat load.",
            "Dapat menyebabkan thermal stress dan degradasi material.",
            "Evaluasi sistem pendingin dan kinerja heat transfer."
        ),
        add_deviation(
            "efficiency_loss_pct",
            data.get("efficiency_loss_pct", 0),
            "%",
            5, 5, 10,
            "Penurunan efisiensi",
            "Terjadi fouling atau penurunan performa heat transfer.",
            "Meningkatkan konsumsi energi dan biaya operasi.",
            "Lakukan cleaning dan evaluasi performa sistem."
        ),
    ]

    for d in checks:
        if d is not None:
            deviations.append(d)
            risk_score += d["score"]

    if risk_score >= 8:
        risk_level = "HIGH"
    elif risk_score >= 4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if not deviations:
        summary = (
            "Seluruh parameter utama masih berada dalam rentang operasi normal. "
            "Tidak terdapat deviasi yang memerlukan tindakan korektif segera."
        )
    else:
        summary = (
            f"Terdeteksi {len(deviations)} deviasi proses. "
            f"Total risk score sebesar {risk_score} dengan level risiko {risk_level}. "
            f"Perhatian utama perlu diberikan pada parameter yang telah melampaui batas kritis."
        )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "deviations": deviations,
        "summary": summary,
    }