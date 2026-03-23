import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from src.hazop_engine import hazop_analysis, PARAMETER_LABELS
from src.chatbot_engine import generate_engineer_response
from src.analytics_engine import get_last_hours_data, get_last_hours_summary

st.set_page_config(page_title="LNG Boiler SCADA AI Assistant", layout="wide")

# =====================
# LOAD MODEL
# =====================
model = joblib.load("outputs/models/random_forest_fault_model.pkl")
with open("outputs/models/best_threshold.txt", "r", encoding="utf-8") as f:
    threshold = float(f.read().strip())

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

st.title("LNG Boiler Industrial SCADA Dashboard")
st.caption("Sistem monitoring, predictive maintenance, interpretasi HAZOP, dan assistant engineer")

# =====================
# SIDEBAR INPUT
# =====================
st.sidebar.header("Input Parameter Operasi")

default_values = {
    "reactor_temp": 320.0,
    "reactor_pressure": 16.0,
    "feed_flow_rate": 100.0,
    "coolant_flow_rate": 100.0,
    "agitator_speed_rpm": 180.0,
    "reaction_rate": 75.0,
    "conversion_rate": 92.0,
    "selectivity": 95.0,
    "yield_pct": 90.0,
    "vibration_rms": 2.0,
    "motor_current": 65.0,
    "power_consumption_kw": 250.0,
    "temp_setpoint": 315.0,
    "pressure_setpoint": 15.0,
    "efficiency_loss_pct": 3.0,
}

input_data = {}
for f in FEATURES:
    input_data[f] = st.sidebar.number_input(
        PARAMETER_LABELS.get(f, f),
        value=float(default_values[f])
    )

trend_tag = st.sidebar.selectbox(
    "Parameter Historian",
    options=FEATURES,
    format_func=lambda x: PARAMETER_LABELS.get(x, x)
)

trend_hours = st.sidebar.selectbox("Rentang Historian", [12, 24], index=0)

X = pd.DataFrame([input_data], columns=FEATURES)

# =====================
# MODEL PREDICTION
# =====================
fault_probability = model.predict_proba(X)[0][1]
prediction = 1 if fault_probability >= threshold else 0

hazop_result = hazop_analysis(input_data)

# =====================
# STATUS BANNER
# =====================
if prediction == 1 or hazop_result["risk_level"] == "HIGH":
    st.error("STATUS SISTEM: PERLU TINDAKAN SEGERA")
elif hazop_result["risk_level"] == "MEDIUM":
    st.warning("STATUS SISTEM: PERLU PEMANTAUAN KETAT")
else:
    st.success("STATUS SISTEM: OPERASI RELATIF NORMAL")

# =====================
# KPI CARDS
# =====================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Probabilitas Fault", f"{fault_probability:.3f}")
with k2:
    st.metric("Threshold Keputusan", f"{threshold:.3f}")
with k3:
    st.metric("Risk Score", hazop_result["risk_score"])
with k4:
    st.metric("Level Risiko", hazop_result["risk_level"])

st.subheader("Ringkasan Engineer")
st.write(hazop_result["summary"])

# =====================
# MAIN PANELS
# =====================
top_left, top_right = st.columns([1.2, 1])

with top_left:
    st.subheader("Alarm / Deviasi Aktif")

    if hazop_result["deviations"]:
        alarm_df = pd.DataFrame(hazop_result["deviations"])[[
            "parameter_label",
            "value",
            "unit",
            "warning_limit",
            "critical_limit",
            "severity",
            "deviation"
        ]].copy()

        alarm_df.columns = [
            "Parameter",
            "Nilai Aktual",
            "Satuan",
            "Batas Warning",
            "Batas Kritis",
            "Severity",
            "Deskripsi Deviasi"
        ]
        st.dataframe(alarm_df, use_container_width=True)
    else:
        st.info("Tidak ada alarm aktif. Semua parameter utama berada dalam batas operasi normal.")

with top_right:
    st.subheader("Analisis Teknis")

    if hazop_result["deviations"]:
        for dev in hazop_result["deviations"]:
            st.markdown(
                f"""
**{dev['deviation']}**  
Parameter **{dev['parameter_label']}** tercatat sebesar **{dev['value']} {dev['unit']}**.  
Batas warning berada pada **{dev['warning_limit']} {dev['unit']}** dan batas kritis pada **{dev['critical_limit']} {dev['unit']}**.  
Kondisi ini diklasifikasikan sebagai **{dev['severity']}**.

**Penyebab teknis:** {dev['cause']}  
**Dampak operasional:** {dev['consequence']}  
**Tindakan yang direkomendasikan:** {dev['recommendation']}
"""
            )
    else:
        st.write(
            "Seluruh parameter utama masih berada dalam rentang operasi yang direkomendasikan. "
            "Belum diperlukan tindakan korektif segera."
        )

# =====================
# HISTORIAN TREND
# =====================
st.subheader("Historian Trend")

trend_df = get_last_hours_data(trend_tag, hours=trend_hours)
trend_summary = get_last_hours_summary(trend_tag, hours=trend_hours)

hist_left, hist_right = st.columns([1.6, 1])

with hist_left:
    if trend_df is not None and not trend_df.empty:
        fig_trend = px.line(
            trend_df,
            x="timestamp",
            y=trend_tag,
            title=f"Trend {PARAMETER_LABELS.get(trend_tag, trend_tag)} - {trend_hours} jam terakhir"
        )
        fig_trend.update_layout(
            xaxis_title="Waktu",
            yaxis_title=PARAMETER_LABELS.get(trend_tag, trend_tag),
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Data trend tidak tersedia.")

with hist_right:
    st.subheader("Ringkasan Historian")
    if trend_summary:
        st.write(f"**Parameter:** {PARAMETER_LABELS.get(trend_summary['tag'], trend_summary['tag'])}")
        st.write(f"**Rentang waktu:** {trend_summary['hours']} jam")
        st.write(f"**Nilai awal:** {trend_summary['first']:.3f}")
        st.write(f"**Nilai akhir:** {trend_summary['last']:.3f}")
        st.write(f"**Minimum:** {trend_summary['min']:.3f}")
        st.write(f"**Maksimum:** {trend_summary['max']:.3f}")
        st.write(f"**Rata-rata:** {trend_summary['mean']:.3f}")
        st.write(f"**Arah tren:** {trend_summary['trend']}")

# =====================
# FEATURE IMPORTANCE
# =====================
st.subheader("Explainability Model")

importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).reset_index()
importance.columns = ["feature", "importance"]
importance["feature"] = importance["feature"].map(PARAMETER_LABELS).fillna(importance["feature"])

fig_importance = px.bar(
    importance,
    x="importance",
    y="feature",
    orientation="h",
    title="Kontribusi Parameter terhadap Keputusan Model",
    text="importance",
)

fig_importance.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
)
fig_importance.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    xaxis_title="Importance Score",
    yaxis_title="Parameter",
    height=500
)
st.plotly_chart(fig_importance, use_container_width=True)

# =====================
# SIMPLE GAUGE
# =====================
st.subheader("Risk Gauge")

gauge_value = min(hazop_result["risk_score"] * 10, 100)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gauge_value,
    title={"text": "Risk Index"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red" if hazop_result["risk_level"] == "HIGH" else "orange" if hazop_result["risk_level"] == "MEDIUM" else "green"},
        "steps": [
            {"range": [0, 30], "color": "#1f7a1f"},
            {"range": [30, 70], "color": "#d9a404"},
            {"range": [70, 100], "color": "#b22222"},
        ],
    }
))
fig_gauge.update_layout(height=300)
st.plotly_chart(fig_gauge, use_container_width=True)

# =====================
# CHATBOT
# =====================
st.subheader("Engineer Chatbot")

user_prompt = st.text_area(
    "Ajukan pertanyaan tentang kondisi plant, histori, metrik model, atau rekomendasi maintenance:",
    value="Jelaskan kondisi boiler ini, level risikonya, deviasi utama, dan rekomendasi tindakan."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Kirim Pertanyaan"):
    response = generate_engineer_response(
        sensor_data=input_data,
        fault_probability=fault_probability,
        threshold=threshold,
        user_prompt=user_prompt,
    )

    st.session_state.chat_history.append(("User", user_prompt))
    st.session_state.chat_history.append(("Assistant", response))

for role, msg in st.session_state.chat_history[-6:]:
    if role == "User":
        st.markdown(f"**User:** {msg}")
    else:
        st.markdown(f"**Assistant:**\n\n{msg}")