import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

def load_plant_data():
    df = pd.read_csv("data/sample_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def load_model_and_threshold():
    model = joblib.load("outputs/models/random_forest_fault_model.pkl")
    threshold_path = Path("outputs/models/best_threshold.txt")

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    return model, threshold

def get_model_metrics():
    df = load_plant_data()

    df["fault_soon"] = (
        (df["time_to_fault_min"] <= 200) &
        (df["time_to_fault_min"].notna())
    ).astype(int)

    df_model = df[FEATURES + ["fault_soon"]].copy()
    df_model = df_model.ffill().bfill().dropna()

    if len(df_model) > 100000:
        df_model = df_model.sample(n=100000, random_state=42)

    X = df_model[FEATURES]
    y = df_model["fault_soon"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model, threshold = load_model_and_threshold()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": report["accuracy"],
        "precision_fault": report["1"]["precision"],
        "recall_fault": report["1"]["recall"],
        "f1_fault": report["1"]["f1-score"],
        "threshold": threshold,
        "confusion_matrix": cm.tolist()
    }

def get_last_hours_data(tag: str, hours: int = 12):
    df = load_plant_data()

    if tag not in df.columns:
        return None

    latest_time = df["timestamp"].max()
    start_time = latest_time - pd.Timedelta(hours=hours)

    subset = df[df["timestamp"] >= start_time][["timestamp", tag]].copy()
    subset[tag] = subset[tag].ffill().bfill()

    return subset

def get_last_hours_trend(tag: str, hours: int = 12):
    subset = get_last_hours_data(tag, hours)
    if subset is None or subset.empty:
        return None

    return {
        "tag": tag,
        "hours": hours,
        "min": float(subset[tag].min()),
        "max": float(subset[tag].max()),
        "mean": float(subset[tag].mean()),
        "first": float(subset[tag].iloc[0]),
        "last": float(subset[tag].iloc[-1]),
        "trend": "Naik" if subset[tag].iloc[-1] > subset[tag].iloc[0] else "Turun/Stabil",
    }

def get_last_hours_summary(tag: str, hours: int = 12):
    subset = get_last_hours_data(tag, hours)
    if subset is None or subset.empty:
        return None

    return {
        "tag": tag,
        "hours": hours,
        "min": float(subset[tag].min()),
        "max": float(subset[tag].max()),
        "mean": float(subset[tag].mean()),
        "first": float(subset[tag].iloc[0]),
        "last": float(subset[tag].iloc[-1]),
        "trend": "Naik" if subset[tag].iloc[-1] > subset[tag].iloc[0] else "Turun/Stabil",
    }

def get_top_feature_importance():
    model, _ = load_model_and_threshold()
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return importance.head(5).to_dict()