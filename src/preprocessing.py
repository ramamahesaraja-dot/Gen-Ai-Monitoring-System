import pandas as pd

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

def load_data(path):
    df = pd.read_csv(path)
    return df

def create_target(df):
    df["fault_soon"] = (
        (df["time_to_fault_min"] <= 200) &
        (df["time_to_fault_min"].notna())
    ).astype(int)
    return df

def clean_data(df):
    df_model = df[FEATURES + ["fault_soon"]].copy()
    df_model = df_model.ffill().bfill().dropna()
    return df_model

def sample_data(df, n=100000):
    return df.sample(n=n, random_state=42)