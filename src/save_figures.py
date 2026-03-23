import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/chemical_process_timeseries.csv")

figures_dir = Path("outputs") / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10,4))
df["vibration_rms"].ffill().bfill().iloc[:1000].plot()
plt.title("Vibration Trend")
plt.xlabel("Index")
plt.ylabel("vibration_rms")
plt.tight_layout()
plt.savefig(figures_dir / "vibration_trend.png")
plt.show()

plt.figure(figsize=(8,4))
df["motor_current"].ffill().bfill().iloc[:1000].plot()
plt.title("Motor Current Trend")
plt.xlabel("Index")
plt.ylabel("motor_current")
plt.tight_layout()
plt.savefig(figures_dir / "motor_current_trend.png")
plt.show()