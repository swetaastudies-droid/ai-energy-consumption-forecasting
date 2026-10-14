from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM_FILE = ROOT / "outputs" / "simulation" / "scenario_hourly.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"


def generate_simulation_graphs():
    if not SIM_FILE.exists():
        raise FileNotFoundError("Run python -m src.simulation first.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SIM_FILE, parse_dates=["timestamp"])
    threshold = float(data["peak_threshold_kw"].iloc[0])

    plt.figure(figsize=(12, 5))
    plt.plot(data["timestamp"], data["baseline_energy_kwh"], label="Baseline forecast")
    plt.plot(data["timestamp"], data["scenario_energy_kwh"], label="Hypothetical reduced scenario")
    plt.title("Baseline vs Scenario Energy")
    plt.xlabel("Timestamp")
    plt.ylabel("Estimated energy (kWh per hour)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "simulation_energy_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(data["timestamp"], data["baseline_cost_inr"], label="Baseline cost")
    plt.plot(data["timestamp"], data["scenario_cost_inr"], label="Scenario cost")
    plt.title("Estimated Hourly Electricity Cost")
    plt.xlabel("Timestamp")
    plt.ylabel("Estimated cost (INR per hour)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "simulation_cost_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(data["timestamp"], data["predicted_consumption_kw"], label="Predicted power")
    plt.axhline(threshold, linestyle="--", label=f"Threshold: {threshold} kW")
    plt.title("Predicted Power vs Peak Threshold")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted average power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "simulation_peak_threshold.png", dpi=150)
    plt.close()
    print(f"Simulation graphs saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_simulation_graphs()
