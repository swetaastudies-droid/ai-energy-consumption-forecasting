from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "hourly_energy.csv"
PREDICTIONS_FILE = ROOT / "outputs" / "forecasts" / "test_predictions.csv"
FORECAST_FILE = ROOT / "outputs" / "forecasts" / "future_forecast.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"


def plot_test_predictions():
    data = pd.read_csv(PREDICTIONS_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    data = data.tail(24 * 7)
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, data["actual_kw"], label="Actual", linewidth=2)
    for column in ["seasonal_naive_kw", "linear_regression_kw", "random_forest_kw"]:
        if column in data:
            plt.plot(data.index, data[column], label=column.replace("_kw", "").replace("_", " ").title(), alpha=0.8)
    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_future_forecast():
    data = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    plt.figure(figsize=(12, 5))
    plt.plot(data["timestamp"], data["predicted_consumption_kw"], marker="o", markersize=3, label="Forecast")
    plt.title("Future Energy Consumption Forecast")
    plt.xlabel("Future Timestamp")
    plt.ylabel("Predicted Average Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "future_forecast.png", dpi=150)
    plt.close()


def plot_hourly_profile():
    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp"]).dropna(subset=["energy_consumption_kw"])
    data["hour"] = data["timestamp"].dt.hour
    profile = data.groupby("hour")["energy_consumption_kw"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(profile.index, profile.values)
    plt.title("Average Consumption by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Power (kW)")
    plt.xticks(range(24))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hourly_consumption_profile.png", dpi=150)
    plt.close()


def create_all_visualizations():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_test_predictions()
    plot_future_forecast()
    plot_hourly_profile()
    print(f"Graphs saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    create_all_visualizations()
