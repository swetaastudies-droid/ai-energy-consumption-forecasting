from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORECAST_FILE = ROOT / "outputs" / "forecasts" / "future_forecast.csv"
OUTPUT_DIR = ROOT / "outputs" / "simulation"


def run_simulation(tariff_per_kwh=8.0, reduction_percent=5.0,
                   peak_threshold_kw=3.0, daily_budget=500.0):
    """Run a hypothetical cost and peak-threshold simulation."""
    if tariff_per_kwh < 0 or daily_budget < 0 or peak_threshold_kw < 0:
        raise ValueError("Tariff, budget, and threshold cannot be negative.")
    if not 0 <= reduction_percent <= 100:
        raise ValueError("Reduction must be between 0 and 100 percent.")
    if not FORECAST_FILE.exists():
        raise FileNotFoundError("Run python -m src.forecasting first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    if data.empty:
        raise ValueError("Forecast file is empty.")
    if data["timestamp"].duplicated().any():
        raise ValueError("Forecast timestamps must be unique.")
    if data["predicted_consumption_kw"].isna().any() or (data["predicted_consumption_kw"] < 0).any():
        raise ValueError("Forecast has missing or negative values.")
    diffs = data.sort_values("timestamp")["timestamp"].diff().dropna()
    if not diffs.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast timestamps must be consecutive hourly intervals.")

    data = data.sort_values("timestamp").copy()
    data["baseline_energy_kwh"] = data["predicted_consumption_kw"]
    data["baseline_cost_inr"] = data["baseline_energy_kwh"] * tariff_per_kwh
    data["scenario_energy_kwh"] = data["baseline_energy_kwh"] * (1 - reduction_percent / 100)
    data["scenario_cost_inr"] = data["scenario_energy_kwh"] * tariff_per_kwh
    data["estimated_cost_difference_inr"] = data["baseline_cost_inr"] - data["scenario_cost_inr"]
    data["peak_threshold_kw"] = peak_threshold_kw
    data["peak_alert"] = data["predicted_consumption_kw"] > peak_threshold_kw
    data.to_csv(OUTPUT_DIR / "scenario_hourly.csv", index=False)

    daily_costs = data.assign(forecast_date=data["timestamp"].dt.date).groupby("forecast_date")["baseline_cost_inr"].sum()
    over_budget = daily_costs > daily_budget
    summary = {
        "scenario_type": "Hypothetical virtual simulation",
        "forecast_hours": int(len(data)),
        "tariff_inr_per_kwh": float(tariff_per_kwh),
        "assumed_reduction_percent": float(reduction_percent),
        "peak_threshold_kw": float(peak_threshold_kw),
        "daily_budget_inr": float(daily_budget),
        "predicted_energy_kwh": round(float(data["baseline_energy_kwh"].sum()), 3),
        "scenario_energy_kwh": round(float(data["scenario_energy_kwh"].sum()), 3),
        "baseline_estimated_cost_inr": round(float(data["baseline_cost_inr"].sum()), 2),
        "scenario_estimated_cost_inr": round(float(data["scenario_cost_inr"].sum()), 2),
        "estimated_cost_difference_inr": round(float(data["estimated_cost_difference_inr"].sum()), 2),
        "hours_above_peak_threshold": int(data["peak_alert"].sum()),
        "days_over_daily_budget": int(over_budget.sum()),
        "budget_exceeded_dates": [str(day) for day in over_budget[over_budget].index],
        "note": "Hypothetical estimates, not verified real-world savings.",
    }
    with open(OUTPUT_DIR / "scenario_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Virtual simulation summary:")
    print(json.dumps(summary, indent=2))
    return data, summary


if __name__ == "__main__":
    run_simulation()
