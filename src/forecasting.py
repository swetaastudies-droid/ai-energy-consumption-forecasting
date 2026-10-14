from pathlib import Path

import joblib
import pandas as pd

from src.features import TARGET_COLUMN, create_next_hour_features

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "hourly_energy.csv"
MODEL_FILE = ROOT / "models" / "best_model.joblib"
OUTPUT_DIR = ROOT / "outputs" / "forecasts"
FORECAST_FILE = OUTPUT_DIR / "future_forecast.csv"


def generate_forecast(horizon=24):
    if not isinstance(horizon, int) or not 1 <= horizon <= 168:
        raise ValueError("Horizon must be an integer between 1 and 168.")
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Run data preparation and model training first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    series = data[TARGET_COLUMN].astype(float)

    # Find the latest 168 consecutive valid observations.
    valid = series.notna()
    series = series.loc[valid]
    if len(series) < 168:
        raise ValueError("At least 168 valid hourly observations are required.")
    history = series.iloc[-168:].copy()
    if not history.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Latest 168 valid readings are not consecutive hourly observations.")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle["model_name"]

    forecasts = []
    for step in range(1, horizon + 1):
        next_timestamp = history.index[-1] + pd.Timedelta(hours=1)
        X_next = create_next_hour_features(history, next_timestamp)[feature_columns]
        predicted = max(0.0, float(model.predict(X_next)[0]))
        forecasts.append({
            "timestamp": next_timestamp,
            "predicted_consumption_kw": predicted,
            "forecast_step": step,
            "model": model_name,
        })
        history.loc[next_timestamp] = predicted

    result = pd.DataFrame(forecasts)
    result.to_csv(FORECAST_FILE, index=False)
    print(f"Forecast saved to: {FORECAST_FILE}")
    print(result.head())
    return result


if __name__ == "__main__":
    generate_forecast(horizon=24)
