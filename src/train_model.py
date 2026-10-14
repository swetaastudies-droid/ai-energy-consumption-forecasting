import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, create_supervised_data
from src.evaluation import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "hourly_energy.csv"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
FORECAST_DIR = ROOT / "outputs" / "forecasts"


def train_models():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run python -m src.data_preparation first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    series = data[TARGET_COLUMN].astype(float)
    X, y = create_supervised_data(series)

    split = int(len(X) * 0.8)
    if split <= 0 or split >= len(X):
        raise ValueError("Not enough valid rows for a chronological train/test split.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    predictions = pd.DataFrame({"actual_kw": y_test}, index=y_test.index)

    # The baseline is the consumption from 24 hours earlier.
    predictions["seasonal_naive_kw"] = X_test["lag_24"]
    results = [{
        "model": "Seasonal Naive",
        **calculate_metrics(y_test, predictions["seasonal_naive_kw"]),
    }]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        column = name.lower().replace(" ", "_") + "_kw"
        predictions[column] = pred
        results.append({"model": name, **calculate_metrics(y_test, pred)})

    results_df = pd.DataFrame(results).sort_values("RMSE_kW")
    results_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(FORECAST_DIR / "test_predictions.csv", index_label="timestamp")

    # Select the best ML model, retaining the seasonal-naive model as benchmark.
    ml_results = results_df[results_df["model"].isin(models.keys())]
    best_name = ml_results.iloc[0]["model"]
    joblib.dump({
        "model": models[best_name],
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }, MODEL_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "selection_metric": "RMSE_kW",
        "training_start": str(X_train.index.min()),
        "training_end": str(X_train.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "note": "Test period was used for model selection; use a separate validation period for independent final evaluation.",
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"Best ML model: {best_name}")
    return results_df


if __name__ == "__main__":
    train_models()
