import pandas as pd
from src.features import create_supervised_data
from src.evaluation import calculate_metrics


def test_feature_creation_excludes_initial_lags():
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    series = pd.Series(range(200), index=index, dtype=float)
    X, y = create_supervised_data(series)
    assert len(X) == 200 - 168
    assert len(X) == len(y)
    assert "lag_168" in X.columns


def test_metrics_are_reasonable():
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    result = calculate_metrics(actual, predicted)
    assert result["MAE_kW"] == 0
    assert result["RMSE_kW"] == 0
    assert result["R2"] == 1
