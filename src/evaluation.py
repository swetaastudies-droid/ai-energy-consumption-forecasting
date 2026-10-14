import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, predicted):
    """Return common regression metrics."""
    return {
        "MAE_kW": float(mean_absolute_error(actual, predicted)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }
