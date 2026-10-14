import pandas as pd

TARGET_COLUMN = "energy_consumption_kw"
FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_mean_168",
]


def create_supervised_data(series):
    """Create leakage-aware features and target values."""
    features = pd.DataFrame(index=series.index)
    features["hour"] = series.index.hour
    features["dayofweek"] = series.index.dayofweek
    features["month"] = series.index.month
    features["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
    features["lag_1"] = series.shift(1)
    features["lag_24"] = series.shift(24)
    features["lag_168"] = series.shift(168)
    previous = series.shift(1)
    features["rolling_mean_24"] = previous.rolling(24).mean()
    features["rolling_mean_168"] = previous.rolling(168).mean()

    valid = features.notna().all(axis=1) & series.notna()
    return features.loc[valid, FEATURE_COLUMNS], series.loc[valid].rename(TARGET_COLUMN)


def create_next_hour_features(history, next_timestamp):
    """Create features for one future hour using only known history."""
    if len(history) < 168 or history.tail(168).isna().any():
        raise ValueError("At least 168 consecutive valid hourly observations are required.")

    row = {
        "hour": next_timestamp.hour,
        "dayofweek": next_timestamp.dayofweek,
        "month": next_timestamp.month,
        "is_weekend": int(next_timestamp.dayofweek >= 5),
        "lag_1": history.iloc[-1],
        "lag_24": history.iloc[-24],
        "lag_168": history.iloc[-168],
        "rolling_mean_24": history.iloc[-24:].mean(),
        "rolling_mean_168": history.iloc[-168:].mean(),
    }
    return pd.DataFrame([row], index=[next_timestamp], columns=FEATURE_COLUMNS)
