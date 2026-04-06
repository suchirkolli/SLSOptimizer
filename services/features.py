import numpy as np

def add_time_features(df):
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_lag_features(df, target="demand"):
    df = df.sort_values("timestamp").copy()

    df["lag_1"] = df[target].shift(1)
    df["lag_7"] = df[target].shift(7)

    df["rolling_mean_7"] = df[target].rolling(7).mean()
    df["rolling_std_7"] = df[target].rolling(7).std()

    return df