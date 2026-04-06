import pandas as pd
import joblib
from xgboost import XGBRegressor
from services.features import add_time_features, add_lag_features

df = pd.read_parquet("data/processed/modeling_dataset.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = add_time_features(df)
df = add_lag_features(df, target="peak_demand_mw")
df = df.dropna().copy()

drop_cols = [
    "timestamp",
    "peak_demand_mw",
    "Date",
    "district",
    "district_name",
    "district_id",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
y = pd.to_numeric(df["peak_demand_mw"], errors="coerce")

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(0)
y = y.fillna(0)

print(X.dtypes)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X, y)

joblib.dump(model, "models/saved/demand_model_NEW.joblib")
print("done")
print("trained with columns:", list(X.columns))