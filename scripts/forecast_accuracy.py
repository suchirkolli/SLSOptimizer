# forecast_accuracy.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths - adjust if your files are elsewhere
ACTUAL_PATH = Path("data/processed/modeling_dataset.parquet")  # contains actuals
PRED_PATH   = Path("outputs/forecast.csv")                    # contains predictions
DIST_REF    = Path("data/geographic/ap_districts_reference.csv")

# Load
actual = pd.read_parquet(ACTUAL_PATH)
pred = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])

# Normalize column names for ease (adapt if yours differ)
# expected: actual has 'timestamp','district_id','peak_demand_mw' or 'demand_mw'
if 'peak_demand_mw' not in actual.columns and 'demand_mw' in actual.columns:
    actual = actual.rename(columns={'demand_mw':'peak_demand_mw'})

# predictions expected column: 'predicted_peak_demand_mw' or similar
if 'predicted_peak_demand_mw' not in pred.columns:
    # try to guess a numeric column that isn't id/timestamp
    for c in pred.columns:
        if c not in ('timestamp','district_id') and np.issubdtype(pred[c].dtype, np.number):
            pred = pred.rename(columns={c:'predicted_peak_demand_mw'})
            break

# Merge on timestamp + district_id
df = pd.merge(actual[['timestamp','district_id','peak_demand_mw']],
              pred[['timestamp','district_id','predicted_peak_demand_mw']],
              on=['timestamp','district_id'], how='inner')

# If no rows, check timestamps alignment or timezones
if df.empty:
    raise SystemExit("Merged DataFrame is empty: check timestamps and district_id values in actuals/predictions")

# Helper metrics
def mape(y_true, y_pred):
    mask = y_true != 0
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100 if mask.any() else np.nan

def mase(y_true, y_pred, seasonal_lag=24):
    # Mean Absolute Scaled Error: scale by in-sample naive forecast MAE (seasonal lag)
    # Use each district's training series naive MAE. Here we approximate using actual series in df.
    # Warning: proper MASE needs the training set; this is an approximate in-sample version.
    n = len(y_true)
    if n <= seasonal_lag:
        return np.nan
    denom = np.mean(np.abs(y_true[seasonal_lag:].values - y_true[:-seasonal_lag].values))
    return np.mean(np.abs(y_true - y_pred)) / denom if denom != 0 else np.nan

# Per-district metrics
grouped = []
for district, g in df.groupby('district_id'):
    y = g['peak_demand_mw'].values
    yhat = g['predicted_peak_demand_mw'].values
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    mape_v = mape(y, yhat)
    mase_v = mase(pd.Series(y), pd.Series(yhat), seasonal_lag=24)
    within5 = np.mean(np.abs(y - yhat) <= 0.05 * np.abs(y)) * 100  # percent within 5%
    grouped.append({
        'district_id': district,
        'n': len(g),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape_pct': mape_v,
        'mase': mase_v,
        'within_5pct_pct': within5
    })
metrics_df = pd.DataFrame(grouped).sort_values('mae')

# Overall metrics (simple aggregate)
overall_mae = mean_absolute_error(df['peak_demand_mw'], df['predicted_peak_demand_mw'])
overall_rmse = np.sqrt(mean_squared_error(df['peak_demand_mw'], df['predicted_peak_demand_mw']))
overall_r2 = r2_score(df['peak_demand_mw'], df['predicted_peak_demand_mw'])
overall_mape = mape(df['peak_demand_mw'], df['predicted_peak_demand_mw'])

# Population-weighted aggregate (helps if districts differ in size)
pop_df = pd.read_csv(DIST_REF) if DIST_REF.exists() else pd.DataFrame()
if not pop_df.empty and 'population' in pop_df.columns:
    pop_map = dict(zip(pop_df['district_id'].astype(str), pop_df['population'].astype(float)))
    df['population'] = df['district_id'].map(pop_map).fillna(0.0)
    # Weighted MAE:
    df['abs_err'] = (df['peak_demand_mw'] - df['predicted_peak_demand_mw']).abs()
    weighted_mae = (df['abs_err'] * df['population']).sum() / df['population'].sum()
else:
    weighted_mae = None

# Print results
print("Per-district (top by MAE):")
print(metrics_df.head(10).to_string(index=False))
print("\nOverall metrics:")
print(f"MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}, R²: {overall_r2:.3f}, MAPE: {overall_mape:.2f}%")
if weighted_mae is not None:
    print(f"Population-weighted MAE: {weighted_mae:.3f}")

# Save per-district metrics
metrics_df.to_csv('outputs/forecast_metrics_by_district.csv', index=False)
print("Saved outputs/forecast_metrics_by_district.csv")