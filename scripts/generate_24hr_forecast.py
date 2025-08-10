# scripts/generate_24h_forecast.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.predictor import load_demand_models, get_latest_snapshot, predict_demand_on_snapshot
from pathlib import Path
import pandas as pd

def main():
    models = load_demand_models()
    snap = get_latest_snapshot()
    if snap.empty:
        raise SystemExit("Latest snapshot empty — run validate_data.py / ensure modeling_dataset.parquet exists.")
    latest_ts = snap['timestamp'].max()
    future_ts = pd.date_range(start=latest_ts + pd.Timedelta(hours=1), periods=24, freq='H')
    rows = []
    for ts in future_ts:
        tmp = snap.copy()
        tmp['timestamp'] = ts
        rows.append(tmp)
    future_df = pd.concat(rows, ignore_index=True)
    forecasts = predict_demand_on_snapshot(models, future_df)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    # Ensure column name stays predicted_peak_demand_mw (dashboard expects this)
    # If predict_demand_on_snapshot returned 'predicted_peak_demand_mw', keep it; otherwise rename.
    if 'predicted_peak_demand_mw' not in forecasts.columns and 'predicted_peak_demand' in forecasts.columns:
        forecasts = forecasts.rename(columns={'predicted_peak_demand':'predicted_peak_demand_mw'})
    forecasts.to_csv(out / "forecast_24h.csv", index=False)
    print("Saved outputs/forecast_24h.csv with", len(forecasts), "rows")

if __name__ == "__main__":
    main()