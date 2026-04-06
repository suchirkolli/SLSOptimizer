import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.predictor import load_modeling_history


def main():
    history = load_modeling_history()
    rows = []
    periods = 24
    for district_id, hist in history.groupby("district_id"):
        hist = hist.sort_values("timestamp").copy()
        last_ts = pd.Timestamp(hist["timestamp"].iloc[-1])
        series = hist["peak_demand_mw"].astype(float).tolist()
        district_name = hist["district_name"].iloc[-1] if "district_name" in hist.columns else district_id
        for step in range(1, periods + 1):
            ts = last_ts + pd.Timedelta(days=step)
            lag1 = series[-1]
            lag7 = series[-7] if len(series) >= 7 else np.mean(series)
            roll7 = float(np.mean(series[-7:]))
            dow_mask = hist["timestamp"].dt.dayofweek == ts.dayofweek
            same_dow_avg = float(hist.loc[dow_mask, "peak_demand_mw"].tail(8).mean()) if dow_mask.any() else roll7
            pred = 0.45 * lag1 + 0.30 * lag7 + 0.20 * roll7 + 0.05 * same_dow_avg
            series.append(pred)
            rows.append({
                "district_id": district_id,
                "district_name": district_name,
                "timestamp": ts,
                "predicted_peak_demand_mw": float(pred),
            })

    out_df = pd.DataFrame(rows)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    out_df.to_csv(out / "forecast_24h.csv", index=False)
    print(f"Saved {out / 'forecast_24h.csv'} with {len(out_df)} rows across {out_df['district_id'].nunique()} districts")
    print("Generated a 24-step daily forecast because the source dataset is daily, not hourly.")


if __name__ == "__main__":
    main()
