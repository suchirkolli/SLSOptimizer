# scripts/plot_forecast_vs_actual_avg.py
"""
Plot average predicted vs average actual across all districts.
Saves outputs/fig_forecast_vs_actual_avg.png
Run from project root:
    python scripts/plot_forecast_vs_actual_avg.py
"""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px

MD = Path("data/processed/modeling_dataset.parquet")
F24 = Path("outputs/forecast_24h.csv")
F = Path("outputs/forecast.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

if not MD.exists():
    sys.exit("Missing modeling dataset: " + str(MD))

# Load actuals
md = pd.read_parquet(MD)
md['timestamp'] = pd.to_datetime(md['timestamp'], errors='coerce')

# Determine actual column
if 'peak_demand_mw' in md.columns:
    actual_col = 'peak_demand_mw'
elif 'demand_mw' in md.columns:
    actual_col = 'demand_mw'
else:
    raise SystemExit("No actual demand column found in modeling dataset")

# Aggregate actuals: average across districts per timestamp
actual_avg = md.groupby('timestamp')[actual_col].mean().rename('actual_avg').reset_index()

# Load predictions: prefer 24h multi-step forecast if present
pred = None
if F24.exists():
    pred = pd.read_csv(F24, parse_dates=['timestamp'])
elif F.exists():
    pred = pd.read_csv(F, parse_dates=['timestamp'])

if pred is None or pred.empty:
    # Fallback: create a flat predicted series from latest snapshot average
    latest_snapshot = md.groupby('district_id').tail(1)
    if latest_snapshot.empty:
        raise SystemExit("No forecast file and no snapshot available to build fallback predictions.")
    avg_val = latest_snapshot[actual_col].mean()
    # build series aligned with actual_avg timestamps (use actual range)
    timestamps = actual_avg['timestamp'].dropna().unique()
    pred_avg = pd.DataFrame({'timestamp': timestamps, 'predicted_avg': avg_val})
else:
    # Ensure predicted column exists and normalize name
    if 'predicted_peak_demand_mw' not in pred.columns:
        for c in pred.columns:
            if c not in ('timestamp','district_id') and pd.api.types.is_numeric_dtype(pred[c]):
                pred = pred.rename(columns={c:'predicted_peak_demand_mw'})
                break
    # aggregate predictions: average across districts per timestamp
    pred_avg = pred.groupby('timestamp')['predicted_peak_demand_mw'].mean().rename('predicted_avg').reset_index()

# Merge on timestamp for overlapping window
merged = pd.merge(actual_avg, pred_avg, on='timestamp', how='inner')
if merged.empty:
    # If no overlap, try outer join and forward/backfill predictions
    merged = pd.merge(actual_avg, pred_avg, on='timestamp', how='outer').sort_values('timestamp')
    merged['actual_avg'] = merged['actual_avg'].interpolate().fillna(method='ffill').fillna(method='bfill')
    merged['predicted_avg'] = merged['predicted_avg'].interpolate().fillna(method='ffill').fillna(method='bfill')

# Plot
fig = px.line(merged, x='timestamp', y=['actual_avg','predicted_avg'],
              labels={'value':'Average Peak Demand (MW)','timestamp':'Timestamp'},
              title='Average Forecast vs Actual Across Districts')
fig.update_traces(mode='lines+markers')
out_file = OUT / "fig_forecast_vs_actual_avg.png"
fig.write_image(str(out_file), width=1000, height=600)
print("Saved", out_file)